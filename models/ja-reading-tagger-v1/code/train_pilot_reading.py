#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 パイロット学習・評価スクリプト（両ベース共通・同一条件）。

- モデル: AutoModelForTokenClassification (sbintuitions/modernbert-ja-130m / tohoku-nlp/bert-base-japanese-v3)
- 損失関数: Candidate-Restricted Span Cross-Entropy（多読スパンの有効辞書候補のみ競争）
- 評価: dev.jsonl (2,000件) および homograph-eval.jsonl (1,000件) に対する
        リークゼロ推論（入力 text のみからスパンと候補を生成）＋境界非依存・文単位読み正解率
- 注意: sealed-test.jsonl には一切アクセスしない。
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_reading as er  # noqa: E402
import offsets_util as ou  # noqa: E402
import phonology as ph  # noqa: E402
from span_extractor import SpanExtractor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_label_space(
    train_rows: List[dict], extractor: SpanExtractor
) -> Tuple[Dict[str, int], List[str], List[List[Tuple[int, int, str, List[str], str]]]]:
    """train.jsonl と統合辞書のみからラベル空間 (key|reading -> id) と各文の監督スパンを構築する。"""
    label_set = set()
    train_targets_per_row: List[List[Tuple[int, int, str, List[str], str]]] = []

    for row in train_rows:
        text = row["text"]
        gold_spans = row["spans"]
        ext_spans = extractor.extract_spans(text)
        row_targets: List[Tuple[int, int, str, List[str], str]] = []

        for sp in ext_spans:
            cands = sp["candidates"]
            if len(cands) < 2:
                continue
            # ゴールドからこのスパン位置に厳密対応する読みを取得
            got = er.extract_subspan_reading(gold_spans, sp["start"], sp["end"])
            if got is None:
                continue
            gold_rk = ph.normalize_kana(got[0])
            if gold_rk not in cands:
                continue
            key = sp["key"]
            for c in cands:
                label_set.add(f"{key}|{c}")
            row_targets.append((sp["start"], sp["end"], key, cands, gold_rk))

        train_targets_per_row.append(row_targets)

    id2label = sorted(label_set)
    label2id = {lb: i for i, lb in enumerate(id2label)}
    return label2id, id2label, train_targets_per_row


class ReadingTrainDataset(Dataset):
    """トークン列＋スパン監督情報（候補クラス ID・正解位置）保持データセット。"""

    def __init__(
        self,
        rows: List[dict],
        targets_per_row: List[List[Tuple[int, int, str, List[str], str]]],
        label2id: Dict[str, int],
        tok,
        max_len: int = 256,
        verify_n: int = 200,
    ) -> None:
        self.items: List[dict] = []
        self.truncated_count = 0
        self.total_supervised_spans = 0

        for idx, (row, tlist) in enumerate(zip(rows, targets_per_row)):
            text = row["text"]
            enc, offsets = ou.encode_with_offsets(tok, text, max_len=max_len)
            if idx < verify_n:
                ref = tok(text, truncation=True, max_length=max_len)
                assert enc["input_ids"] == ref["input_ids"], (
                    f"input_ids 不一致 (row={idx}, id={row.get('id')})"
                )
            non_empty_ends = [b for a, b in offsets if b > a]
            if text and (not non_empty_ends or max(non_empty_ends) < len(text.rstrip())):
                self.truncated_count += 1

            span_items = []
            for st, ed, key, cands, gold_rk in tlist:
                tok_idxs = ou.span_to_token_indices(offsets, st, ed)
                if not tok_idxs:
                    continue
                cand_ids = [label2id[f"{key}|{c}"] for c in cands if f"{key}|{c}" in label2id]
                gold_id = label2id.get(f"{key}|{gold_rk}")
                if gold_id is None or len(cand_ids) < 2 or gold_id not in cand_ids:
                    continue
                gold_pos = cand_ids.index(gold_id)
                span_items.append((tok_idxs, cand_ids, gold_pos))
                self.total_supervised_spans += 1

            enc["span_items"] = span_items
            self.items.append(enc)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def collate_train(batch: List[dict], pad_id: int) -> dict:
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, span_items = [], [], []
    has_tti = "token_type_ids" in batch[0]
    ttids = []
    for b in batch:
        pad_n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad_n)
        attn.append(b["attention_mask"] + [0] * pad_n)
        if has_tti:
            ttids.append(b["token_type_ids"] + [0] * pad_n)
        span_items.append(b["span_items"])
    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "span_items": span_items,
    }
    if has_tti:
        out["token_type_ids"] = torch.tensor(ttids, dtype=torch.long)
    return out


def compute_span_ce_loss(
    logits: torch.Tensor, span_items_batch: List[List[Tuple[List[int], List[int], int]]]
) -> Optional[torch.Tensor]:
    """Candidate-Restricted Span Cross-Entropy (fp32 上で -1e9 マスクを安全に適用)。"""
    # バッチ内の全監督スパンの表現を集める
    span_vecs: List[torch.Tensor] = []
    cand_id_lists: List[List[int]] = []
    gold_positions: List[int] = []

    logits_f = logits.float()
    for bi, s_items in enumerate(span_items_batch):
        for tok_idxs, cand_ids, gold_pos in s_items:
            vec = logits_f[bi, tok_idxs].mean(dim=0)  # [C]
            span_vecs.append(vec)
            cand_id_lists.append(cand_ids)
            gold_positions.append(gold_pos)

    if not span_vecs:
        return None

    stacked = torch.stack(span_vecs, dim=0)  # [N_sp, C]
    max_k = max(len(c) for c in cand_id_lists)
    n_sp = len(span_vecs)
    cand_logits = torch.full(
        (n_sp, max_k), -1e9, dtype=torch.float32, device=logits.device
    )
    for si, cids in enumerate(cand_id_lists):
        cand_logits[si, : len(cids)] = stacked[si, cids]

    targets = torch.tensor(gold_positions, dtype=torch.long, device=logits.device)
    return F.cross_entropy(cand_logits, targets)


@torch.no_grad()
def predict_eval_split(
    model,
    tok,
    rows: List[dict],
    extractor: SpanExtractor,
    label2id: Dict[str, int],
    max_len: int,
    batch_size: int,
    device: str,
    verify_all_offsets: bool = True,
) -> Tuple[List[List[dict]], float, int]:
    """リークゼロ推論（入力 text のみからスパン抽出し、モデル出力ロジットで多読候補を決定）。"""
    model.eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    truncated_count = 0

    # 先に全件のエンコードとスパン抽出（オフセット整合検証含む）
    prepared = []
    for idx, row in enumerate(rows):
        txt = row["text"]
        enc, offsets = ou.encode_with_offsets(tok, txt, max_len=max_len)
        if verify_all_offsets:
            ref = tok(txt, truncation=True, max_length=max_len)
            assert enc["input_ids"] == ref["input_ids"], (
                f"eval input_ids 不一致 (row={idx}, id={row.get('id')})"
            )
        non_empty_ends = [b for a, b in offsets if b > a]
        if txt and (not non_empty_ends or max(non_empty_ends) < len(txt.rstrip())):
            truncated_count += 1
        sp_list = extractor.extract_spans(txt)
        prepared.append((enc, offsets, sp_list))

    t0 = time.time()
    preds_all: List[List[dict]] = []
    for b_start in range(0, len(prepared), batch_size):
        chunk = prepared[b_start : b_start + batch_size]
        maxlen = max(len(x[0]["input_ids"]) for x in chunk)
        input_ids, attn, ttids = [], [], []
        has_tti = "token_type_ids" in chunk[0][0]
        for enc, _, _ in chunk:
            pad_n = maxlen - len(enc["input_ids"])
            input_ids.append(enc["input_ids"] + [pad_id] * pad_n)
            attn.append(enc["attention_mask"] + [0] * pad_n)
            if has_tti:
                ttids.append(enc["token_type_ids"] + [0] * pad_n)

        model_in = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attn, dtype=torch.long, device=device),
        }
        if has_tti:
            model_in["token_type_ids"] = torch.tensor(ttids, dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(**model_in).logits  # [B, T, C]

        for bi, (_, offsets, sp_list) in enumerate(chunk):
            sent_logits = logits[bi]
            out_spans: List[dict] = []
            for sp in sp_list:
                cands = sp["candidates"]
                chosen_rk = sp["default_reading"]
                if len(cands) >= 2:
                    cand_label_ids = []
                    for rk in cands:
                        lid = label2id.get(f"{sp['key']}|{rk}")
                        if lid is not None:
                            cand_label_ids.append((rk, lid))
                    if len(cand_label_ids) >= 2:
                        tok_idxs = ou.span_to_token_indices(offsets, sp["start"], sp["end"])
                        if tok_idxs:
                            span_vec = sent_logits[tok_idxs].mean(dim=0)
                            best_rk, _ = max(
                                cand_label_ids,
                                key=lambda item: float(span_vec[item[1]].item()),
                            )
                            chosen_rk = best_rk

                out_spans.append(
                    {
                        "start": sp["start"],
                        "end": sp["end"],
                        "surface": sp["surface"],
                        "reading": ph.kata_to_hira(chosen_rk),
                        "reading_kata": chosen_rk,
                        "pron": extractor.to_pron(chosen_rk, surface=sp["surface"]),
                    }
                )
            preds_all.append(out_spans)

    eval_sec = time.time() - t0
    return preds_all, eval_sec, truncated_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--train", type=str, default=str(ROOT / "data" / "pilot" / "train.jsonl"))
    parser.add_argument("--dev", type=str, default=str(ROOT / "data" / "pilot" / "dev.jsonl"))
    parser.add_argument(
        "--homograph-eval",
        type=str,
        default=str(ROOT / "data" / "pilot" / "homograph-eval.jsonl"),
    )
    parser.add_argument("--master-dir", type=str, default=str(ROOT / "data" / "master"))
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    final_dir = out_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    extractor = SpanExtractor(Path(args.master_dir))
    train_rows = load_jsonl(Path(args.train))
    dev_rows = load_jsonl(Path(args.dev))
    homo_rows = load_jsonl(Path(args.homograph_eval))

    # train のみからラベル空間を構築
    label2id, id2label, train_targets = build_label_space(train_rows, extractor)
    print(f"Base={args.base} | label_space={len(id2label)} classes")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base, num_labels=len(id2label)
    )
    max_pos = getattr(model.config, "max_position_embeddings", args.max_len)
    eff_max_len = min(args.max_len, int(max_pos))
    model.to(device)

    train_ds = ReadingTrainDataset(
        train_rows, train_targets, label2id, tok, max_len=eff_max_len, verify_n=200
    )
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_train(b, pad_id=pad_id),
        generator=torch.Generator().manual_seed(args.seed),
    )

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    total_steps = max(1, int(math.ceil(len(loader) * args.epochs)))
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # 学習開始
    model.train()
    train_losses: List[float] = []
    step_logs: List[dict] = []
    has_nan = False
    global_step = 0
    t_train_start = time.time()

    for ep in range(int(math.ceil(args.epochs))):
        for batch in loader:
            if global_step >= total_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            model_in = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if "token_type_ids" in batch:
                model_in["token_type_ids"] = batch["token_type_ids"].to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(**model_in).logits
                loss = compute_span_ce_loss(logits, batch["span_items"])

            if loss is None:
                global_step += 1
                scheduler.step()
                continue

            loss_val = float(loss.item())
            if math.isnan(loss_val) or math.isinf(loss_val):
                has_nan = True
                raise RuntimeError(f"NaN/Inf loss detected at step {global_step}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss_val)
            global_step += 1
            if global_step in (50, 250, 500, 750, 1000, total_steps) or global_step % 250 == 0:
                avg_recent = sum(train_losses[-50:]) / max(1, len(train_losses[-50:]))
                step_logs.append({"step": global_step, "loss_avg50": round(avg_recent, 5)})
                print(f"[step {global_step}/{total_steps}] loss_avg50={avg_recent:.5f}")

    train_sec = time.time() - t_train_start

    # モデルとラベルマップ保存
    model.save_pretrained(str(final_dir))
    tok.save_pretrained(str(final_dir))
    with open(final_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False)

    # サイズ・パラメータ集計
    n_params = sum(p.numel() for p in model.parameters())
    final_dir_bytes = sum(p.stat().st_size for p in final_dir.rglob("*") if p.is_file())
    safetensors_path = final_dir / "model.safetensors"
    safetensors_bytes = safetensors_path.stat().st_size if safetensors_path.exists() else 0

    # 評価: dev.jsonl と homograph-eval.jsonl
    dev_preds, dev_sec, dev_trunc = predict_eval_split(
        model, tok, dev_rows, extractor, label2id, eff_max_len, 32, device, verify_all_offsets=True
    )
    dev_metrics = er.evaluate_predictions(dev_rows, dev_preds, eval_sec=dev_sec)

    homo_preds, homo_sec, homo_trunc = predict_eval_split(
        model, tok, homo_rows, extractor, label2id, eff_max_len, 32, device, verify_all_offsets=True
    )
    homo_metrics = er.evaluate_predictions(homo_rows, homo_preds, eval_sec=homo_sec)

    eval_out = {
        "base_model": args.base,
        "tokenizer_is_fast": bool(tok.is_fast),
        "vocab_size": int(tok.vocab_size),
        "effective_max_len": eff_max_len,
        "label_classes": len(id2label),
        "train_rows": len(train_rows),
        "train_supervised_spans": train_ds.total_supervised_spans,
        "train_truncated": train_ds.truncated_count,
        "dev_truncated": dev_trunc,
        "homograph_eval_truncated": homo_trunc,
        "hyperparams": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": eff_max_len,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "fp16": (device == "cuda"),
            "seed": args.seed,
        },
        "train_sec": round(train_sec, 2),
        "samples_per_sec": round(len(train_rows) * args.epochs / max(1e-6, train_sec), 2),
        "total_steps": global_step,
        "train_loss_mean": round(sum(train_losses) / max(1, len(train_losses)), 5),
        "step_logs": step_logs,
        "has_nan_in_train_log": has_nan,
        "model_params": n_params,
        "final_dir_bytes": final_dir_bytes,
        "model_safetensors_bytes": safetensors_bytes,
        "dev": dev_metrics,
        "homograph_eval": homo_metrics,
    }
    with open(out_dir / "eval_pilot.json", "w", encoding="utf-8") as f:
        json.dump(eval_out, f, ensure_ascii=False, indent=2)

    print(
        f"DONE {args.base} | train={train_sec:.1f}s | "
        f"dev_sent_read={dev_metrics['sentence_reading_acc']:.4f} | "
        f"homo_sent_read={homo_metrics['sentence_reading_acc']:.4f} | "
        f"homo_target={homo_metrics['target_homograph_acc']}"
    )


if __name__ == "__main__":
    main()
