#!/usr/bin/env python
"""第6号 ja-error-detect-v1 パイロット学習・dev 評価スクリプト（両ベース共通）。

文字単位の誤りスパン（O / B-ERR / I-ERR）をトークン分類で学習し、
dev.jsonl 上で全体・種別別・正常文誤検出率（1,000文字あたり件数・文単位FPR）・
トークナイザ理論上限（Oracle）を一括算出して eval_pilot.json に出力する。

注意: 封印テスト（sealed-test.jsonl）には一切アクセスしない。

使い方:
  python code/train_pilot_err.py --base sbintuitions/modernbert-ja-130m \
      --train data/pilot/train.jsonl --dev data/pilot/dev.jsonl \
      --out runs/pilot-modernbert
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import error_injector as ei  # noqa: E402
import offsets_util as ou  # noqa: E402


class ErrDataset(Dataset):
    """誤り検出用トークン分類データセット。"""

    def __init__(
        self,
        path: Path,
        tok,
        max_len: int = 256,
        verify_n: int = 200,
    ) -> None:
        self.rows: List[dict] = []
        self.raw_rows: List[dict] = []
        self.offsets_list: List[List[Tuple[int, int]]] = []
        self.truncated_count = 0

        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                r = json.loads(line)
                enc, offsets = ou.encode_with_offsets(tok, r["text"], max_len=max_len)
                if idx < verify_n:
                    ref = tok(r["text"], truncation=True, max_length=max_len)
                    assert enc["input_ids"] == ref["input_ids"], (
                        f"input_ids 不一致 (row={idx}, id={r.get('id')}): "
                        f"{enc['input_ids'][:20]} != {ref['input_ids'][:20]}"
                    )
                # 切り詰め有無の確認（元テキスト末尾までオフセットが届いているか）
                non_empty_ends = [b for a, b in offsets if b > a]
                if r["text"] and (not non_empty_ends or max(non_empty_ends) < len(r["text"].rstrip())):
                    self.truncated_count += 1

                labels = ou.align_spans_to_tokens(r["text"], offsets, r["spans"])
                enc["labels"] = labels
                self.rows.append(enc)
                self.raw_rows.append(r)
                self.offsets_list.append(offsets)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def collate_fn(batch: List[dict], pad_id: int) -> dict:
    """動的パディング付きミニバッチ生成。"""
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, labels = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad_n = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad_n)
        attn.append(b["attention_mask"] + [0] * pad_n)
        labels.append(b["labels"] + [-100] * pad_n)
    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = torch.tensor(
            [b["token_type_ids"] + [0] * (maxlen - len(b["token_type_ids"])) for b in batch],
            dtype=torch.long,
        )
    return out


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def _match_overlap_1to1(
    gold_spans: List[dict], pred_spans: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """重なり幅最大の貪欲 1対1 マッチング（(gold_idx, pred_idx) の組を返す）。"""
    pairs = []
    for gi, g in enumerate(gold_spans):
        gs, ge = int(g["start"]), int(g["end"])
        for pi, (ps, pe) in enumerate(pred_spans):
            ov = min(ge, pe) - max(gs, ps)
            if ov > 0:
                pairs.append((ov, gi, pi))
    pairs.sort(key=lambda x: x[0], reverse=True)
    used_g, used_p = set(), set()
    matched: List[Tuple[int, int]] = []
    for _, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            matched.append((gi, pi))
    return matched


def evaluate_predictions(
    raw_rows: List[dict],
    pred_spans_list: List[List[Tuple[int, int]]],
    char_preds_list: List[List[int]],
) -> dict:
    """全体・種別別・正常文誤検出の各種指標を一括算出する。"""
    ex_tp = ex_fp = ex_fn = 0
    ov_tp = ov_fp = ov_fn = 0
    ch_tp = ch_fp = ch_fn = 0

    pt_ex = {t: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for t in ei.ERROR_TYPES}
    pt_ov = {t: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for t in ei.ERROR_TYPES}

    clean_sents = 0
    clean_chars = 0
    clean_fp_spans = 0
    clean_sents_with_fp = 0
    clean_fp_chars = 0
    clean_fp_examples: List[dict] = []

    for row, pred_spans, char_preds in zip(raw_rows, pred_spans_list, char_preds_list):
        gold_spans = row["spans"]
        text = row["text"]

        # 文字単位二値 (ERR vs O)
        for g_lb, p_lb in zip(row["char_labels"], char_preds):
            g_err = g_lb != 0
            p_err = p_lb != 0
            if g_err and p_err:
                ch_tp += 1
            elif (not g_err) and p_err:
                ch_fp += 1
            elif g_err and (not p_err):
                ch_fn += 1

        # 正常文誤検出
        if not row["has_error"]:
            clean_sents += 1
            clean_chars += len(text)
            if pred_spans:
                clean_fp_spans += len(pred_spans)
                clean_sents_with_fp += 1
                clean_fp_chars += sum(1 for c in char_preds if c != 0)
                if len(clean_fp_examples) < 10:
                    clean_fp_examples.append(
                        {
                            "id": row.get("id"),
                            "text": text,
                            "pred_spans": [
                                {"start": a, "end": b, "substr": text[a:b]}
                                for a, b in pred_spans
                            ],
                        }
                    )

        # 完全一致 (Exact Span)
        gold_map = {(int(g["start"]), int(g["end"])): g["type"] for g in gold_spans}
        pred_set = set(pred_spans)
        matched_exact = set(gold_map.keys()) & pred_set
        unmatched_pred_ex = len(pred_set - matched_exact)

        ex_tp += len(matched_exact)
        ex_fp += unmatched_pred_ex
        ex_fn += len(gold_map) - len(matched_exact)

        present_types = {g["type"] for g in gold_spans}
        for (gs, ge), gtype in gold_map.items():
            pt_ex[gtype]["support"] += 1
            if (gs, ge) in matched_exact:
                pt_ex[gtype]["tp"] += 1
            else:
                pt_ex[gtype]["fn"] += 1
        for t in present_types:
            pt_ex[t]["fp"] += unmatched_pred_ex

        # 重なり一致 (Overlap 1-to-1)
        matched_ov_pairs = _match_overlap_1to1(gold_spans, pred_spans)
        matched_g_idxs = {gi for gi, _ in matched_ov_pairs}
        unmatched_pred_ov = len(pred_spans) - len(matched_ov_pairs)

        ov_tp += len(matched_ov_pairs)
        ov_fp += unmatched_pred_ov
        ov_fn += len(gold_spans) - len(matched_ov_pairs)

        for gi, g in enumerate(gold_spans):
            gt = g["type"]
            pt_ov[gt]["support"] += 1
            if gi in matched_g_idxs:
                pt_ov[gt]["tp"] += 1
            else:
                pt_ov[gt]["fn"] += 1
        for t in present_types:
            pt_ov[t]["fp"] += unmatched_pred_ov

    per_type_exact = {}
    per_type_overlap = {}
    for t in ei.ERROR_TYPES:
        d_ex = _prf(pt_ex[t]["tp"], pt_ex[t]["fp"], pt_ex[t]["fn"])
        d_ex["support"] = pt_ex[t]["support"]
        per_type_exact[t] = d_ex

        d_ov = _prf(pt_ov[t]["tp"], pt_ov[t]["fp"], pt_ov[t]["fn"])
        d_ov["support"] = pt_ov[t]["support"]
        per_type_overlap[t] = d_ov

    clean_fpr = {
        "clean_sentences": clean_sents,
        "clean_chars": clean_chars,
        "fp_spans": clean_fp_spans,
        "fp_per_1000_chars": round(clean_fp_spans / (clean_chars / 1000.0), 4)
        if clean_chars
        else 0.0,
        "clean_sentences_with_fp": clean_sents_with_fp,
        "sentence_fpr": round(clean_sents_with_fp / clean_sents, 4)
        if clean_sents
        else 0.0,
        "fp_chars": clean_fp_chars,
        "fp_chars_per_1000_chars": round(clean_fp_chars / (clean_chars / 1000.0), 4)
        if clean_chars
        else 0.0,
    }

    return {
        "exact_span": _prf(ex_tp, ex_fp, ex_fn),
        "overlap_span": _prf(ov_tp, ov_fp, ov_fn),
        "char_binary": _prf(ch_tp, ch_fp, ch_fn),
        "per_type_exact": per_type_exact,
        "per_type_overlap": per_type_overlap,
        "clean_fpr": clean_fpr,
        "clean_fp_examples": clean_fp_examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 封印テスト誤読込防止ガード
    for p_str in (args.train, args.dev):
        if "sealed" in Path(p_str).name.lower():
            raise RuntimeError(f"封印テストファイルは入力禁止です: {p_str}")

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== パイロット学習開始: base={args.base} ===")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base,
        num_labels=len(ou.LABELS),
        id2label=ou.ID2LABEL,
        label2id=ou.LABEL2ID,
    )
    max_pos = getattr(model.config, "max_position_embeddings", args.max_len)
    max_len = min(args.max_len, int(max_pos))

    train_ds = ErrDataset(Path(args.train), tok, max_len=max_len, verify_n=200)
    dev_ds = ErrDataset(Path(args.dev), tok, max_len=max_len, verify_n=999999)
    print(
        f"  データ準備完了: train={len(train_ds)} (trunc={train_ds.truncated_count}), "
        f"dev={len(dev_ds)} (trunc={dev_ds.truncated_count}), max_len={max_len}"
    )

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=lambda b: collate_fn(b, pad_id),
    )

    t0 = time.time()
    train_res = trainer.train()
    train_sec = time.time() - t0

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))

    # train.log の保存と NaN / inf チェック
    has_nan = False
    log_lines = []
    for entry in trainer.state.log_history:
        for k, v in entry.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                has_nan = True
        log_lines.append(json.dumps(entry, ensure_ascii=False))
    (out_dir / "train.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    # dev 推論
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    pred_spans_all: List[List[Tuple[int, int]]] = []
    char_preds_all: List[List[int]] = []
    oracle_spans_all: List[List[Tuple[int, int]]] = []
    oracle_char_all: List[List[int]] = []

    eval_bs = 32
    total_chars = sum(len(r["text"]) for r in dev_ds.raw_rows)
    t_eval0 = time.time()
    with torch.no_grad():
        for start_i in range(0, len(dev_ds), eval_bs):
            batch_items = [dev_ds[j] for j in range(start_i, min(start_i + eval_bs, len(dev_ds)))]
            batch_tensors = collate_fn(batch_items, pad_id)
            kw = {
                "input_ids": batch_tensors["input_ids"].to(device),
                "attention_mask": batch_tensors["attention_mask"].to(device),
            }
            if "token_type_ids" in batch_tensors:
                kw["token_type_ids"] = batch_tensors["token_type_ids"].to(device)
            logits = model(**kw).logits.argmax(dim=-1).cpu().tolist()

            for k, item in enumerate(batch_items):
                g_idx = start_i + k
                seq_len = len(item["input_ids"])
                pred_ids = logits[k][:seq_len]
                raw_text = dev_ds.raw_rows[g_idx]["text"]
                offsets = dev_ds.offsets_list[g_idx]

                ps, cp = ou.decode_token_preds_to_spans(raw_text, offsets, pred_ids)
                pred_spans_all.append(ps)
                char_preds_all.append(cp)

                oracle_ids = [0 if lb == -100 else lb for lb in item["labels"]]
                os_spans, o_cp = ou.decode_token_preds_to_spans(raw_text, offsets, oracle_ids)
                oracle_spans_all.append(os_spans)
                oracle_char_all.append(o_cp)

    eval_sec = time.time() - t_eval0
    metrics = evaluate_predictions(dev_ds.raw_rows, pred_spans_all, char_preds_all)
    oracle_metrics = evaluate_predictions(dev_ds.raw_rows, oracle_spans_all, oracle_char_all)

    final_dir_bytes = sum(f.stat().st_size for f in final_dir.rglob("*") if f.is_file())
    st_path = final_dir / "model.safetensors"
    safetensors_bytes = st_path.stat().st_size if st_path.exists() else 0
    model_params = sum(p.numel() for p in model.parameters())

    report = {
        "base": args.base,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.bs,
        "max_len": max_len,
        "fp16": bool(torch.cuda.is_available()),
        "train_rows": len(train_ds),
        "dev_rows": len(dev_ds),
        "train_truncated": train_ds.truncated_count,
        "dev_truncated": dev_ds.truncated_count,
        "train_sec": round(train_sec, 2),
        "train_loss": round(float(train_res.training_loss), 5),
        "has_nan_in_train_log": has_nan,
        "eval_sec": round(eval_sec, 3),
        "eval_chars": total_chars,
        "eval_chars_per_sec": round(total_chars / eval_sec, 1) if eval_sec > 0 else 0.0,
        "eval_ms_per_sent": round(eval_sec * 1000.0 / len(dev_ds), 3),
        "model_params": int(model_params),
        "final_dir_bytes": int(final_dir_bytes),
        "model_safetensors_bytes": int(safetensors_bytes),
        "exact_span": metrics["exact_span"],
        "overlap_span": metrics["overlap_span"],
        "char_binary": metrics["char_binary"],
        "per_type_exact": metrics["per_type_exact"],
        "per_type_overlap": metrics["per_type_overlap"],
        "clean_fpr": metrics["clean_fpr"],
        "tokenizer_oracle": {
            "exact_span": oracle_metrics["exact_span"],
            "overlap_span": oracle_metrics["overlap_span"],
            "per_type_exact": oracle_metrics["per_type_exact"],
        },
        "clean_fp_examples": metrics["clean_fp_examples"],
    }

    (out_dir / "eval_pilot.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "base": report["base"],
        "train_sec": report["train_sec"],
        "train_loss": report["train_loss"],
        "exact_span": report["exact_span"],
        "overlap_span": report["overlap_span"],
        "clean_fpr": report["clean_fpr"],
        "oracle_exact_f1": report["tokenizer_oracle"]["exact_span"]["f1"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
