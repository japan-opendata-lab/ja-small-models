#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 本番 v2 学習・dev 収束＆閾値キャリブレーションスクリプト。

対象ベースモデル: sbintuitions/modernbert-ja-130m (勝者)
主な仕様 (v2):
- ラベル空間: デフォルトで同形異音語カタログ 135 語 (HOMOGRAPH_CATALOG) に純化（extra_lex ノイズ排除）
- 2層クラス不均衡是正:
  1) 学習バッチ内オーバーサンプリング（Qwen3 合成文＋青空の明示ルビ少数派読み文を 3x 投入）
  2) スパン単位の穏やかな逆頻度重み付け (min(4.0, (max_cnt / cnt)^0.35))
- dev (data/prod/dev.jsonl, 5,000件) のみで fp32 softmax 信頼度閾値 tau をスイープ較正し、
  runs/prod-modernbert/best/calibration.json に保存して最良エポックを決定する
- 注意: 本番 sealed-test.jsonl には一切アクセスしない。
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
import homograph_catalog as hc  # noqa: E402
import offsets_util as ou  # noqa: E402
import phonology as ph  # noqa: E402
from span_extractor import SpanExtractor  # noqa: E402
import train_pilot_reading as tpr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TAU_GRID = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def build_refined_label_space(
    train_rows: List[dict],
    extractor: SpanExtractor,
    min_extra_freq: int = 4,
    min_extra_distinct_reads: int = 2,
    catalog_only: bool = True,
) -> Tuple[
    Dict[str, int],
    List[str],
    List[List[Tuple[int, int, str, List[str], str, str]]],
    dict,
]:
    """train.jsonl と統合辞書のみからラベル空間と各文の監督スパン (st, ed, key, cands, gold_rk, def_rk) を構築する。"""
    catalog_keys: Set[str] = {item["key"] for item in hc.HOMOGRAPH_CATALOG}

    raw_targets_per_row: List[List[Tuple[int, int, str, List[str], str, str]]] = []
    key_total_cnt: Counter = Counter()
    key_seen_reads: Dict[str, Set[str]] = defaultdict(set)

    for row in train_rows:
        text = row["text"]
        gold_spans = row["spans"]
        ext_spans = extractor.extract_spans(text)
        row_t: List[Tuple[int, int, str, List[str], str, str]] = []
        for sp in ext_spans:
            cands = sp["candidates"]
            if len(cands) < 2:
                continue
            got = er.extract_subspan_reading(gold_spans, sp["start"], sp["end"])
            if got is None:
                continue
            gold_rk = ph.normalize_kana(got[0])
            if gold_rk not in cands:
                continue
            key = sp["key"]
            key_total_cnt[key] += 1
            key_seen_reads[key].add(gold_rk)
            row_t.append((sp["start"], sp["end"], key, cands, gold_rk, sp["default_reading"]))
        raw_targets_per_row.append(row_t)

    if catalog_only:
        kept_keys = set(catalog_keys)
    else:
        kept_keys = set()
        for k, cnt in key_total_cnt.items():
            if k in catalog_keys:
                kept_keys.add(k)
            elif cnt >= min_extra_freq and len(key_seen_reads[k]) >= min_extra_distinct_reads:
                kept_keys.add(k)

    label_set: Set[str] = set()
    # カタログ / 採用キーの全辞書候補をあらかじめ登録（低頻度ターゲットでも推論時に欠落しない）
    for k in kept_keys:
        if k in extractor.multi_readings:
            for c in extractor.multi_readings[k].get("candidates", []):
                label_set.add(f"{k}|{c}")

    filtered_targets: List[List[Tuple[int, int, str, List[str], str, str]]] = []
    for row_t in raw_targets_per_row:
        kept_t = [t for t in row_t if t[2] in kept_keys]
        for _, _, key, cands, _, _ in kept_t:
            for c in cands:
                label_set.add(f"{key}|{c}")
        filtered_targets.append(kept_t)

    id2label = sorted(label_set)
    label2id = {lb: i for i, lb in enumerate(id2label)}
    meta = {
        "catalog_only": catalog_only,
        "total_multi_keys_seen_in_train": len(key_total_cnt),
        "kept_multi_keys": len(kept_keys),
        "kept_catalog_keys": len(kept_keys & catalog_keys),
        "kept_extra_keys": len(kept_keys - catalog_keys),
        "label_classes": len(id2label),
    }
    return label2id, id2label, filtered_targets, meta


class WeightedReadingTrainDataset(Dataset):
    """文オーバーサンプリング＋スパン単位逆頻度重み付け対応の学習データセット。"""

    def __init__(
        self,
        active_rows: List[dict],
        active_targets: List[List[Tuple[int, int, str, List[str], str, str]]],
        label2id: Dict[str, int],
        tok,
        max_len: int = 256,
        upsample_factor: int = 3,
        weight_power: float = 0.35,
        max_span_weight: float = 4.0,
        verify_n: int = 200,
    ) -> None:
        self.items: List[dict] = []
        self.truncated_count = 0
        self.total_supervised_spans_unique = 0
        self.total_supervised_spans_effective = 0
        self.upsampled_synth_rows = 0
        self.upsampled_aozora_minority_rows = 0

        # 第1段階: 各アクティブ文を1回だけトークナイズ＆スパン解決し、倍率を決める
        base_records: List[Tuple[dict, List[Tuple[List[int], List[int], int, str, str]], int]] = []
        upsampled_kr_counts: Counter = Counter()

        for idx, (row, tlist) in enumerate(zip(active_rows, active_targets)):
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

            raw_spans: List[Tuple[List[int], List[int], int, str, str]] = []
            has_minority_aozora = False
            for st, ed, key, cands, gold_rk, def_rk in tlist:
                tok_idxs = ou.span_to_token_indices(offsets, st, ed)
                if not tok_idxs:
                    continue
                cand_ids = [label2id[f"{key}|{c}"] for c in cands if f"{key}|{c}" in label2id]
                gold_id = label2id.get(f"{key}|{gold_rk}")
                if gold_id is None or len(cand_ids) < 2 or gold_id not in cand_ids:
                    continue
                gold_pos = cand_ids.index(gold_id)
                raw_spans.append((tok_idxs, cand_ids, gold_pos, key, gold_rk))
                if gold_rk != def_rk:
                    has_minority_aozora = True

            if not raw_spans:
                continue

            is_synth = row.get("source") == "qwen3_synth"
            if is_synth or (row.get("source") == "aozora" and has_minority_aozora):
                rep = upsample_factor
                if is_synth:
                    self.upsampled_synth_rows += 1
                else:
                    self.upsampled_aozora_minority_rows += 1
            else:
                rep = 1

            self.total_supervised_spans_unique += len(raw_spans)
            self.total_supervised_spans_effective += len(raw_spans) * rep
            for _, _, _, key, gold_rk in raw_spans:
                upsampled_kr_counts[(key, gold_rk)] += rep

            base_records.append((enc, raw_spans, rep))

        # 第2段階: アップサンプリング後の実効頻度でスパン逆頻度重みを算出して展開
        max_cnt_per_key: Dict[str, int] = defaultdict(int)
        for (k, _), c in upsampled_kr_counts.items():
            if c > max_cnt_per_key[k]:
                max_cnt_per_key[k] = c

        weight_by_kr: Dict[Tuple[str, str], float] = {}
        for (k, rk), c in upsampled_kr_counts.items():
            ratio = max_cnt_per_key[k] / max(1, c)
            weight_by_kr[(k, rk)] = min(max_span_weight, ratio ** weight_power)

        for enc, raw_spans, rep in base_records:
            weighted_spans = [
                (tok_idxs, cand_ids, gold_pos, float(weight_by_kr[(key, gold_rk)]))
                for tok_idxs, cand_ids, gold_pos, key, gold_rk in raw_spans
            ]
            item = dict(enc)
            item["span_items"] = weighted_spans
            for _ in range(rep):
                self.items.append(item)

        self.weight_stats = {
            "min_weight": round(min(weight_by_kr.values()), 4) if weight_by_kr else 1.0,
            "max_weight": round(max(weight_by_kr.values()), 4) if weight_by_kr else 1.0,
            "distinct_kr_pairs": len(weight_by_kr),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def compute_weighted_span_ce_loss(
    logits: torch.Tensor,
    span_items_batch: List[List[Tuple[List[int], List[int], int, float]]],
) -> Optional[torch.Tensor]:
    """重み付き Candidate-Restricted Span Cross-Entropy (fp32 上で -1e9 マスク適用)。"""
    span_vecs: List[torch.Tensor] = []
    cand_id_lists: List[List[int]] = []
    gold_positions: List[int] = []
    weights: List[float] = []

    logits_f = logits.float()
    for bi, s_items in enumerate(span_items_batch):
        for tok_idxs, cand_ids, gold_pos, w in s_items:
            vec = logits_f[bi, tok_idxs].mean(dim=0)
            span_vecs.append(vec)
            cand_id_lists.append(cand_ids)
            gold_positions.append(gold_pos)
            weights.append(w)

    if not span_vecs:
        return None

    stacked = torch.stack(span_vecs, dim=0)
    max_k = max(len(c) for c in cand_id_lists)
    n_sp = len(span_vecs)
    cand_logits = torch.full((n_sp, max_k), -1e9, dtype=torch.float32, device=logits.device)
    for si, cids in enumerate(cand_id_lists):
        cand_logits[si, : len(cids)] = stacked[si, cids]

    targets = torch.tensor(gold_positions, dtype=torch.long, device=logits.device)
    losses = F.cross_entropy(cand_logits, targets, reduction="none")
    ws = torch.tensor(weights, dtype=torch.float32, device=logits.device)
    return (losses * ws).sum() / ws.sum().clamp(min=1e-6)


@torch.no_grad()
def collect_eval_logits_and_decode(
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
    """1回だけ順伝播を実行し、各スパンの fp32 softmax 最良候補と確信度を収集する。"""
    model.eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    truncated_count = 0

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
    collected_all: List[List[dict]] = []
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
        logits_f = logits.float()

        for bi, (_, offsets, sp_list) in enumerate(chunk):
            sent_logits = logits_f[bi]
            sent_collected: List[dict] = []
            for sp in sp_list:
                cands = sp["candidates"]
                def_rk = sp["default_reading"]
                best_rk = def_rk
                best_prob = 1.0
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
                            cand_vals = torch.tensor(
                                [float(span_vec[lid].item()) for _, lid in cand_label_ids],
                                dtype=torch.float32,
                            )
                            probs = F.softmax(cand_vals, dim=0)
                            b_idx = int(torch.argmax(probs).item())
                            best_rk = cand_label_ids[b_idx][0]
                            best_prob = float(probs[b_idx].item())

                sent_collected.append(
                    {
                        "start": sp["start"],
                        "end": sp["end"],
                        "surface": sp["surface"],
                        "default_reading": def_rk,
                        "best_rk": best_rk,
                        "best_prob": best_prob,
                    }
                )
            collected_all.append(sent_collected)

    eval_sec = time.time() - t0
    return collected_all, eval_sec, truncated_count


def decode_from_collected(
    collected_all: List[List[dict]],
    extractor: SpanExtractor,
    override_tau: float,
) -> List[List[dict]]:
    """収集済みの確信度結果に上書き閾値 override_tau を適用して最終スパン予測を生成する。"""
    preds_all: List[List[dict]] = []
    for sent_sp in collected_all:
        out_spans: List[dict] = []
        for sp in sent_sp:
            def_rk = sp["default_reading"]
            chosen_rk = def_rk
            if sp["best_rk"] != def_rk and sp["best_prob"] >= override_tau:
                chosen_rk = sp["best_rk"]
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
    return preds_all


def sweep_override_tau(
    rows: List[dict],
    collected_all: List[List[dict]],
    extractor: SpanExtractor,
    eval_sec: float,
    taus: Optional[List[float]] = None,
) -> Tuple[float, dict, List[dict]]:
    """prod_dev 上で override_tau をスイープし、最良閾値・最良メトリクス・全履歴を返す。"""
    tau_list = taus if taus is not None else TAU_GRID
    sweep_records: List[dict] = []
    metrics_by_tau: Dict[float, dict] = {}

    for tau in tau_list:
        preds = decode_from_collected(collected_all, extractor, override_tau=tau)
        m = er.evaluate_predictions(rows, preds, eval_sec=eval_sec)
        metrics_by_tau[tau] = m
        real_acc = m["by_subset_type"].get("real", {}).get("sentence_reading_acc", 0.0)
        synth_acc = m["by_subset_type"].get("synth", {}).get("sentence_reading_acc", 0.0)
        sweep_records.append(
            {
                "override_tau": tau,
                "sentence_reading_acc": m["sentence_reading_acc"],
                "sentence_pron_acc": m["sentence_pron_acc"],
                "span_reading_acc": m["span_reading_acc"],
                "real_sentence_reading_acc": real_acc,
                "synth_sentence_reading_acc": synth_acc,
            }
        )

    best_sent_acc = max(rec["sentence_reading_acc"] for rec in sweep_records)
    near_best = [
        rec
        for rec in sweep_records
        if (best_sent_acc - rec["sentence_reading_acc"]) <= 0.0002 + 1e-12
    ]
    chosen = max(
        near_best,
        key=lambda r: (
            r["synth_sentence_reading_acc"],
            -r["override_tau"],
            r["sentence_reading_acc"],
        ),
    )
    best_tau = float(chosen["override_tau"])
    return best_tau, metrics_by_tau[best_tau], sweep_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="sbintuitions/modernbert-ja-130m")
    parser.add_argument("--train", type=str, default=str(ROOT / "data" / "prod" / "train.jsonl"))
    parser.add_argument("--dev", type=str, default=str(ROOT / "data" / "prod" / "dev.jsonl"))
    parser.add_argument("--pilot-dev", type=str, default=str(ROOT / "data" / "pilot" / "dev.jsonl"))
    parser.add_argument("--master-dir", type=str, default=str(ROOT / "data" / "master"))
    parser.add_argument("--out", type=str, default=str(ROOT / "runs" / "prod-modernbert"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--upsample-factor", type=int, default=3)
    parser.add_argument("--loss-weight-power", type=float, default=0.35)
    parser.add_argument("--max-span-weight", type=float, default=4.0)
    parser.add_argument("--catalog-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    best_dir = out_dir / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    extractor = SpanExtractor(Path(args.master_dir))
    train_rows = tpr.load_jsonl(Path(args.train))
    dev_rows = tpr.load_jsonl(Path(args.dev))
    pilot_dev_rows = tpr.load_jsonl(Path(args.pilot_dev))

    label2id, id2label, targets_per_row, label_meta = build_refined_label_space(
        train_rows,
        extractor,
        min_extra_freq=4,
        min_extra_distinct_reads=2,
        catalog_only=args.catalog_only,
    )
    print("Refined label space:", json.dumps(label_meta, ensure_ascii=False), flush=True)

    active_rows = []
    active_targets = []
    for r, t in zip(train_rows, targets_per_row):
        if t:
            active_rows.append(r)
            active_targets.append(t)
    print(
        f"Active unique training sentences: {len(active_rows)} / {len(train_rows)} "
        f"({len(active_rows)/len(train_rows):.1%})",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base, num_labels=len(id2label)
    )
    max_pos = getattr(model.config, "max_position_embeddings", args.max_len)
    eff_max_len = min(args.max_len, int(max_pos))
    model.to(device)

    train_ds = WeightedReadingTrainDataset(
        active_rows,
        active_targets,
        label2id,
        tok,
        max_len=eff_max_len,
        upsample_factor=args.upsample_factor,
        weight_power=args.loss_weight_power,
        max_span_weight=args.max_span_weight,
        verify_n=200,
    )
    print(
        f"Weighted dataset ready: effective_rows={len(train_ds)} "
        f"(upsampled synth={train_ds.upsampled_synth_rows}, aozora_minority={train_ds.upsampled_aozora_minority_rows}), "
        f"unique_spans={train_ds.total_supervised_spans_unique}, "
        f"effective_spans={train_ds.total_supervised_spans_effective}, "
        f"weights={train_ds.weight_stats}",
        flush=True,
    )

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: tpr.collate_train(b, pad_id=pad_id),
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
    total_steps = max(1, len(loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    epoch_history = []
    best_score = (-1.0, -1.0)
    best_epoch = 0
    best_calibration = None
    has_nan = False
    global_step = 0
    t_train_start = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_losses: List[float] = []
        t_ep = time.time()

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            model_in = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if "token_type_ids" in batch:
                model_in["token_type_ids"] = batch["token_type_ids"].to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(**model_in).logits
                loss = compute_weighted_span_ce_loss(logits, batch["span_items"])

            if loss is None:
                global_step += 1
                scheduler.step()
                continue

            loss_val = float(loss.item())
            if math.isnan(loss_val) or math.isinf(loss_val):
                has_nan = True
                raise RuntimeError(f"NaN/Inf loss at epoch {ep} step {global_step}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ep_losses.append(loss_val)
            global_step += 1
            if global_step % 1500 == 0:
                avg_r = sum(ep_losses[-100:]) / max(1, len(ep_losses[-100:]))
                print(
                    f"[Epoch {ep}/{args.epochs} step {global_step}/{total_steps}] loss_avg100={avg_r:.5f}",
                    flush=True,
                )

        ep_train_sec = time.time() - t_ep

        # prod_dev (5,000件) で 1 パス推論 → tau=0.0 生評価 & tau スイープ較正
        collected_dev, dev_sec, dev_trunc = collect_eval_logits_and_decode(
            model,
            tok,
            dev_rows,
            extractor,
            label2id,
            eff_max_len,
            32,
            device,
            verify_all_offsets=(ep == 1),
        )
        raw_preds = decode_from_collected(collected_dev, extractor, override_tau=0.0)
        raw_metrics = er.evaluate_predictions(dev_rows, raw_preds, eval_sec=dev_sec)

        best_tau_ep, cal_metrics_ep, sweep_ep = sweep_override_tau(
            dev_rows, collected_dev, extractor, eval_sec=dev_sec
        )

        ep_rec = {
            "epoch": ep,
            "steps_end": global_step,
            "epoch_train_sec": round(ep_train_sec, 2),
            "epoch_loss_mean": round(sum(ep_losses) / max(1, len(ep_losses)), 5),
            "dev_truncated": dev_trunc,
            "override_tau": best_tau_ep,
            "dev_raw": raw_metrics,
            "dev": cal_metrics_ep,
            "tau_sweep": sweep_ep,
        }
        epoch_history.append(ep_rec)

        cur_score = (cal_metrics_ep["sentence_reading_acc"], cal_metrics_ep["span_reading_acc"])
        is_best = cur_score > best_score
        if is_best:
            best_score = cur_score
            best_epoch = ep
            best_calibration = {
                "override_tau": best_tau_ep,
                "selected_on": "data/prod/dev.jsonl",
                "best_epoch": ep,
                "sweep": sweep_ep,
            }
            model.save_pretrained(str(best_dir))
            tok.save_pretrained(str(best_dir))
            with open(best_dir / "label_map.json", "w", encoding="utf-8") as f:
                json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False)
            with open(best_dir / "calibration.json", "w", encoding="utf-8") as f:
                json.dump(best_calibration, f, ensure_ascii=False, indent=2)

        print(
            f"=== Epoch {ep} DONE ({ep_train_sec:.1f}s) | loss={ep_rec['epoch_loss_mean']:.5f} | "
            f"raw={raw_metrics['sentence_reading_acc']:.4f} "
            f"(real={raw_metrics['by_subset_type']['real']['sentence_reading_acc']:.4f}, "
            f"synth={raw_metrics['by_subset_type']['synth']['sentence_reading_acc']:.4f}) | "
            f"cal(tau={best_tau_ep:.2f})={cal_metrics_ep['sentence_reading_acc']:.4f} "
            f"(real={cal_metrics_ep['by_subset_type']['real']['sentence_reading_acc']:.4f}, "
            f"synth={cal_metrics_ep['by_subset_type']['synth']['sentence_reading_acc']:.4f}) "
            f"{'[BEST]' if is_best else ''} ===",
            flush=True,
        )

    total_train_sec = time.time() - t_train_start

    # 確定最良モデル＋calibration.json をロードして参考用に pilot_dev (2,000件) を比較評価
    best_model = AutoModelForTokenClassification.from_pretrained(str(best_dir)).to(device)
    chosen_tau = float(best_calibration["override_tau"])
    p_collected, p_sec, p_trunc = collect_eval_logits_and_decode(
        best_model, tok, pilot_dev_rows, extractor, label2id, eff_max_len, 32, device, verify_all_offsets=True
    )
    pilot_raw_preds = decode_from_collected(p_collected, extractor, override_tau=0.0)
    pilot_cal_preds = decode_from_collected(p_collected, extractor, override_tau=chosen_tau)
    pilot_raw_metrics = er.evaluate_predictions(pilot_dev_rows, pilot_raw_preds, eval_sec=p_sec)
    pilot_cal_metrics = er.evaluate_predictions(pilot_dev_rows, pilot_cal_preds, eval_sec=p_sec)

    n_params = sum(p.numel() for p in best_model.parameters())
    best_dir_bytes = sum(p.stat().st_size for p in best_dir.rglob("*") if p.is_file())
    st_path = best_dir / "model.safetensors"
    st_bytes = st_path.stat().st_size if st_path.exists() else 0

    summary = {
        "base_model": args.base,
        "version": "prod-v2",
        "best_epoch": best_epoch,
        "best_override_tau": chosen_tau,
        "total_epochs_trained": args.epochs,
        "label_meta": label_meta,
        "train_total_rows": len(train_rows),
        "train_active_unique_rows": len(active_rows),
        "train_effective_rows": len(train_ds),
        "train_upsampled_synth_rows": train_ds.upsampled_synth_rows,
        "train_upsampled_aozora_minority_rows": train_ds.upsampled_aozora_minority_rows,
        "train_supervised_spans_unique": train_ds.total_supervised_spans_unique,
        "train_supervised_spans_effective": train_ds.total_supervised_spans_effective,
        "train_truncated": train_ds.truncated_count,
        "weight_stats": train_ds.weight_stats,
        "hyperparams": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": eff_max_len,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "upsample_factor": args.upsample_factor,
            "loss_weight_power": args.loss_weight_power,
            "max_span_weight": args.max_span_weight,
            "catalog_only": args.catalog_only,
            "fp16": (device == "cuda"),
            "seed": args.seed,
        },
        "total_train_sec": round(total_train_sec, 2),
        "total_steps": global_step,
        "has_nan_in_train_log": has_nan,
        "model_params": n_params,
        "best_dir_bytes": best_dir_bytes,
        "model_safetensors_bytes": st_bytes,
        "epoch_history": epoch_history,
        "best_prod_dev_raw": epoch_history[best_epoch - 1]["dev_raw"],
        "best_prod_dev": epoch_history[best_epoch - 1]["dev"],
        "best_on_pilot_dev_raw": pilot_raw_metrics,
        "best_on_pilot_dev": pilot_cal_metrics,
        "pilot_dev_truncated": p_trunc,
    }
    with open(out_dir / "train_prod_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"PROD V2 TRAINING COMPLETE | best_epoch={best_epoch} | tau={chosen_tau:.2f} | "
        f"prod_dev={summary['best_prod_dev']['sentence_reading_acc']:.4f} "
        f"(synth={summary['best_prod_dev']['by_subset_type']['synth']['sentence_reading_acc']:.4f}) | "
        f"pilot_dev={pilot_cal_metrics['sentence_reading_acc']:.4f} "
        f"(synth={pilot_cal_metrics['by_subset_type']['synth']['sentence_reading_acc']:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
