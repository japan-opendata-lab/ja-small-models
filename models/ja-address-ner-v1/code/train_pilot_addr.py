#!/usr/bin/env python3
"""第7号 ja-address-ner-v1 パイロット学習・dev 評価スクリプト（両ベース共通）。

11クラス BIO（O + PREF/CITY/TOWN/BLOCK/BUILDING）によるトークン分類モデルを学習し、
dev.jsonl 上で以下を算出・出力する：
  - 住所スパン完全一致 F1（主指標）、重なり一致 F1（参考）
  - 5構成要素ごとの分解正確さ（PREF/CITY/TOWN/BLOCK/BUILDING 各 P/R/F1, micro/macro）
  - 抽出成功住所における全構成要素完全一致率
  - 住所なし文誤検出率（no_address_fpr）
  - 10カテゴリ別内訳、トークナイザ理論上限（tokenizer_oracle）
  - 学習/推論速度・サイズメタ情報および dev_predictions.jsonl

注意: 封印テスト（sealed-test.jsonl）は一切入力禁止。
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
import offsets_util as ou  # noqa: E402

CATEGORIES = [
    "standard",
    "omit_pref",
    "same_name_resolved",
    "old_muni",
    "variant",
    "omit_block",
    "reordered",
    "unresolvable_ambiguous",
    "unresolvable_nonexistent",
    "no_address",
]


class AddrDataset(Dataset):
    """住所抽出・5要素分解用トークン分類データセット。"""

    def __init__(
        self,
        path: Path,
        tok,
        max_len: int = 384,
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
                non_empty_ends = [b for a, b in offsets if b > a]
                if r["text"] and (
                    not non_empty_ends or max(non_empty_ends) < len(r["text"].rstrip())
                ):
                    self.truncated_count += 1

                labels = ou.align_spans_to_tokens(r["text"], offsets, r["components"])
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
            [
                b["token_type_ids"] + [0] * (maxlen - len(b["token_type_ids"]))
                for b in batch
            ],
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


def _match_overlap_1to1(gold_spans: List[dict], pred_spans: List[dict]) -> List[Tuple[int, int]]:
    pairs = []
    for gi, g in enumerate(gold_spans):
        gs, ge = int(g["start"]), int(g["end"])
        for pi, p in enumerate(pred_spans):
            ps, pe = int(p["start"]), int(p["end"])
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
    pred_comps_list: List[List[dict]],
    pred_addrs_list: List[List[dict]],
) -> dict:
    """住所スパン抽出・5構成要素分解・住所なし文FPR・カテゴリ別内訳を一括算出する。"""
    addr_ex_tp = addr_ex_fp = addr_ex_fn = 0
    addr_ov_tp = addr_ov_fp = addr_ov_fn = 0

    comp_counts: Dict[str, Dict[str, int]] = {
        lb: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for lb in ou.COMPONENTS
    }

    exact_addr_matched_total = 0
    exact_addr_full_decomp_ok = 0
    ov_addr_matched_total = 0
    ov_addr_full_decomp_ok = 0

    noaddr_docs = 0
    noaddr_chars = 0
    noaddr_docs_with_fp = 0
    noaddr_fp_addrs = 0
    noaddr_fp_comps = 0
    noaddr_fp_examples: List[dict] = []

    cat_stats: Dict[str, dict] = {
        c: {
            "docs": 0,
            "gold_addresses": 0,
            "addr_tp": 0,
            "addr_fp": 0,
            "addr_fn": 0,
            "comp_tp": 0,
            "comp_fp": 0,
            "comp_fn": 0,
            "per_comp": {
                lb: {"tp": 0, "fp": 0, "fn": 0} for lb in ou.COMPONENTS
            },
        }
        for c in CATEGORIES
    }

    for row, p_comps, p_addrs in zip(raw_rows, pred_comps_list, pred_addrs_list):
        cat = row["category"]
        cat_st = cat_stats[cat]
        cat_st["docs"] += 1

        g_addrs = row["addresses"]
        g_comps = row["components"]
        cat_st["gold_addresses"] += len(g_addrs)

        # 住所なし文誤検出
        if cat == "no_address":
            noaddr_docs += 1
            noaddr_chars += len(row["text"])
            if p_addrs or p_comps:
                noaddr_docs_with_fp += 1
                noaddr_fp_addrs += len(p_addrs)
                noaddr_fp_comps += len(p_comps)
                if len(noaddr_fp_examples) < 10:
                    noaddr_fp_examples.append(
                        {
                            "id": row["id"],
                            "text": row["text"],
                            "pred_addrs": [a["text"] for a in p_addrs],
                            "pred_comps": [(c["label"], c["text"]) for c in p_comps],
                        }
                    )

        # 住所スパン完全一致
        g_addr_map = {(int(a["start"]), int(a["end"])): a for a in g_addrs}
        p_addr_map = {(int(a["start"]), int(a["end"])): a for a in p_addrs}
        g_addr_set = set(g_addr_map.keys())
        p_addr_set = set(p_addr_map.keys())

        tp_a = len(g_addr_set & p_addr_set)
        fp_a = len(p_addr_set - g_addr_set)
        fn_a = len(g_addr_set - p_addr_set)
        addr_ex_tp += tp_a
        addr_ex_fp += fp_a
        addr_ex_fn += fn_a
        cat_st["addr_tp"] += tp_a
        cat_st["addr_fp"] += fp_a
        cat_st["addr_fn"] += fn_a

        for k in g_addr_set & p_addr_set:
            exact_addr_matched_total += 1
            ga_cs = {
                (int(c["start"]), int(c["end"]), c["label"])
                for c in g_comps
                if c["start"] >= k[0] and c["end"] <= k[1]
            }
            pa_cs = {
                (int(c["start"]), int(c["end"]), c["label"])
                for c in p_addr_map[k]["component_spans"]
            }
            if ga_cs == pa_cs:
                exact_addr_full_decomp_ok += 1

        # 住所スパン重なり一致
        ov_pairs = _match_overlap_1to1(g_addrs, p_addrs)
        addr_ov_tp += len(ov_pairs)
        addr_ov_fp += len(p_addrs) - len(ov_pairs)
        addr_ov_fn += len(g_addrs) - len(ov_pairs)
        for gi, pi in ov_pairs:
            ov_addr_matched_total += 1
            ga = g_addrs[gi]
            pa = p_addrs[pi]
            ga_cs = {
                (int(c["start"]), int(c["end"]), c["label"])
                for c in g_comps
                if c["start"] >= ga["start"] and c["end"] <= ga["end"]
            }
            pa_cs = {
                (int(c["start"]), int(c["end"]), c["label"])
                for c in pa["component_spans"]
            }
            if ga_cs == pa_cs:
                ov_addr_full_decomp_ok += 1

        # 5構成要素ごとの完全一致（位置＋種別一致）
        g_comp_set = {(int(c["start"]), int(c["end"]), c["label"]) for c in g_comps}
        p_comp_set = {(int(c["start"]), int(c["end"]), c["label"]) for c in p_comps}

        cat_st["comp_tp"] += len(g_comp_set & p_comp_set)
        cat_st["comp_fp"] += len(p_comp_set - g_comp_set)
        cat_st["comp_fn"] += len(g_comp_set - p_comp_set)

        for lb in ou.COMPONENTS:
            gs_lb = {(s, e) for (s, e, l) in g_comp_set if l == lb}
            ps_lb = {(s, e) for (s, e, l) in p_comp_set if l == lb}
            tp_l = len(gs_lb & ps_lb)
            fp_l = len(ps_lb - gs_lb)
            fn_l = len(gs_lb - ps_lb)
            comp_counts[lb]["support"] += len(gs_lb)
            comp_counts[lb]["tp"] += tp_l
            comp_counts[lb]["fp"] += fp_l
            comp_counts[lb]["fn"] += fn_l
            cat_st["per_comp"][lb]["tp"] += tp_l
            cat_st["per_comp"][lb]["fp"] += fp_l
            cat_st["per_comp"][lb]["fn"] += fn_l

    per_component_exact: Dict[str, dict] = {}
    mic_tp = mic_fp = mic_fn = 0
    f1_list = []
    for lb in ou.COMPONENTS:
        cc = comp_counts[lb]
        d = _prf(cc["tp"], cc["fp"], cc["fn"])
        d["support"] = cc["support"]
        per_component_exact[lb] = d
        mic_tp += cc["tp"]
        mic_fp += cc["fp"]
        mic_fn += cc["fn"]
        f1_list.append(d["f1"])
    per_component_exact["micro"] = _prf(mic_tp, mic_fp, mic_fn)
    per_component_exact["macro_f1"] = round(sum(f1_list) / len(f1_list), 4)

    per_category: Dict[str, dict] = {}
    for cat in CATEGORIES:
        cs = cat_st = cat_stats[cat]
        per_category[cat] = {
            "docs": cs["docs"],
            "gold_addresses": cs["gold_addresses"],
            "address_exact": _prf(cs["addr_tp"], cs["addr_fp"], cs["addr_fn"]),
            "component_micro": _prf(cs["comp_tp"], cs["comp_fp"], cs["comp_fn"]),
            "per_comp_f1": {
                lb: _prf(cs["per_comp"][lb]["tp"], cs["per_comp"][lb]["fp"], cs["per_comp"][lb]["fn"])["f1"]
                for lb in ou.COMPONENTS
            },
        }

    no_address_fpr = {
        "no_address_docs": noaddr_docs,
        "no_address_chars": noaddr_chars,
        "docs_with_fp": noaddr_docs_with_fp,
        "sentence_fpr": round(noaddr_docs_with_fp / noaddr_docs, 4) if noaddr_docs else 0.0,
        "fp_addresses": noaddr_fp_addrs,
        "fp_components": noaddr_fp_comps,
        "fp_addrs_per_1000_chars": round(noaddr_fp_addrs / (noaddr_chars / 1000.0), 4)
        if noaddr_chars
        else 0.0,
    }

    return {
        "address_exact_span": _prf(addr_ex_tp, addr_ex_fp, addr_ex_fn),
        "address_overlap_span": _prf(addr_ov_tp, addr_ov_fp, addr_ov_fn),
        "per_component_exact": per_component_exact,
        "full_decomp_exact_match_on_extracted": round(
            exact_addr_full_decomp_ok / exact_addr_matched_total, 4
        )
        if exact_addr_matched_total
        else 0.0,
        "full_decomp_exact_match_on_matched": round(
            ov_addr_full_decomp_ok / ov_addr_matched_total, 4
        )
        if ov_addr_matched_total
        else 0.0,
        "no_address_fpr": no_address_fpr,
        "per_category": per_category,
        "no_address_fp_examples": noaddr_fp_examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=384)
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

    train_ds = AddrDataset(Path(args.train), tok, max_len=max_len, verify_n=200)
    dev_ds = AddrDataset(Path(args.dev), tok, max_len=max_len, verify_n=999999)
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

    pred_comps_all: List[List[dict]] = []
    pred_addrs_all: List[List[dict]] = []
    oracle_comps_all: List[List[dict]] = []
    oracle_addrs_all: List[List[dict]] = []

    eval_bs = 32
    total_chars = sum(len(r["text"]) for r in dev_ds.raw_rows)
    t_eval0 = time.time()
    with torch.no_grad():
        for start_i in range(0, len(dev_ds), eval_bs):
            batch_items = [
                dev_ds[j] for j in range(start_i, min(start_i + eval_bs, len(dev_ds)))
            ]
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

                p_comps = ou.decode_token_preds_to_components(raw_text, offsets, pred_ids)
                p_addrs = ou.group_components_to_addresses(raw_text, p_comps)
                pred_comps_all.append(p_comps)
                pred_addrs_all.append(p_addrs)

                oracle_ids = [0 if lb == -100 else lb for lb in item["labels"]]
                o_comps = ou.decode_token_preds_to_components(raw_text, offsets, oracle_ids)
                o_addrs = ou.group_components_to_addresses(raw_text, o_comps)
                oracle_comps_all.append(o_comps)
                oracle_addrs_all.append(o_addrs)

    eval_sec = time.time() - t_eval0

    # dev_predictions.jsonl 出力（第2段階辞書リンカー評価入力用）
    with open(out_dir / "dev_predictions.jsonl", "w", encoding="utf-8") as pf:
        for row, p_comps, p_addrs in zip(dev_ds.raw_rows, pred_comps_all, pred_addrs_all):
            pf.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "category": row["category"],
                        "pred_components": p_comps,
                        "pred_addresses": p_addrs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = evaluate_predictions(dev_ds.raw_rows, pred_comps_all, pred_addrs_all)
    oracle_metrics = evaluate_predictions(
        dev_ds.raw_rows, oracle_comps_all, oracle_addrs_all
    )

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
        "address_exact_span": metrics["address_exact_span"],
        "address_overlap_span": metrics["address_overlap_span"],
        "per_component_exact": metrics["per_component_exact"],
        "full_decomp_exact_match_on_extracted": metrics[
            "full_decomp_exact_match_on_extracted"
        ],
        "full_decomp_exact_match_on_matched": metrics[
            "full_decomp_exact_match_on_matched"
        ],
        "no_address_fpr": metrics["no_address_fpr"],
        "per_category": metrics["per_category"],
        "tokenizer_oracle": {
            "address_exact_span": oracle_metrics["address_exact_span"],
            "per_component_exact": oracle_metrics["per_component_exact"],
            "full_decomp_exact_match_on_extracted": oracle_metrics[
                "full_decomp_exact_match_on_extracted"
            ],
        },
        "no_address_fp_examples": metrics["no_address_fp_examples"],
    }

    (out_dir / "eval_pilot.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "base": report["base"],
        "train_sec": report["train_sec"],
        "train_loss": report["train_loss"],
        "address_exact_f1": report["address_exact_span"]["f1"],
        "component_micro_f1": report["per_component_exact"]["micro"]["f1"],
        "component_macro_f1": report["per_component_exact"]["macro_f1"],
        "no_address_sentence_fpr": report["no_address_fpr"]["sentence_fpr"],
        "oracle_addr_f1": report["tokenizer_oracle"]["address_exact_span"]["f1"],
        "oracle_comp_micro_f1": report["tokenizer_oracle"]["per_component_exact"]["micro"]["f1"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
