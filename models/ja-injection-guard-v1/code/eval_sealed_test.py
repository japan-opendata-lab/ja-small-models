#!/usr/bin/env python3
"""確定本番モデル (runs/select-modernbert/final) による封印テスト最終評価スクリプト。

仕様 (Phase 5):
1. dev.jsonl 先頭200件でローカル PyTorch (FP32) と Colab 学習時 dev_logits.npy の整合を確認する。
2. data/prod/sealed-test.jsonl (2,500件) を1回限りで評価し、sealed_logits.npy / sealed_probs.npy と
   eval_sealed_test.json を出力する (閾値再調整・再学習は行わない)。
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_select_cls import binary_metrics, load_jsonl, source_group  # noqa: E402


def predict_logits(model, tok, rows, max_len: int, batch_size: int, device: torch.device) -> np.ndarray:
    """データ行をトークン長順にソートしてパディング膨張を抑えつつ推論し、元の順序のロジット配列 (N, 2) を返す。"""
    texts = [r["text"] for r in rows]
    enc_all = tok(texts, truncation=True, max_length=max_len)
    lengths = [len(ids) for ids in enc_all["input_ids"]]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    out_logits = np.empty((len(texts), 2), dtype=np.float64)
    total_batches = (len(texts) + batch_size - 1) // batch_size

    model.eval()
    with torch.no_grad():
        for b_idx, start in enumerate(range(0, len(order), batch_size), start=1):
            batch_idxs = order[start : start + batch_size]
            maxlen = max(len(enc_all["input_ids"][i]) for i in batch_idxs)
            input_ids = []
            attention_mask = []
            for i in batch_idxs:
                ids = enc_all["input_ids"][i]
                mask = enc_all["attention_mask"][i]
                pad_n = maxlen - len(ids)
                input_ids.append(ids + [pad_id] * pad_n)
                attention_mask.append(mask + [0] * pad_n)
            enc = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
            }
            out = model(**enc)
            logits = out.logits.detach().cpu().to(torch.float64).numpy()
            for row_j, orig_i in enumerate(batch_idxs):
                out_logits[orig_i] = logits[row_j]
            if b_idx % 25 == 0 or b_idx == total_batches:
                print(f"[predict_logits] {b_idx}/{total_batches} batches ({min(start + batch_size, len(texts))}/{len(texts)})", flush=True)
    return out_logits


def eval_at_threshold(probs: np.ndarray, gold: np.ndarray, rows: list, threshold: float) -> dict:
    """指定閾値における全体・サブカテゴリ別・出典別・長文スライス指標を一括算出する。"""
    overall = binary_metrics(probs, gold, threshold)

    subcat_indices = defaultdict(list)
    src_grp_indices = defaultdict(list)
    src_raw_indices = defaultdict(list)
    for idx, r in enumerate(rows):
        subcat_indices[r.get("subcategory", "?")].append(idx)
        src = r["source"]
        src_grp_indices[source_group(src)].append(idx)
        src_raw_indices[src].append(idx)

    per_subcategory = {}
    for cat, idxs in sorted(subcat_indices.items()):
        idx_arr = np.array(idxs, dtype=int)
        bm = binary_metrics(probs[idx_arr], gold[idx_arr], threshold)
        pred = (probs[idx_arr] >= threshold).astype(int)
        bm["correct"] = int((pred == gold[idx_arr]).sum())
        lbl_vals = sorted(set(int(x) for x in gold[idx_arr]))
        bm["label"] = lbl_vals[0] if len(lbl_vals) == 1 else -1
        per_subcategory[cat] = bm

    per_source_group = {}
    for grp, idxs in sorted(src_grp_indices.items()):
        idx_arr = np.array(idxs, dtype=int)
        per_source_group[grp] = binary_metrics(probs[idx_arr], gold[idx_arr], threshold)

    per_source_raw = {}
    for src, idxs in sorted(src_raw_indices.items()):
        idx_arr = np.array(idxs, dtype=int)
        d = binary_metrics(probs[idx_arr], gold[idx_arr], threshold)
        d["source_group"] = source_group(src)
        per_source_raw[src] = d

    mask_long = np.array([len(r["text"]) > 512 for r in rows], dtype=bool)
    long_slice = {
        "rule": "len(text) > 512 chars",
        "n_total": int(mask_long.sum()),
        "n_attack": int((gold[mask_long] == 1).sum()),
        "n_benign": int((gold[mask_long] == 0).sum()),
    }
    long_slice.update(binary_metrics(probs[mask_long], gold[mask_long], threshold))

    return {
        "threshold": float(threshold),
        "overall": overall,
        "per_subcategory": per_subcategory,
        "per_source_group": per_source_group,
        "per_source_raw": per_source_raw,
        "long_slice_gt512chars": long_slice,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="runs/select-modernbert/final")
    ap.add_argument("--sealed", default="data/prod/sealed-test.jsonl")
    ap.add_argument("--dev", default="data/prod/dev.jsonl")
    ap.add_argument("--dev-ref-logits", default="runs/select-modernbert/dev_logits.npy")
    ap.add_argument("--dev-eval-json", default="runs/select-modernbert/eval_select_cls.json")
    ap.add_argument("--out-dir", default="runs/select-modernbert")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dev-thresh", type=float, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--heldout-ni-xs-from", default="", help="NotInject/XSTest 誤検出率を別途計測する JSONL (例: diagnostic-test.jsonl)")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.float()
    model.to(device)
    model.eval()

    # 1. dev 先頭200件での環境再現性確認 (ローカル FP32 vs Colab 保存ロジット)
    dev_rows = load_jsonl(args.dev)
    dev_rows_200 = dev_rows[:200]
    local_dev200_logits = predict_logits(model, tok, dev_rows_200, args.max_len, args.batch, device)
    local_dev200_probs = torch.softmax(torch.tensor(local_dev200_logits, dtype=torch.float64), dim=-1)[:, 1].numpy()

    ref_logits_full = np.load(args.dev_ref_logits).astype(np.float64)
    ref_probs_full = torch.softmax(torch.tensor(ref_logits_full, dtype=torch.float64), dim=-1)[:, 1].numpy()
    ref_dev200_logits = ref_logits_full[:200]
    ref_dev200_probs = ref_probs_full[:200]

    logit_diff = np.abs(local_dev200_logits - ref_dev200_logits)
    prob_diff = np.abs(local_dev200_probs - ref_dev200_probs)
    dev200_sanity = {
        "n": 200,
        "device": str(device),
        "dev_thresh_fixed": args.dev_thresh,
        "agree_at_0_5": int(((local_dev200_probs >= 0.5) == (ref_dev200_probs >= 0.5)).sum()),
        "agree_at_dev_thresh": int(((local_dev200_probs >= args.dev_thresh) == (ref_dev200_probs >= args.dev_thresh)).sum()),
        "max_abs_logit_diff": float(logit_diff.max()),
        "mean_abs_logit_diff": float(logit_diff.mean()),
        "max_abs_prob_diff": float(prob_diff.max()),
        "mean_abs_prob_diff": float(prob_diff.mean()),
    }

    # 2. 封印テスト全 2,500 件の推論
    sealed_rows = load_jsonl(args.sealed)
    assert len(sealed_rows) > 0, "sealed-test.jsonl が空です"
    t_inf0 = time.perf_counter()
    sealed_logits = predict_logits(model, tok, sealed_rows, args.max_len, args.batch, device)
    inf_sec = time.perf_counter() - t_inf0
    sealed_probs = torch.softmax(torch.tensor(sealed_logits, dtype=torch.float64), dim=-1)[:, 1].numpy()
    gold = np.array([int(r["label"]) for r in sealed_rows], dtype=int)

    np.save(os.path.join(args.out_dir, "sealed_logits.npy"), sealed_logits)
    np.save(os.path.join(args.out_dir, "sealed_probs.npy"), sealed_probs)

    # dev の厳密閾値も読み込み、指定丸め値との予測差分を確認
    dev_eval = json.load(open(args.dev_eval_json, encoding="utf-8"))
    dev_exact_thresh = float(dev_eval["fpr_1pct_operating_point"]["threshold"])

    res_05 = eval_at_threshold(sealed_probs, gold, sealed_rows, 0.5)
    res_dev = eval_at_threshold(sealed_probs, gold, sealed_rows, args.dev_thresh)
    res_dev_exact = eval_at_threshold(sealed_probs, gold, sealed_rows, dev_exact_thresh)
    pred_diff_fixed_vs_exact = int(
        ((sealed_probs >= args.dev_thresh) != (sealed_probs >= dev_exact_thresh)).sum()
    )

    fp_high_conf_ge_099 = {
        "at_0_5": {
            "fp_total": int(((gold == 0) & (sealed_probs >= 0.5)).sum()),
            "fp_prob_ge_099": int(((gold == 0) & (sealed_probs >= 0.5) & (sealed_probs >= 0.99)).sum()),
        },
        "at_dev_fixed": {
            "fp_total": int(((gold == 0) & (sealed_probs >= args.dev_thresh)).sum()),
            "fp_prob_ge_099": int(((gold == 0) & (sealed_probs >= args.dev_thresh) & (sealed_probs >= 0.99)).sum()),
        },
        "total_benign_prob_ge_099": int(((gold == 0) & (sealed_probs >= 0.99)).sum()),
    }

    # 3. (任意) 学習未見 NotInject / XSTest の誤検出率確認
    heldout_ni_xs = None
    if args.heldout_ni_xs_from and os.path.exists(args.heldout_ni_xs_from):
        ext_rows = [
            r for r in load_jsonl(args.heldout_ni_xs_from)
            if r.get("source") in ("notinject", "xstest_safe")
        ]
        ext_logits = predict_logits(model, tok, ext_rows, args.max_len, args.batch, device)
        ext_probs = torch.softmax(torch.tensor(ext_logits, dtype=torch.float64), dim=-1)[:, 1].numpy()

        def _fpr_stats(sub_probs: np.ndarray) -> dict:
            n = int(len(sub_probs))
            if n == 0:
                return {"n": 0}
            fp_fix = int((sub_probs >= args.dev_thresh).sum())
            fp_ex = int((sub_probs >= dev_exact_thresh).sum())
            fp_05 = int((sub_probs >= 0.5).sum())
            return {
                "n": n,
                "at_dev_fixed": {"fp": fp_fix, "fpr": round(fp_fix / n, 6)},
                "at_dev_exact": {"fp": fp_ex, "fpr": round(fp_ex / n, 6)},
                "at_0_5": {"fp": fp_05, "fpr": round(fp_05 / n, 6)},
            }

        heldout_ni_xs = {"source_file": args.heldout_ni_xs_from}
        for src_k in ("notinject", "xstest_safe"):
            p_diag = np.array([p for r, p in zip(ext_rows, ext_probs) if r["source"] == src_k], dtype=np.float64)
            p_dev = np.array([p for r, p in zip(dev_rows, ref_probs_full) if r["source"] == src_k], dtype=np.float64)
            p_all = np.concatenate([p_dev, p_diag]) if len(p_dev) + len(p_diag) > 0 else np.array([], dtype=np.float64)
            heldout_ni_xs[src_k] = {
                "diagnostic_half": _fpr_stats(p_diag),
                "dev_half": _fpr_stats(p_dev),
                "combined_full": _fpr_stats(p_all),
            }

    out = {
        "model_dir": args.model_dir,
        "sealed_path": args.sealed,
        "sealed_n": len(sealed_rows),
        "sealed_n_attack": int((gold == 1).sum()),
        "sealed_n_benign": int((gold == 0).sum()),
        "max_len": args.max_len,
        "batch": args.batch,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "inference_sec": round(inf_sec, 2),
        "total_elapsed_sec": round(time.time() - t0, 2),
        "dev_thresh_fixed": float(args.dev_thresh),
        "dev_thresh_exact": dev_exact_thresh,
        "dev200_sanity_vs_colab": dev200_sanity,
        "pred_diff_fixed_vs_dev_exact": pred_diff_fixed_vs_exact,
        "fp_high_conf_ge_099": fp_high_conf_ge_099,
        "threshold_dev_fixed": res_dev,
        "threshold_dev_exact": res_dev_exact,
        "threshold_0_5": res_05,
        "heldout_notinject_xstest": heldout_ni_xs,
    }

    out_path = os.path.join(args.out_dir, "eval_sealed_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "sealed_n": out["sealed_n"],
                "inference_sec": out["inference_sec"],
                "total_elapsed_sec": out["total_elapsed_sec"],
                "dev200_sanity_vs_colab": dev200_sanity,
                "pred_diff_fixed_vs_dev_exact": pred_diff_fixed_vs_exact,
                "threshold_dev_fixed_overall": res_dev["overall"],
                "threshold_0_5_overall": res_05["overall"],
                "long_dev_fixed": res_dev["long_slice_gt512chars"],
                "long_0_5": res_05["long_slice_gt512chars"],
                "fp_high_conf_ge_099": fp_high_conf_ge_099,
                "heldout_notinject_xstest": heldout_ni_xs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
