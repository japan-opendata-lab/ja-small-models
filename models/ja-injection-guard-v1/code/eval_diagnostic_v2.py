#!/usr/bin/env python3
"""再学習後モデルによる診断用セット (diagnostic-test.jsonl: 旧 v1 封印テスト格下げ版) 評価スクリプト。

実行内容:
1. dev.jsonl 先頭 200 件について、保存済み dev_logits.npy との推論再現性（一致率・最大差分）を確認する。
2. diagnostic-test.jsonl（2,500件）を FP32 で全件推論し、diagnostic_logits.npy / diagnostic_probs.npy を保存する。
3. 閾値 0.5 固定、および dev で決定した FPR<=1% 閾値（厳密値および小数第4位丸め値）で
   全体・サブカテゴリ別・出典別・長文スライス (>512文字) の各指標を算出する。
4. v1 で誤検出 455/455 件の原因となった静的定型段落を含む synth_ja_v1 benign_long スライスの改善度も計測する。
5. 新しい sealed-test.jsonl には一切触れない。
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

# v1 の旧 sealed-test で benign_long の高誤検出率の原因となった固定段落
V1_STATIC_PARA = "引き続き標準チェックリストに沿った確認を継続し、追加の変更が発生した場合のみ臨時で通知を行います。"


def predict_logits(model, tok, rows: list, max_len: int, batch_size: int, device: torch.device) -> np.ndarray:
    """トークン長順ソートによるパディング削減バッチ推論を実行し、元順序のロジット配列 (N, 2) を返す。"""
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
            if b_idx % 50 == 0 or b_idx == total_batches:
                print(
                    f"[predict_logits] {b_idx}/{total_batches} batches ({min(start + batch_size, len(texts))}/{len(texts)})",
                    flush=True,
                )
    return out_logits


def eval_v1_static_para_slice(probs: np.ndarray, gold: np.ndarray, rows: list, threshold: float) -> dict:
    """v1 高誤検出要因の固定段落の有無で synth_ja_v1 benign_long を分けて誤検出率を計測する。"""
    has_para_idx = []
    no_para_idx = []
    for idx, r in enumerate(rows):
        if r.get("source") == "synth_ja_v1" and r.get("subcategory") == "benign_long":
            if V1_STATIC_PARA in r["text"]:
                has_para_idx.append(idx)
            else:
                no_para_idx.append(idx)

    def _subset(idxs: list) -> dict:
        if not idxs:
            return {"n_total": 0}
        arr = np.array(idxs, dtype=int)
        return binary_metrics(probs[arr], gold[arr], threshold)

    return {
        "static_paragraph": V1_STATIC_PARA,
        "synth_ja_v1_benign_long_with_static_para": _subset(has_para_idx),
        "synth_ja_v1_benign_long_without_static_para": _subset(no_para_idx),
    }


def eval_at_threshold(probs: np.ndarray, gold: np.ndarray, rows: list, threshold: float) -> dict:
    """指定閾値における全体・サブカテゴリ別・出典別・長文スライス・v1問題段落スライスを算出する。"""
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
        "v1_static_paragraph_slice": eval_v1_static_para_slice(probs, gold, rows, threshold),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="runs/retrain-modernbert-v2/final")
    ap.add_argument("--diagnostic", default="data/prod/diagnostic-test.jsonl")
    ap.add_argument("--dev", default="data/prod/dev.jsonl")
    ap.add_argument("--dev-eval-json", default="runs/retrain-modernbert-v2/eval_select_cls.json")
    ap.add_argument("--dev-ref-logits", default="runs/retrain-modernbert-v2/dev_logits.npy")
    ap.add_argument("--out-dir", default="runs/retrain-modernbert-v2")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
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

    with open(args.dev_eval_json, encoding="utf-8") as f:
        dev_eval = json.load(f)
    t_dev_exact = float(dev_eval["fpr_1pct_operating_point"]["threshold"])
    t_dev_round4 = round(t_dev_exact, 4)

    # 1. dev 先頭 200 件での推論整合確認 (保存済み dev_logits.npy との比較)
    dev_rows_200 = load_jsonl(args.dev)[:200]
    local_dev200_logits = predict_logits(model, tok, dev_rows_200, args.max_len, args.batch, device)
    local_dev200_probs = torch.softmax(torch.tensor(local_dev200_logits, dtype=torch.float64), dim=-1)[:, 1].numpy()

    ref_logits_full = np.load(args.dev_ref_logits)
    ref_dev200_logits = ref_logits_full[:200].astype(np.float64)
    ref_dev200_probs = torch.softmax(torch.tensor(ref_dev200_logits, dtype=torch.float64), dim=-1)[:, 1].numpy()

    logit_diff = np.abs(local_dev200_logits - ref_dev200_logits)
    prob_diff = np.abs(local_dev200_probs - ref_dev200_probs)
    dev200_sanity = {
        "n": 200,
        "device": str(device),
        "t_dev_exact": t_dev_exact,
        "t_dev_round4": t_dev_round4,
        "agree_at_0_5": int(((local_dev200_probs >= 0.5) == (ref_dev200_probs >= 0.5)).sum()),
        "agree_at_dev_fpr1": int(((local_dev200_probs >= t_dev_exact) == (ref_dev200_probs >= t_dev_exact)).sum()),
        "max_abs_logit_diff": float(logit_diff.max()),
        "mean_abs_logit_diff": float(logit_diff.mean()),
        "max_abs_prob_diff": float(prob_diff.max()),
        "mean_abs_prob_diff": float(prob_diff.mean()),
    }

    # 2. diagnostic-test.jsonl 全 2,500 件の推論
    diag_rows = load_jsonl(args.diagnostic)
    assert len(diag_rows) > 0, "diagnostic-test.jsonl が空です"
    t_inf0 = time.perf_counter()
    diag_logits = predict_logits(model, tok, diag_rows, args.max_len, args.batch, device)
    inf_sec = time.perf_counter() - t_inf0
    diag_probs = torch.softmax(torch.tensor(diag_logits, dtype=torch.float64), dim=-1)[:, 1].numpy()
    gold = np.array([int(r["label"]) for r in diag_rows], dtype=int)

    np.save(os.path.join(args.out_dir, "diagnostic_logits.npy"), diag_logits)
    np.save(os.path.join(args.out_dir, "diagnostic_probs.npy"), diag_probs)

    res_05 = eval_at_threshold(diag_probs, gold, diag_rows, 0.5)
    res_dev_exact = eval_at_threshold(diag_probs, gold, diag_rows, t_dev_exact)
    res_dev_round4 = eval_at_threshold(diag_probs, gold, diag_rows, t_dev_round4)
    pred_diff_round4_vs_exact = int(
        ((diag_probs >= t_dev_round4) != (diag_probs >= t_dev_exact)).sum()
    )

    out = {
        "model_dir": args.model_dir,
        "diagnostic_path": args.diagnostic,
        "diagnostic_n": len(diag_rows),
        "diagnostic_n_attack": int((gold == 1).sum()),
        "diagnostic_n_benign": int((gold == 0).sum()),
        "max_len": args.max_len,
        "batch": args.batch,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "inference_sec": round(inf_sec, 2),
        "total_elapsed_sec": round(time.time() - t0, 2),
        "t_dev_exact": t_dev_exact,
        "t_dev_round4": t_dev_round4,
        "pred_diff_round4_vs_exact": pred_diff_round4_vs_exact,
        "dev200_sanity": dev200_sanity,
        "threshold_0_5": res_05,
        "threshold_dev_exact": res_dev_exact,
        "threshold_dev_round4": res_dev_round4,
    }

    out_path = os.path.join(args.out_dir, "eval_diagnostic_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "diagnostic_n": out["diagnostic_n"],
                "inference_sec": out["inference_sec"],
                "t_dev_exact": t_dev_exact,
                "t_dev_round4": t_dev_round4,
                "pred_diff_round4_vs_exact": pred_diff_round4_vs_exact,
                "dev200_sanity": dev200_sanity,
                "threshold_0_5_overall": res_05["overall"],
                "threshold_0_5_benign_long": res_05["per_subcategory"].get("benign_long"),
                "threshold_dev_exact_overall": res_dev_exact["overall"],
                "threshold_dev_exact_benign_long": res_dev_exact["per_subcategory"].get("benign_long"),
                "v1_static_para_at_0_5": res_05["v1_static_paragraph_slice"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
