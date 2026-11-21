#!/usr/bin/env python
"""第11号 ja-nli-v1 共通 NLI 評価・多言語ゼロショット評価スクリプト。

全モデル共通で以下を実施・計測する:
- 統一ラベル空間: {0: entailment, 1: neutral, 2: contradiction}
- model.config.id2label 動的検査と並び替え（多言語 NLI ゼロショット対応）
- 自明な日本語3例によるラベルサニティチェック
- truncation="only_first" 前提末尾優先切り詰め + 切り詰め件数記録
- Accuracy / Macro F1 / クラス別 P,R,F1 / E↔C 極性反転率 / 3x3 混同行列 /
  RAG 2値根拠チェック (P, R, F1, PR-AUC, ROC-AUC) / 長さ帯・512トークン境界・ドメイン・サブタイプ層別指標
※ sealed-test は安全装置により評価対象から除外。
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CANONICAL_LABELS = ["entailment", "neutral", "contradiction"]
LABEL2ID = {l: i for i, l in enumerate(CANONICAL_LABELS)}

SANITY_EXAMPLES = [
    ("東京駅は東京都千代田区に位置する主要な鉄道駅である。", "東京駅は東京都千代田区にある。", "entailment"),
    ("東京駅は東京都千代田区に位置する主要な鉄道駅である。", "東京駅は1914年に開業し多くの利用客がいる。", "neutral"),
    ("東京駅は東京都千代田区に位置する主要な鉄道駅である。", "東京駅は大阪府大阪市に位置する駅である。", "contradiction"),
]


def resolve_label_remap(config) -> Tuple[List[int], Dict[int, str]]:
    """モデルの id2label を検査し、[E, N, C] 順へマップする旧IDリストを返す。"""
    id2label = getattr(config, "id2label", None) or {}
    norm_map = {}
    for k, v in id2label.items():
        idx = int(k)
        s = str(v).strip().lower()
        if "entail" in s and "not" not in s:
            norm_map["entailment"] = idx
        elif "neutral" in s:
            norm_map["neutral"] = idx
        elif "contradict" in s:
            norm_map["contradiction"] = idx
    if set(norm_map.keys()) != set(CANONICAL_LABELS):
        raise ValueError(f"Cannot resolve 3-class NLI labels from id2label={id2label}")
    remap = [norm_map["entailment"], norm_map["neutral"], norm_map["contradiction"]]
    return remap, {int(k): str(v) for k, v in id2label.items()}


def predict_batches(
    model,
    tokenizer,
    rows: List[dict],
    max_len: int,
    batch_size: int,
    remap: List[int],
    device: torch.device,
) -> Tuple[np.ndarray, int, float]:
    """バッチ推論し、正規化順確率 [N, 3]、切り詰め件数、経過秒数を返す。"""
    # 切り詰め件数確認（1件ずつではなく高速バッチ長チェック）
    truncated = 0
    for st in range(0, len(rows), 256):
        b = rows[st : st + 256]
        enc_full = tokenizer(
            [r["premise"] for r in b],
            [r["hypothesis"] for r in b],
            truncation=False,
            padding=False,
        )
        for ids in enc_full["input_ids"]:
            if len(ids) > max_len:
                truncated += 1

    probs_list = []
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for st in range(0, len(rows), batch_size):
            b = rows[st : st + batch_size]
            enc = tokenizer(
                [r["premise"] for r in b],
                [r["hypothesis"] for r in b],
                truncation="only_first",
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(**enc).logits
            logits = logits[:, remap].float()
            pr = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_list.append(pr)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    probs = np.vstack(probs_list) if probs_list else np.zeros((0, 3), dtype=np.float32)
    return probs, truncated, elapsed


def sanity_check(model, tokenizer, max_len: int, remap: List[int], device: torch.device) -> List[dict]:
    dummy_rows = [{"premise": p, "hypothesis": h, "label": l} for p, h, l in SANITY_EXAMPLES]
    probs, _, _ = predict_batches(model, tokenizer, dummy_rows, max_len, 8, remap, device)
    out = []
    for (_, _, gold), pr in zip(SANITY_EXAMPLES, probs):
        pred_lbl = CANONICAL_LABELS[int(np.argmax(pr))]
        out.append({
            "expected": gold,
            "predicted": pred_lbl,
            "ok": (pred_lbl == gold),
            "probs": {l: round(float(pr[i]), 4) for i, l in enumerate(CANONICAL_LABELS)},
        })
    return out


def compute_metrics_subset(y_true: np.ndarray, probs: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"n": 0}
    y_pred = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0))
    p_arr, r_arr, f_arr, s_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    n_total = len(y_true)
    n_e = int(np.sum(y_true == 0))
    n_c = int(np.sum(y_true == 2))
    e_to_c = int(cm[0, 2])
    c_to_e = int(cm[2, 0])
    flip_total = e_to_c + c_to_e

    # RAG 2値: entailment (0) vs non-entailment (1 or 2)
    y_bin = (y_true == 0).astype(int)
    p_ent = probs[:, 0]
    pred_bin = (y_pred == 0).astype(int)
    tp = int(np.sum((y_bin == 1) & (pred_bin == 1)))
    fp = int(np.sum((y_bin == 0) & (pred_bin == 1)))
    fn = int(np.sum((y_bin == 1) & (pred_bin == 0)))
    bin_p = tp / max(1, tp + fp)
    bin_r = tp / max(1, tp + fn)
    bin_f1 = (2 * bin_p * bin_r / (bin_p + bin_r)) if (bin_p + bin_r) > 0 else 0.0
    has_both = len(np.unique(y_bin)) == 2
    pr_auc = float(average_precision_score(y_bin, p_ent)) if has_both else None
    roc_auc = float(roc_auc_score(y_bin, p_ent)) if has_both else None

    return {
        "n": int(n_total),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            lbl: {
                "precision": round(float(p_arr[i]), 4),
                "recall": round(float(r_arr[i]), 4),
                "f1": round(float(f_arr[i]), 4),
                "support": int(s_arr[i]),
            }
            for i, lbl in enumerate(CANONICAL_LABELS)
        },
        "polarity_flip": {
            "e_to_c_count": e_to_c,
            "e_to_c_rate": round(e_to_c / max(1, n_e), 4),
            "c_to_e_count": c_to_e,
            "c_to_e_rate": round(c_to_e / max(1, n_c), 4),
            "ec_flip_count": flip_total,
            "ec_flip_rate_all": round(flip_total / max(1, n_total), 4),
            "ec_flip_rate_ec": round(flip_total / max(1, n_e + n_c), 4),
        },
        "confusion_matrix": cm.tolist(),
        "rag_binary_entailment": {
            "precision": round(bin_p, 4),
            "recall": round(bin_r, 4),
            "f1": round(bin_f1, 4),
            "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        },
    }


def compute_bert_token_lengths(rows: List[dict]) -> List[int]:
    """基準トークン長として tohoku-nlp/bert-base-japanese-v3 のトークン長を計算する。"""
    btok = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
    lens = []
    for st in range(0, len(rows), 256):
        b = rows[st : st + 256]
        enc = btok(
            [r["premise"] for r in b],
            [r["hypothesis"] for r in b],
            truncation=False,
            padding=False,
        )
        lens.extend(len(ids) for ids in enc["input_ids"])
    return lens


def evaluate_dataset(
    model,
    tokenizer,
    rows: List[dict],
    max_len: int,
    batch_size: int,
    remap: List[int],
    device: torch.device,
    bert_lens: List[int],
) -> Tuple[dict, List[dict]]:
    probs, truncated, elapsed = predict_batches(model, tokenizer, rows, max_len, batch_size, remap, device)
    y_true = np.array([LABEL2ID[r["label"]] for r in rows], dtype=np.int64)
    y_pred = np.argmax(probs, axis=1)

    overall = compute_metrics_subset(y_true, probs)
    overall["eval_sec"] = round(elapsed, 3)
    overall["ms_per_pair"] = round(elapsed * 1000.0 / max(1, len(rows)), 3)
    overall["pairs_per_sec"] = round(len(rows) / max(1e-6, elapsed), 2)
    overall["truncated_count"] = truncated

    # 層別集計
    def stratify(key_fn):
        groups = defaultdict(list)
        for idx, r in enumerate(rows):
            groups[key_fn(r, bert_lens[idx])].append(idx)
        out = {}
        for k, idxs in sorted(groups.items()):
            sub_m = compute_metrics_subset(y_true[idxs], probs[idxs])
            out[k] = {
                "n": sub_m["n"],
                "accuracy": sub_m.get("accuracy"),
                "macro_f1": sub_m.get("macro_f1"),
                "ec_flip_rate_all": sub_m.get("polarity_flip", {}).get("ec_flip_rate_all"),
                "rag_f1": sub_m.get("rag_binary_entailment", {}).get("f1"),
            }
        return out

    strat = {
        "by_len_bucket": stratify(lambda r, _: r.get("len_bucket", "unknown")),
        "by_bert512_boundary": stratify(lambda _, bl: "gt_512_bert_toks" if bl > 512 else "le_512_bert_toks"),
        "by_domain_group": stratify(lambda r, _: r.get("domain_group", "unknown")),
        "by_sector": stratify(lambda r, _: "oss" if r.get("domain_group") == "oss_docs" else "gov"),
        "by_subtype": stratify(lambda r, _: r.get("subtype") or "unknown"),
    }

    preds_detail = []
    for idx, r in enumerate(rows):
        preds_detail.append({
            "id": r["id"],
            "label": r["label"],
            "pred": CANONICAL_LABELS[int(y_pred[idx])],
            "correct": bool(y_true[idx] == y_pred[idx]),
            "probs": [round(float(x), 4) for x in probs[idx]],
            "bert_tokens": bert_lens[idx],
        })

    return {"overall": overall, "stratified": strat}, preds_detail


def run_eval_cli(
    model_path: str,
    run_name: str,
    eval_sets: List[str],
    out_dir: Path,
    max_len: int,
    batch_size: int,
) -> dict:
    for s in eval_sets:
        if "sealed" in s.lower():
            raise ValueError("SECURITY GUARD: sealed-test must NEVER be evaluated in Stage 1!")

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model_max_pos = getattr(model.config, "max_position_embeddings", None)
    eff_max_len = min(max_len, int(model_max_pos)) if (model_max_pos and int(model_max_pos) > 0) else max_len
    remap, orig_id2label = resolve_label_remap(model.config)
    n_params = sum(p.numel() for p in model.parameters())

    sanity = sanity_check(model, tok, eff_max_len, remap, device)
    print(f"[{run_name}] id2label={orig_id2label} -> remap={remap} | sanity_ok={all(x['ok'] for x in sanity)}")
    for sx in sanity:
        print("  sanity:", sx)

    set_paths = {
        "human_eval": ROOT / "data" / "eval" / "human-eval.jsonl",
        "dev": ROOT / "data" / "pilot" / "dev.jsonl",
    }

    results = {
        "run_name": run_name,
        "model_path": model_path,
        "num_params": n_params,
        "max_len": eff_max_len,
        "model_max_position_embeddings": model_max_pos,
        "orig_id2label": orig_id2label,
        "canonical_remap_indices": remap,
        "sanity_check": sanity,
        "datasets": {},
    }

    for ds_name in eval_sets:
        p = set_paths[ds_name]
        with p.open("r", encoding="utf-8") as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        bert_lens = compute_bert_token_lengths(rows)
        ds_res, preds = evaluate_dataset(
            model, tok, rows, eff_max_len, batch_size, remap, device, bert_lens
        )
        results["datasets"][ds_name] = ds_res
        (out_dir / f"preds_{ds_name}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in preds) + "\n",
            encoding="utf-8",
        )
        ov = ds_res["overall"]
        print(
            f"[{run_name}] {ds_name} (n={ov['n']}): "
            f"acc={ov['accuracy']:.4f}, macro_f1={ov['macro_f1']:.4f}, "
            f"ec_flip_all={ov['polarity_flip']['ec_flip_rate_all']:.4f}, "
            f"rag_f1={ov['rag_binary_entailment']['f1']:.4f}, "
            f"trunc={ov['truncated_count']}, {ov['ms_per_pair']:.2f}ms/pair"
        )

    out_json = out_dir / "eval_results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--eval-sets", default="human_eval,dev")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    eval_sets = [x.strip() for x in args.eval_sets.split(",") if x.strip()]
    run_eval_cli(args.model, args.name, eval_sets, Path(args.out_dir), args.max_len, args.batch_size)


if __name__ == "__main__":
    main()
