#!/usr/bin/env python3
"""第12号 ja-sentiment-v1 共通評価スクリプト（3クラス全体＋中立除外2クラス比較併記）。

統一ラベル空間: {0: negative, 1: neutral, 2: positive}
※ sealed-test は安全装置により評価対象から除外。
"""
import argparse
import json
import time
from collections import defaultdict
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
CANONICAL_LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(CANONICAL_LABELS)}

SANITY_EXAMPLES = [
    ("とても使いやすく大満足です。毎日愛用しています。", "positive"),
    ("昨日届き、サイズは幅30センチでした。", "neutral"),
    ("すぐに壊れてしまい、まったく使い物になりません。", "negative"),
]


def resolve_label_remap(config) -> Tuple[str, List[int], Dict[int, str]]:
    """モデルの id2label を検査し、3クラスまたは2クラスの canonical index 対応を返す。"""
    id2label = getattr(config, "id2label", None) or {}
    norm_map: Dict[str, int] = {}
    for k, v in id2label.items():
        idx = int(k)
        s = str(v).strip().lower()
        if "neg" in s or "ネガ" in s:
            norm_map["negative"] = idx
        elif "neu" in s or "中立" in s:
            norm_map["neutral"] = idx
        elif "pos" in s or "ポジ" in s:
            norm_map["positive"] = idx

    if set(norm_map.keys()) == set(CANONICAL_LABELS):
        remap = [norm_map["negative"], norm_map["neutral"], norm_map["positive"]]
        return "3class", remap, {int(k): str(v) for k, v in id2label.items()}
    if set(norm_map.keys()) == {"negative", "positive"}:
        remap = [norm_map["negative"], norm_map["positive"]]
        return "2class", remap, {int(k): str(v) for k, v in id2label.items()}
    raise ValueError(f"Cannot resolve sentiment labels from id2label={id2label}")


def predict_batches(
    model,
    tokenizer,
    texts: List[str],
    max_len: int,
    batch_size: int,
    mode: str,
    remap: List[int],
    device: torch.device,
) -> Tuple[np.ndarray, int, float]:
    """バッチ推論し、3クラス正規化確率 [N, 3]（2クラスモデル時は neutral=0.0）、切り詰め件数、経過秒数を返す。"""
    truncated = 0
    for st in range(0, len(texts), 256):
        enc_full = tokenizer(texts[st : st + 256], truncation=False, padding=False)
        for ids in enc_full["input_ids"]:
            if len(ids) > max_len:
                truncated += 1

    probs_list = []
    model.eval()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for st in range(0, len(texts), batch_size):
            b = texts[st : st + batch_size]
            enc = tokenizer(
                b,
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**enc).logits
            logits = logits[:, remap].float()
            pr = torch.softmax(logits, dim=-1).cpu().numpy()
            if mode == "2class":
                # [neg, pos] -> [neg, 0.0, pos]
                full = np.zeros((pr.shape[0], 3), dtype=np.float32)
                full[:, 0] = pr[:, 0]
                full[:, 2] = pr[:, 1]
                pr = full
            probs_list.append(pr)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    probs = np.vstack(probs_list) if probs_list else np.zeros((0, 3), dtype=np.float32)
    return probs, truncated, elapsed


def sanity_check(model, tokenizer, max_len: int, mode: str, remap: List[int], device: torch.device) -> List[dict]:
    texts = [t for t, _ in SANITY_EXAMPLES]
    probs, _, _ = predict_batches(model, tokenizer, texts, max_len, 8, mode, remap, device)
    out = []
    for (_, gold), pr in zip(SANITY_EXAMPLES, probs):
        pred_lbl = CANONICAL_LABELS[int(np.argmax(pr))]
        out.append({
            "expected": gold,
            "predicted": pred_lbl,
            "ok": (pred_lbl == gold),
            "probs": {l: round(float(pr[i]), 4) for i, l in enumerate(CANONICAL_LABELS)},
        })
    return out


def compute_binary_no_neutral(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """中立除外 2クラス比較（正解が negative または positive の部分集合で p_pos/(p_neg+p_pos) を評価）。"""
    mask = (y_true == 0) | (y_true == 2)
    if int(np.sum(mask)) == 0:
        return {"n": 0}
    yt = (y_true[mask] == 2).astype(int)  # 0=neg, 1=pos
    pn = probs[mask, 0]
    pp = probs[mask, 2]
    denom = np.maximum(pn + pp, 1e-12)
    p_pos = pp / denom
    yp = (p_pos >= 0.5).astype(int)

    acc = float(accuracy_score(yt, yp))
    macro_f1 = float(f1_score(yt, yp, labels=[0, 1], average="macro", zero_division=0))
    p_arr, r_arr, f_arr, s_arr = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    flip_cnt = int(cm[0, 1] + cm[1, 0])
    has_both = len(np.unique(yt)) == 2
    roc_auc = float(roc_auc_score(yt, p_pos)) if has_both else None
    pr_auc = float(average_precision_score(yt, p_pos)) if has_both else None

    return {
        "n": int(len(yt)),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "negative": {"precision": round(float(p_arr[0]), 4), "recall": round(float(r_arr[0]), 4), "f1": round(float(f_arr[0]), 4), "support": int(s_arr[0])},
        "positive": {"precision": round(float(p_arr[1]), 4), "recall": round(float(r_arr[1]), 4), "f1": round(float(f_arr[1]), 4), "support": int(s_arr[1])},
        "neg_to_pos_count": int(cm[0, 1]),
        "pos_to_neg_count": int(cm[1, 0]),
        "pn_flip_count": flip_cnt,
        "pn_flip_rate": round(flip_cnt / max(1, len(yt)), 4),
        "roc_auc_pos": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc_pos": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix_neg_pos": cm.tolist(),
    }


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
    n_neg = int(np.sum(y_true == 0))
    n_pos = int(np.sum(y_true == 2))
    neg_to_pos = int(cm[0, 2])
    pos_to_neg = int(cm[2, 0])
    flip_total = neg_to_pos + pos_to_neg

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
            "neg_to_pos_count": neg_to_pos,
            "neg_to_pos_rate": round(neg_to_pos / max(1, n_neg), 4),
            "pos_to_neg_count": pos_to_neg,
            "pos_to_neg_rate": round(pos_to_neg / max(1, n_pos), 4),
            "pn_flip_count": flip_total,
            "pn_flip_rate_all": round(flip_total / max(1, n_total), 4),
            "pn_flip_rate_pn": round(flip_total / max(1, n_neg + n_pos), 4),
        },
        "confusion_matrix": cm.tolist(),
        "binary_no_neutral": compute_binary_no_neutral(y_true, probs),
    }


def evaluate_dataset(
    model,
    tokenizer,
    rows: List[dict],
    max_len: int,
    batch_size: int,
    mode: str,
    remap: List[int],
    device: torch.device,
    include_calibration: bool = False,
) -> Tuple[dict, List[dict]]:
    texts = [r["text"] for r in rows]
    probs, truncated, elapsed = predict_batches(model, tokenizer, texts, max_len, batch_size, mode, remap, device)
    y_true = np.array([LABEL2ID[r["label"]] for r in rows], dtype=np.int64)
    y_pred = np.argmax(probs, axis=1)

    overall = compute_metrics_subset(y_true, probs)
    if include_calibration:
        overall["calibration"] = compute_calibration_metrics(y_true, probs, n_bins=15)
    total_chars = sum(len(t) for t in texts)
    overall["eval_sec"] = round(elapsed, 3)
    overall["ms_per_sample"] = round(elapsed * 1000.0 / max(1, len(rows)), 3)
    overall["samples_per_sec"] = round(len(rows) / max(1e-6, elapsed), 2)
    overall["chars_per_sec"] = round(total_chars / max(1e-6, elapsed), 1)
    overall["truncated_count"] = truncated

    def stratify(key_fn):
        groups = defaultdict(list)
        for idx, r in enumerate(rows):
            groups[key_fn(r)].append(idx)
        out = {}
        for k, idxs in sorted(groups.items()):
            sub_m = compute_metrics_subset(y_true[idxs], probs[idxs])
            # 存在するクラスだけでマクロ F1 も出す（サブタイプ単一クラス対応）
            present_labels = sorted(set(int(x) for x in y_true[idxs]))
            present_f1 = float(f1_score(y_true[idxs], y_pred[idxs], labels=present_labels, average="macro", zero_division=0))
            out[k] = {
                "n": sub_m["n"],
                "accuracy": sub_m.get("accuracy"),
                "macro_f1": sub_m.get("macro_f1"),
                "present_macro_f1": round(present_f1, 4),
                "pn_flip_rate_all": sub_m.get("polarity_flip", {}).get("pn_flip_rate_all"),
                "binary_acc": sub_m.get("binary_no_neutral", {}).get("accuracy"),
                "binary_macro_f1": sub_m.get("binary_no_neutral", {}).get("macro_f1"),
            }
        return out

    strat = {
        "by_domain": stratify(lambda r: r.get("domain", "unknown")),
        "by_difficulty": stratify(lambda r: r.get("difficulty", "standard")),
        "by_subtype": stratify(lambda r: r.get("subtype") or "unknown"),
        "by_length_band": stratify(lambda r: r.get("length_band", "unknown")),
    }

    preds_detail = []
    for idx, r in enumerate(rows):
        preds_detail.append({
            "id": r["id"],
            "label": r["label"],
            "pred": CANONICAL_LABELS[int(y_pred[idx])],
            "correct": bool(y_true[idx] == y_pred[idx]),
            "probs": [round(float(x), 4) for x in probs[idx]],
            "domain": r.get("domain", ""),
            "subtype": r.get("subtype", ""),
            "difficulty": r.get("difficulty", ""),
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
    mode, remap, orig_id2label = resolve_label_remap(model.config)
    n_params = sum(p.numel() for p in model.parameters())

    sanity = sanity_check(model, tok, eff_max_len, mode, remap, device)
    print(f"[{run_name}] mode={mode}, id2label={orig_id2label} -> remap={remap} | sanity_ok={all(x['ok'] for x in sanity)}")
    for sx in sanity:
        print("  sanity:", sx)

    set_paths = {
        "human_eval": ROOT / "data" / "eval" / "human-eval.jsonl",
        "dev": ROOT / "data" / "pilot" / "dev.jsonl",
    }

    results = {
        "run_name": run_name,
        "model_path": model_path,
        "label_mode": mode,
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
        rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        ds_res, preds = evaluate_dataset(
            model, tok, rows, eff_max_len, batch_size, mode, remap, device
        )
        results["datasets"][ds_name] = ds_res
        (out_dir / f"preds_{ds_name}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in preds) + "\n",
            encoding="utf-8",
        )
        ov = ds_res["overall"]
        bin_m = ov["binary_no_neutral"]
        print(
            f"[{run_name}] {ds_name} (n={ov['n']}): "
            f"3cls_acc={ov['accuracy']:.4f}, 3cls_macro_f1={ov['macro_f1']:.4f}, "
            f"pn_flip={ov['polarity_flip']['pn_flip_rate_all']:.4f}, "
            f"bin2cls_acc={bin_m['accuracy']:.4f}, bin2cls_f1={bin_m['macro_f1']:.4f}, "
            f"trunc={ov['truncated_count']}, {ov['ms_per_sample']:.2f}ms/sample"
        )

    out_json = out_dir / "eval_results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def compute_calibration_metrics(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict:
    """3クラス確率に対する NLL・Brier スコア・15ビン ECE を算出する。"""
    n = len(y_true)
    if n == 0:
        return {"n": 0, "nll": None, "brier_score": None, "ece_15bins": None}
    p_clip = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    p_true = p_clip[np.arange(n), y_true]
    nll = float(-np.mean(np.log(p_true)))

    one_hot = np.zeros_like(p_clip)
    one_hot[np.arange(n), y_true] = 1.0
    brier = float(np.mean(np.sum((p_clip - one_hot) ** 2, axis=1)))

    confs = np.max(p_clip, axis=1)
    preds = np.argmax(p_clip, axis=1)
    corrects = (preds == y_true).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for bi in range(n_bins):
        lo, hi = edges[bi], edges[bi + 1]
        if bi == 0:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs > lo) & (confs <= hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / n) * abs(float(np.mean(corrects[mask])) - float(np.mean(confs[mask])))

    return {
        "n": int(n),
        "nll": round(nll, 5),
        "brier_score": round(brier, 5),
        "ece_15bins": round(float(ece), 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--eval-sets", default="human_eval,dev")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    eval_sets = [x.strip() for x in args.eval_sets.split(",") if x.strip()]
    run_eval_cli(args.model, args.name, eval_sets, Path(args.out_dir), args.max_len, args.batch_size)


if __name__ == "__main__":
    main()
