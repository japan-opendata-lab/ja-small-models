#!/usr/bin/env python3
"""第5号 ja-phish-guard-v1 封印テスト評価スクリプト。

dev データで dev 再現確認と FPR<=1% 厳密閾値 thr_fpr1 を復元・固定し、
勝者モデル（bert-v3）で sealed-test.jsonl（2,990件）を評価して JSON へ保存する。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["benign", "benign_hard", "fraud", "phishing", "spam"]
ATTACK = {"phishing", "fraud", "spam"}
DEV_HOLDOUT = {"phish_password", "fraud_prize"}
SEALED_HOLDOUT = ["phish_sms_code", "fraud_rent"]
FRAUD_FAMILIES = [
    "fraud_invest",
    "fraud_job",
    "fraud_task",
    "fraud_romance",
    "fraud_unpaid",
    "fraud_rent",
]


def load_rows(path: Path) -> list[dict]:
    """JSONL ファイルを読み込む。"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def infer_probs(
    rows: list[dict], tok, model, device: str, max_len: int = 512
) -> tuple[np.ndarray, float]:
    """train_pilot_cls.py と同一の単一文 FP32 推論で確率配列と総所要秒を返す。"""
    probs = []
    t0 = time.time()
    with torch.no_grad():
        for r in rows:
            enc = tok(
                r["text"], truncation=True, max_length=max_len, return_tensors="pt"
            ).to(device)
            p = torch.softmax(model(**enc).logits, -1)[0].cpu().numpy()
            probs.append(p)
    elapsed = time.time() - t0
    return np.array(probs, dtype=np.float64), elapsed


def eval_at_threshold(
    risk: np.ndarray, ys: list[str], fams: list[str], thr: float
) -> dict:
    """指定閾値における発見率・誤検出率・サブグループ・ファミリー別指標を算出する。"""
    ys_arr = np.array(ys)
    fams_arr = np.array(fams)
    y_atk = np.array([1 if y in ATTACK else 0 for y in ys], dtype=int)
    pred = (risk >= thr).astype(int)

    tp = int(((pred == 1) & (y_atk == 1)).sum())
    fp = int(((pred == 1) & (y_atk == 0)).sum())
    fn = int(((pred == 0) & (y_atk == 1)).sum())
    tn = int(((pred == 0) & (y_atk == 0)).sum())

    det = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    prec = tp / max(1, tp + fp)
    f1 = (2 * prec * det / (prec + det)) if (prec + det) > 0 else 0.0

    neg_sub = {}
    for neg_lbl in ("benign_hard", "benign"):
        m = ys_arr == neg_lbl
        cnt = int(m.sum())
        n_fp = int((pred[m] == 1).sum())
        neg_sub[neg_lbl] = {
            "fp": n_fp,
            "total": cnt,
            "fpr": round(n_fp / max(1, cnt), 4),
        }

    holdout_set = set(SEALED_HOLDOUT)
    m_hold = np.array([f in holdout_set for f in fams])
    m_known_atk = (y_atk == 1) & (~m_hold)

    hold_by_fam = {}
    for hf in SEALED_HOLDOUT:
        mf = fams_arr == hf
        tot_f = int(mf.sum())
        det_f = int((pred[mf] == 1).sum())
        hold_by_fam[hf] = {
            "detected": det_f,
            "total": tot_f,
            "detection_rate": round(det_f / max(1, tot_f), 4) if tot_f else None,
        }

    per_family = {}
    for fn_name in sorted(set(fams)):
        mf = fams_arr == fn_name
        tot_f = int(mf.sum())
        is_atk = bool(y_atk[mf][0] == 1)
        pos_f = int((pred[mf] == 1).sum())
        per_family[fn_name] = {
            "kind": "attack" if is_atk else "negative",
            "flagged": pos_f,
            "total": tot_f,
            "rate": round(pos_f / max(1, tot_f), 4),
        }

    return {
        "threshold": float(thr),
        "threshold_rounded_4": round(float(thr), 4),
        "detection_rate": round(float(det), 4),
        "fpr": round(float(fpr), 4),
        "precision": round(float(prec), 4),
        "f1": round(float(f1), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fpr_benign_hard": neg_sub["benign_hard"]["fpr"],
        "benign_hard_detail": neg_sub["benign_hard"],
        "fpr_benign": neg_sub["benign"]["fpr"],
        "benign_detail": neg_sub["benign"],
        "known_attack_detection": {
            "detected": int((pred[m_known_atk] == 1).sum()),
            "total": int(m_known_atk.sum()),
            "detection_rate": round(
                float((pred[m_known_atk] == 1).mean()), 4
            )
            if m_known_atk.any()
            else None,
        },
        "holdout_detection": {
            "detected": int((pred[m_hold] == 1).sum()),
            "total": int(m_hold.sum()),
            "detection_rate": round(float((pred[m_hold] == 1).mean()), 4)
            if m_hold.any()
            else None,
        },
        "holdout_by_family": hold_by_fam,
        "per_family": per_family,
    }


def eval_5class(probs: np.ndarray, ys: list[str], fams: list[str]) -> dict:
    """argmax による 5 クラス精度・混同行列・fraud 系ファミリー内訳を算出する。"""
    l2i = {l: i for i, l in enumerate(LABELS)}
    y_idx = np.array([l2i[y] for y in ys], dtype=int)
    pred_idx = probs.argmax(axis=1)

    per_class_acc = {}
    per_class_counts = {}
    for lbl in LABELS:
        idx = l2i[lbl]
        m = y_idx == idx
        tot = int(m.sum())
        corr = int((pred_idx[m] == idx).sum())
        acc = round(corr / max(1, tot), 4)
        per_class_acc[lbl] = acc
        per_class_counts[lbl] = {"correct": corr, "total": tot, "accuracy": acc}

    cm = confusion_matrix(y_idx, pred_idx, labels=list(range(len(LABELS)))).tolist()

    fams_arr = np.array(fams)
    fraud_breakdown = {}
    for ff in FRAUD_FAMILIES:
        mf = fams_arr == ff
        if not mf.any():
            continue
        tot = int(mf.sum())
        corr = int((pred_idx[mf] == l2i["fraud"]).sum())
        pred_dist = {
            LABELS[i]: int((pred_idx[mf] == i).sum())
            for i in range(len(LABELS))
            if (pred_idx[mf] == i).any()
        }
        fraud_breakdown[ff] = {
            "correct": corr,
            "total": tot,
            "accuracy": round(corr / max(1, tot), 4),
            "predicted_labels": pred_dist,
        }

    return {
        "labels": LABELS,
        "overall_accuracy": round(float((pred_idx == y_idx).mean()), 4),
        "per_class_accuracy": per_class_acc,
        "per_class_counts": per_class_counts,
        "confusion_matrix": cm,
        "fraud_family_breakdown": fraud_breakdown,
    }


def compute_auc_metrics(risk: np.ndarray, ys: list[str], fams: list[str]) -> dict:
    """リスクスコアの ROC-AUC および PR-AUC 群を算出する。"""
    ys_arr = np.array(ys)
    y_atk = np.array([1 if y in ATTACK else 0 for y in ys], dtype=int)

    roc_all = float(roc_auc_score(y_atk, risk))
    pr_all = float(average_precision_score(y_atk, risk))

    m_bh = (y_atk == 1) | (ys_arr == "benign_hard")
    roc_bh = float(roc_auc_score(y_atk[m_bh], risk[m_bh]))

    holdout_set = set(SEALED_HOLDOUT)
    m_hold = np.array([f in holdout_set for f in fams])
    m_hn = m_hold | (y_atk == 0)
    roc_hold_neg = float(roc_auc_score(m_hold[m_hn].astype(int), risk[m_hn]))

    return {
        "roc_auc_all": round(roc_all, 6),
        "pr_auc_all": round(pr_all, 6),
        "roc_auc_vs_benign_hard": round(roc_bh, 6),
        "roc_auc_holdout_vs_neg": round(roc_hold_neg, 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--sealed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probs-dir", required=True)
    args = ap.parse_args()

    dev_rows = load_rows(Path(args.dev))
    sealed_rows = load_rows(Path(args.sealed))
    observed_labels = sorted({r["label"] for r in dev_rows + sealed_rows})
    assert observed_labels == LABELS, f"Label order mismatch: {observed_labels}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device).eval()

    l2i = {l: i for i, l in enumerate(LABELS)}
    atk_idx = [l2i[l] for l in LABELS if l in ATTACK]

    # 手順A: dev 再現と厳密 FPR<=1% 閾値復元
    dev_probs, dev_sec = infer_probs(dev_rows, tok, model, device)
    probs_dir = Path(args.probs_dir)
    probs_dir.mkdir(parents=True, exist_ok=True)
    np.save(probs_dir / "dev_probs.npy", dev_probs)

    dev_ys = [r["label"] for r in dev_rows]
    dev_fams = [r.get("family", "") for r in dev_rows]
    dev_risk = dev_probs[:, atk_idx].sum(axis=1)
    dev_y_atk = np.array([1 if y in ATTACK else 0 for y in dev_ys], dtype=int)

    neg_sorted = np.sort(dev_risk[dev_y_atk == 0])[::-1]
    k = max(1, int(0.01 * len(neg_sorted)))
    thr_fpr1 = float(neg_sorted[min(k, len(neg_sorted) - 1)]) + 1e-6

    dev_pred_cls = dev_probs.argmax(axis=1)
    dev_per_cls = {
        lbl: round(
            float((dev_pred_cls[np.array(dev_ys) == lbl] == l2i[lbl]).mean()), 4
        )
        for lbl in LABELS
    }
    dev_hm = np.array([f in DEV_HOLDOUT for f in dev_fams])
    dev_repro = {
        "dev_rows": len(dev_rows),
        "dev_eval_sec_per_doc": round(dev_sec / len(dev_rows), 4),
        "thr_fpr1_exact": thr_fpr1,
        "thr_fpr1_rounded_4": round(thr_fpr1, 4),
        "k_index": k,
        "neg_count": int(len(neg_sorted)),
        "dev_at_0.5": {
            "detection_rate": round(
                float(((dev_risk >= 0.5) & (dev_y_atk == 1)).sum() / dev_y_atk.sum()),
                4,
            ),
            "fpr": round(
                float(
                    ((dev_risk >= 0.5) & (dev_y_atk == 0)).sum()
                    / (dev_y_atk == 0).sum()
                ),
                4,
            ),
        },
        "dev_at_fpr1": {
            "detection_rate": round(
                float(
                    ((dev_risk >= thr_fpr1) & (dev_y_atk == 1)).sum()
                    / dev_y_atk.sum()
                ),
                4,
            ),
            "fpr": round(
                float(
                    ((dev_risk >= thr_fpr1) & (dev_y_atk == 0)).sum()
                    / (dev_y_atk == 0).sum()
                ),
                4,
            ),
        },
        "dev_per_class_accuracy": dev_per_cls,
        "dev_holdout_detection_at_fpr1": round(
            float((dev_risk[dev_hm] >= thr_fpr1).mean()), 4
        ),
    }

    # 手順B: 封印テスト 2,990 件の最終評価
    sealed_probs, sealed_sec = infer_probs(sealed_rows, tok, model, device)
    np.save(probs_dir / "sealed_probs.npy", sealed_probs)

    sealed_ys = [r["label"] for r in sealed_rows]
    sealed_fams = [r.get("family", "") for r in sealed_rows]
    sealed_risk = sealed_probs[:, atk_idx].sum(axis=1)

    res_fpr1 = eval_at_threshold(sealed_risk, sealed_ys, sealed_fams, thr_fpr1)
    res_rounded = eval_at_threshold(
        sealed_risk, sealed_ys, sealed_fams, round(thr_fpr1, 4)
    )
    res_05 = eval_at_threshold(sealed_risk, sealed_ys, sealed_fams, 0.5)
    cls_5 = eval_5class(sealed_probs, sealed_ys, sealed_fams)
    aucs = compute_auc_metrics(sealed_risk, sealed_ys, sealed_fams)

    gate_check = {
        "threshold_used": thr_fpr1,
        "fpr_le_0.01": bool(res_fpr1["fpr"] <= 0.01),
        "detection_rate_ge_0.90": bool(res_fpr1["detection_rate"] >= 0.90),
        "fpr_benign_hard_le_0.02": bool(res_fpr1["fpr_benign_hard"] <= 0.02),
        "all_passed": bool(
            res_fpr1["fpr"] <= 0.01
            and res_fpr1["detection_rate"] >= 0.90
            and res_fpr1["fpr_benign_hard"] <= 0.02
        ),
    }

    report = {
        "model": args.model,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "sealed_rows": len(sealed_rows),
        "eval_sec": round(sealed_sec, 2),
        "eval_sec_per_doc": round(sealed_sec / len(sealed_rows), 4),
        "dev_reproduction": dev_repro,
        "at_dev_fpr1": res_fpr1,
        "at_dev_fpr1_rounded_0.0001": res_rounded,
        "at_0.5": res_05,
        "classification_5class": cls_5,
        "auc": aucs,
        "spec_v1_gate_check": gate_check,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
