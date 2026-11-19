#!/usr/bin/env python3
"""第4号 ja-doc-classify-v1 (v2モデル) カバレッジ対精度 & 高機密フラグ閾値分析スクリプト。

再学習は一切行わず、既存の学習済み v2 モデル（runs/prod-modernbert-v2/model_state.pt）を用いて
OOD v1（200件）および OOD v2（200件）に対する推論確率を算出し、以下を計測・保存する。
1. カバレッジ対精度曲線（種別・領域 × OOD v1・OOD v2、棄却率 0%/10%/20%/30%/40% および固定閾値 τ）
2. 高機密フラグの閾値調整曲線（OOD v2 主分析・OOD v1 参考、棄却連動フォールバック、FN 詳細）

※ 封印テスト（data/pilot/sealed-test.jsonl）は一切参照しない。
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from transformers import AutoModel, AutoTokenizer

DOMAINS = ["金融", "法務", "医療", "教育", "EC", "雇用", "通信・IT", "行政"]
FORMATS = ["structured", "unstructured"]
HIGH_RISK = {
    "給付申請書", "入院同意書", "口座開設申込書", "住民異動届", "本人確認書類一覧",
    "振込依頼書", "学生記録票", "保護者連絡票", "処方箋", "検査結果報告書",
    "税務申告関連書類", "診療情報提供書", "成績通知書", "診断書", "本人確認台帳",
    "クレジットカード利用明細", "給与連絡票", "本人確認記録", "海外送金依頼書",
    "ローン審査書類", "奨学金申請書", "退職手続書類",
}

REJECT_RATES = [0.0, 0.10, 0.20, 0.30, 0.40]
FIXED_TAUS = [0.0, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995]
HR_THRESHOLDS = [
    0.90, 0.80, 0.50, 0.40, 0.30, 0.20, 0.10,
    0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.0,
]
TARGET_RECALLS = [0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]


class MultiHead(nn.Module):
    """共有エンコーダ＋3ヘッド分類モデル（eval_cls_multi.py と同一定義）。"""

    def __init__(self, base: str, n_types: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        h = self.encoder.config.hidden_size
        self.head_domain = nn.Linear(h, len(DOMAINS))
        self.head_type = nn.Linear(h, n_types)
        self.head_format = nn.Linear(h, len(FORMATS))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kw):
        kw_in = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self.encoder.config.model_type == "bert":
            kw_in["token_type_ids"] = token_type_ids
        out = self.encoder(**kw_in)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]
        return {
            "logits_domain": self.head_domain(pooled),
            "logits_type": self.head_type(pooled),
            "logits_format": self.head_format(pooled),
        }


def load_jsonl(path: Path) -> list[dict]:
    """JSONL ファイルを読み込む。"""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_device(pref: str) -> tuple[str, str]:
    """実行デバイスを選択する（--device auto 時は空き VRAM >= 4 GiB で cuda、それ以外は cpu）。"""
    if pref != "auto":
        return pref, f"explicit:{pref}"
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gib = free_bytes / (1024 ** 3)
        total_gib = total_bytes / (1024 ** 3)
        if free_bytes >= 4 * (1024 ** 3):
            return "cuda", f"cuda (free {free_gib:.2f}/{total_gib:.2f} GiB)"
        return "cpu", f"cpu fallback (cuda free {free_gib:.2f} GiB < 4.0 GiB)"
    return "cpu", "cpu (cuda unavailable)"


def calc_log1m_conf(logits_1d: torch.Tensor) -> float:
    """float64 softmax の飽和を回避するため、log(1 - max_prob) = logsumexp(non_max) - logsumexp(all) を算出する。"""
    z = logits_1d.to(torch.float64)
    max_idx = int(torch.argmax(z))
    if z.numel() <= 1:
        return float("-inf")
    mask = torch.ones(z.numel(), dtype=torch.bool, device=z.device)
    mask[max_idx] = False
    log_all = torch.logsumexp(z, dim=0)
    log_other = torch.logsumexp(z[mask], dim=0)
    return float((log_other - log_all).item())


def run_inference(
    model: MultiHead,
    tok,
    rows: list[dict],
    types: list[str],
    hr_idx: list[int],
    device: str,
    max_len: int,
) -> tuple[float, list[dict]]:
    """バッチサイズ1で推論を実行し、文書ごとの高精度確率レコードを返す。"""
    t2i = {t: i for i, t in enumerate(types)}
    d2i = {d: i for i, d in enumerate(DOMAINS)}
    hr_idx_t = torch.tensor(hr_idx, dtype=torch.long)

    records: list[dict] = []
    t0 = time.time()
    with torch.no_grad():
        for row_idx, r in enumerate(rows):
            enc = tok(r["text"], truncation=True, max_length=max_len, return_tensors="pt").to(device)
            out = model(**enc)

            logits_type_1d = out["logits_type"][0].detach().cpu()
            logits_domain_1d = out["logits_domain"][0].detach().cpu()

            probs_type = torch.softmax(logits_type_1d.to(torch.float64), dim=-1)
            probs_domain = torch.softmax(logits_domain_1d.to(torch.float64), dim=-1)

            pred_type_idx = int(torch.argmax(logits_type_1d).item())
            pred_domain_idx = int(torch.argmax(logits_domain_1d).item())
            gold_type_idx = t2i[r["document_type"]]
            gold_domain_idx = d2i[r["domain"]]

            conf_type = float(probs_type[pred_type_idx].item())
            conf_domain = float(probs_domain[pred_domain_idx].item())
            log1m_conf_type = calc_log1m_conf(logits_type_1d)
            log1m_conf_domain = calc_log1m_conf(logits_domain_1d)

            p_hr_sum = float(probs_type[hr_idx_t].sum().item())
            p_hr_max = float(probs_type[hr_idx_t].max().item())

            top3_vals, top3_indices = torch.topk(probs_type, k=3)
            top3_types = [
                {"type": types[int(idx)], "prob": float(val)}
                for val, idx in zip(top3_vals.tolist(), top3_indices.tolist())
            ]

            gold_hr = int(r["document_type"] in HIGH_RISK)
            if "risk_tier" in r:
                assert (r["risk_tier"] == "高機密") == bool(gold_hr), (
                    f"risk_tier mismatch at {r['id']}: {r['risk_tier']} vs HIGH_RISK"
                )

            rec = {
                "row_idx": row_idx,
                "id": r["id"],
                "style": r.get("style", ""),
                "gold_type": r["document_type"],
                "gold_type_idx": gold_type_idx,
                "pred_type": types[pred_type_idx],
                "pred_type_idx": pred_type_idx,
                "conf_type": conf_type,
                "log1m_conf_type": log1m_conf_type,
                "gold_domain": r["domain"],
                "gold_domain_idx": gold_domain_idx,
                "pred_domain": DOMAINS[pred_domain_idx],
                "pred_domain_idx": pred_domain_idx,
                "conf_domain": conf_domain,
                "log1m_conf_domain": log1m_conf_domain,
                "gold_high_risk": gold_hr,
                "pred_high_risk_argmax": int(types[pred_type_idx] in HIGH_RISK),
                "p_high_risk_sum": p_hr_sum,
                "p_high_risk_max": p_hr_max,
                "top3_types": top3_types,
            }
            # NaN/Inf チェック
            for k in ("conf_type", "log1m_conf_type", "conf_domain", "log1m_conf_domain", "p_high_risk_sum", "p_high_risk_max"):
                assert not math.isnan(rec[k]), f"NaN in {k} at {r['id']}"
            records.append(rec)

    eval_sec = time.time() - t0
    return eval_sec, records


def save_probs_jsonl(path: Path, records: list[dict]) -> None:
    """計画指定のフィールド構成で文書別確率 JSONL を保存する。"""
    keys = [
        "id",
        "style",
        "gold_type",
        "pred_type",
        "conf_type",
        "log1m_conf_type",
        "gold_domain",
        "pred_domain",
        "conf_domain",
        "log1m_conf_domain",
        "gold_high_risk",
        "pred_high_risk_argmax",
        "p_high_risk_sum",
        "p_high_risk_max",
        "top3_types",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            obj = {k: r[k] for k in keys}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def verify_parity(name: str, records: list[dict], ref_json_path: Path) -> dict:
    """0% 棄却時の指標が Colab L4 既存評価 JSON と小数4桁で完全一致することを検証する。"""
    ref = json.loads(ref_json_path.read_text(encoding="utf-8"))
    yt = np.array([r["gold_type_idx"] for r in records])
    pt = np.array([r["pred_type_idx"] for r in records])
    yd = np.array([r["gold_domain_idx"] for r in records])
    pd = np.array([r["pred_domain_idx"] for r in records])
    hr_g = np.array([r["gold_high_risk"] for r in records])
    hr_p = np.array([r["pred_high_risk_argmax"] for r in records])

    present_idx = sorted(set(yt.tolist()))
    got = {
        "ood_rows": len(records),
        "type_accuracy": round(float(accuracy_score(yt, pt)), 4),
        "type_macro_f1_covered": round(float(f1_score(yt, pt, labels=present_idx, average="macro", zero_division=0)), 4),
        "domain_accuracy": round(float(accuracy_score(yd, pd)), 4),
        "domain_macro_f1": round(float(f1_score(yd, pd, average="macro", zero_division=0)), 4),
        "high_risk_recall": round(float(recall_score(hr_g, hr_p, zero_division=0)), 4),
    }
    for k, v in got.items():
        assert v == ref[k], f"[{name}] Parity mismatch on {k}: local={v} != ref={ref[k]}"
    return got


def count_boundary_ties(
    records: list[dict],
    kept: list[dict],
    rejected: list[dict],
    conf_key: str,
) -> tuple[int, dict]:
    """棄却境界における確信度値のタイ（同値衝突）件数を算出する。"""
    if not rejected or not kept:
        return 0, {"kept_at_boundary": 0, "rejected_at_boundary": 0, "equal_across_cut": False}
    b_kept = kept[-1][conf_key]
    b_rej = rejected[0][conf_key]
    if b_kept == b_rej:
        in_kept = sum(1 for r in kept if r[conf_key] == b_kept)
        in_rej = sum(1 for r in rejected if r[conf_key] == b_kept)
        return in_kept + in_rej, {
            "kept_at_boundary": in_kept,
            "rejected_at_boundary": in_rej,
            "equal_across_cut": True,
        }
    return 0, {"kept_at_boundary": 1, "rejected_at_boundary": 1, "equal_across_cut": False}


def compute_type_metrics(
    kept: list[dict],
    rejected: list[dict],
    all_covered_idx: list[int],
    all_styles: list[str],
) -> dict:
    """残存分・棄却分における種別分類の精度・Macro F1・文体別残存内訳を計算する。"""
    total = len(kept) + len(rejected)
    if kept:
        yt = np.array([r["gold_type_idx"] for r in kept])
        pt = np.array([r["pred_type_idx"] for r in kept])
        kept_true_idx = sorted(set(yt.tolist()))
        appeared_idx = sorted(set(yt.tolist()) | set(pt.tolist()))
        acc = round(float(accuracy_score(yt, pt)), 4)
        f1_kept_true = round(float(f1_score(yt, pt, labels=kept_true_idx, average="macro", zero_division=0)), 4)
        f1_all_cov = round(float(f1_score(yt, pt, labels=all_covered_idx, average="macro", zero_division=0)), 4)
        f1_appeared = round(float(f1_score(yt, pt, labels=appeared_idx, average="macro", zero_division=0)), 4)
        kept_true_cnt = len(kept_true_idx)
    else:
        acc = f1_kept_true = f1_all_cov = f1_appeared = None
        kept_true_cnt = 0

    if rejected:
        rej_err = sum(1 for r in rejected if r["pred_type_idx"] != r["gold_type_idx"])
        rej_err_rate = round(rej_err / len(rejected), 4)
    else:
        rej_err = 0
        rej_err_rate = None

    by_style = {}
    all_recs = kept + rejected
    for st in all_styles:
        st_total = sum(1 for r in all_recs if r["style"] == st)
        st_kept = [r for r in kept if r["style"] == st]
        st_acc = (
            round(sum(1 for r in st_kept if r["pred_type_idx"] == r["gold_type_idx"]) / len(st_kept), 4)
            if st_kept
            else None
        )
        by_style[st] = {
            "total_rows": st_total,
            "kept_rows": len(st_kept),
            "kept_rate": round(len(st_kept) / st_total, 4) if st_total else None,
            "type_accuracy": st_acc,
        }

    return {
        "kept_rows": len(kept),
        "rejected_rows": len(rejected),
        "coverage": round(len(kept) / total, 4),
        "type_accuracy": acc,
        "type_macro_f1_kept_true": f1_kept_true,
        "kept_true_types": kept_true_cnt,
        "total_covered_types": len(all_covered_idx),
        "type_macro_f1_all_covered": f1_all_cov,
        "type_macro_f1_appeared": f1_appeared,
        "rejected_errors": rej_err,
        "rejected_error_rate": rej_err_rate,
        "by_style": by_style,
    }


def compute_domain_metrics(kept: list[dict], rejected: list[dict]) -> dict:
    """残存分・棄却分における領域分類の精度・Macro F1 を計算する。"""
    total = len(kept) + len(rejected)
    all8_idx = list(range(len(DOMAINS)))
    if kept:
        yd = np.array([r["gold_domain_idx"] for r in kept])
        pd = np.array([r["pred_domain_idx"] for r in kept])
        kept_true_idx = sorted(set(yd.tolist()))
        acc = round(float(accuracy_score(yd, pd)), 4)
        f1_kept_true = round(float(f1_score(yd, pd, labels=kept_true_idx, average="macro", zero_division=0)), 4)
        f1_all8 = round(float(f1_score(yd, pd, labels=all8_idx, average="macro", zero_division=0)), 4)
        kept_true_cnt = len(kept_true_idx)
    else:
        acc = f1_kept_true = f1_all8 = None
        kept_true_cnt = 0

    if rejected:
        rej_err = sum(1 for r in rejected if r["pred_domain_idx"] != r["gold_domain_idx"])
        rej_err_rate = round(rej_err / len(rejected), 4)
    else:
        rej_err = 0
        rej_err_rate = None

    return {
        "kept_rows": len(kept),
        "rejected_rows": len(rejected),
        "coverage": round(len(kept) / total, 4),
        "domain_accuracy": acc,
        "domain_macro_f1": f1_kept_true,
        "domain_macro_f1_all8": f1_all8,
        "kept_true_domains": kept_true_cnt,
        "total_domains": len(DOMAINS),
        "rejected_errors": rej_err,
        "rejected_error_rate": rej_err_rate,
    }


def analyze_coverage_vs_accuracy(records: list[dict]) -> dict:
    """種別・領域それぞれについて、棄却率別曲線と固定確信度閾値テーブルを算出する。"""
    all_covered_idx = sorted({r["gold_type_idx"] for r in records})
    all_styles = sorted({r["style"] for r in records})
    n = len(records)

    # 種別: conf_type 降順（飽和時は log1m_conf_type 昇順で厳密順位付け）
    sorted_type = sorted(
        records,
        key=lambda r: (r["conf_type"], -r["log1m_conf_type"], -r["row_idx"]),
        reverse=True,
    )
    type_by_reject_rate = []
    for rr in REJECT_RATES:
        kept_n = round(n * (1.0 - rr))
        kept = sorted_type[:kept_n]
        rejected = sorted_type[kept_n:]
        ties, tie_detail = count_boundary_ties(sorted_type, kept, rejected, "conf_type")
        row = {
            "reject_rate": rr,
            "boundary_conf_kept_min": round(kept[-1]["conf_type"], 6) if kept else None,
            "boundary_conf_rej_max": round(rejected[0]["conf_type"], 6) if rejected else None,
            "boundary_conf_kept_min_raw": kept[-1]["conf_type"] if kept else None,
            "boundary_conf_rej_max_raw": rejected[0]["conf_type"] if rejected else None,
            "ties_at_boundary": ties,
            "ties_at_boundary_detail": tie_detail,
        }
        row.update(compute_type_metrics(kept, rejected, all_covered_idx, all_styles))
        type_by_reject_rate.append(row)

    type_by_fixed_tau = []
    for tau in FIXED_TAUS:
        kept = [r for r in sorted_type if r["conf_type"] >= tau]
        rejected = [r for r in sorted_type if r["conf_type"] < tau]
        row = {
            "tau": tau,
            "reject_rate": round(len(rejected) / n, 4),
        }
        row.update(compute_type_metrics(kept, rejected, all_covered_idx, all_styles))
        type_by_fixed_tau.append(row)

    # 領域: conf_domain 降順（飽和時は log1m_conf_domain 昇順で厳密順位付け）
    sorted_domain = sorted(
        records,
        key=lambda r: (r["conf_domain"], -r["log1m_conf_domain"], -r["row_idx"]),
        reverse=True,
    )
    domain_by_reject_rate = []
    for rr in REJECT_RATES:
        kept_n = round(n * (1.0 - rr))
        kept = sorted_domain[:kept_n]
        rejected = sorted_domain[kept_n:]
        ties, tie_detail = count_boundary_ties(sorted_domain, kept, rejected, "conf_domain")
        row = {
            "reject_rate": rr,
            "boundary_conf_kept_min": round(kept[-1]["conf_domain"], 6) if kept else None,
            "boundary_conf_rej_max": round(rejected[0]["conf_domain"], 6) if rejected else None,
            "boundary_conf_kept_min_raw": kept[-1]["conf_domain"] if kept else None,
            "boundary_conf_rej_max_raw": rejected[0]["conf_domain"] if rejected else None,
            "ties_at_boundary": ties,
            "ties_at_boundary_detail": tie_detail,
        }
        row.update(compute_domain_metrics(kept, rejected))
        domain_by_reject_rate.append(row)

    domain_by_fixed_tau = []
    for tau in FIXED_TAUS:
        kept = [r for r in sorted_domain if r["conf_domain"] >= tau]
        rejected = [r for r in sorted_domain if r["conf_domain"] < tau]
        row = {
            "tau": tau,
            "reject_rate": round(len(rejected) / n, 4),
        }
        row.update(compute_domain_metrics(kept, rejected))
        domain_by_fixed_tau.append(row)

    # 精度非単調チェックではなく、境界確信度の単調性と件数整合をアサート
    assert [x["kept_rows"] for x in type_by_reject_rate] == [200, 180, 160, 140, 120]
    assert [x["kept_rows"] for x in domain_by_reject_rate] == [200, 180, 160, 140, 120]
    for i in range(len(type_by_reject_rate) - 1):
        assert type_by_reject_rate[i]["boundary_conf_kept_min_raw"] <= type_by_reject_rate[i + 1]["boundary_conf_kept_min_raw"]
        assert domain_by_reject_rate[i]["boundary_conf_kept_min_raw"] <= domain_by_reject_rate[i + 1]["boundary_conf_kept_min_raw"]

    return {
        "n_rows": n,
        "covered_types": len(all_covered_idx),
        "type": {
            "by_reject_rate": type_by_reject_rate,
            "by_fixed_tau": type_by_fixed_tau,
        },
        "domain": {
            "by_reject_rate": domain_by_reject_rate,
            "by_fixed_tau": domain_by_fixed_tau,
        },
    }


def eval_binary_flags(gold: np.ndarray, pred: np.ndarray) -> dict:
    """2値高機密フラグの混同行列・Recall・標準文書誤フラグ率（FPR）等を計算する。"""
    tp = int(((gold == 1) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    tn = int(((gold == 0) & (pred == 0)).sum())
    hr_total = tp + fn
    std_total = fp + tn
    n = len(gold)
    recall = round(tp / hr_total, 4) if hr_total else None
    std_fpr = round(fp / std_total, 4) if std_total else None
    precision = round(float(precision_score(gold, pred, zero_division=0)), 4)
    f1 = round(float(f1_score(gold, pred, zero_division=0)), 4)
    flag_rate = round((tp + fp) / n, 4) if n else None
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "hr_total": hr_total,
        "std_total": std_total,
        "recall": recall,
        "std_fpr": std_fpr,
        "precision": precision,
        "f1": f1,
        "flag_rate": flag_rate,
    }


def analyze_high_risk(records: list[dict]) -> dict:
    """高機密フラグの閾値調整曲線・目標 Recall 到達点・棄却連動フォールバック・FN 内訳を算出する。"""
    gold = np.array([r["gold_high_risk"] for r in records])
    argmax_pred = np.array([r["pred_high_risk_argmax"] for r in records])
    scores_sum = np.array([r["p_high_risk_sum"] for r in records], dtype=np.float64)
    scores_max = np.array([r["p_high_risk_max"] for r in records], dtype=np.float64)

    baseline_argmax = eval_binary_flags(gold, argmax_pred)

    # 主曲線: p_high_risk_sum >= t
    sum_curve = []
    for t in HR_THRESHOLDS:
        m = eval_binary_flags(gold, (scores_sum >= t).astype(int))
        m["threshold"] = t
        sum_curve.append(m)

    # 閾値低下に対して Recall・std_fpr が非減少であることをアサート
    for i in range(len(sum_curve) - 1):
        assert sum_curve[i]["recall"] <= sum_curve[i + 1]["recall"]
        assert sum_curve[i]["std_fpr"] <= sum_curve[i + 1]["std_fpr"]

    # 補助曲線: p_high_risk_max >= t
    max_curve = []
    for t in HR_THRESHOLDS:
        m = eval_binary_flags(gold, (scores_max >= t).astype(int))
        m["threshold"] = t
        max_curve.append(m)

    # 目標 Recall 到達点（高機密スコア降順 k 番目の厳密閾値）
    hr_scores_desc = sorted(scores_sum[gold == 1].tolist(), reverse=True)
    target_points = []
    for target in TARGET_RECALLS:
        k = math.ceil(target * len(hr_scores_desc) - 1e-12)
        t_exact = hr_scores_desc[k - 1]
        m = eval_binary_flags(gold, (scores_sum >= t_exact).astype(int))
        m["target_recall"] = target
        m["max_threshold_exact"] = t_exact
        m["max_threshold_rounded"] = round(t_exact, 6)
        target_points.append(m)

    # AUC
    roc_auc_sum = round(float(roc_auc_score(gold, scores_sum)), 4)
    pr_auc_sum = round(float(average_precision_score(gold, scores_sum)), 4)
    roc_auc_max = round(float(roc_auc_score(gold, scores_max)), 4)
    pr_auc_max = round(float(average_precision_score(gold, scores_max)), 4)

    # 棄却連動安全フォールバック:
    # ルールA: (argmax in HIGH_RISK) OR (種別棄却された文書)
    # ルールB: (p_high_risk_sum >= 0.50) OR (種別棄却された文書)
    sorted_type = sorted(
        records,
        key=lambda r: (r["conf_type"], -r["log1m_conf_type"], -r["row_idx"]),
        reverse=True,
    )
    n = len(records)
    rejection_fallback = []
    for rr in REJECT_RATES:
        kept_n = round(n * (1.0 - rr))
        kept = sorted_type[:kept_n]
        rejected = sorted_type[kept_n:]
        rej_ids = {r["id"] for r in rejected}
        tau_reject = kept[-1]["conf_type"] if kept else 1.0
        is_rej_rank = np.array([r["id"] in rej_ids for r in records])
        is_rej_tau = np.array([r["conf_type"] < tau_reject for r in records])

        rule_argmax_rank = eval_binary_flags(gold, np.logical_or(argmax_pred == 1, is_rej_rank).astype(int))
        rule_sum05_rank = eval_binary_flags(gold, np.logical_or(scores_sum >= 0.50, is_rej_rank).astype(int))
        rule_argmax_tau = eval_binary_flags(gold, np.logical_or(argmax_pred == 1, is_rej_tau).astype(int))

        rejection_fallback.append({
            "type_reject_rate": rr,
            "rejected_rows": len(rejected),
            "tau_reject_boundary": round(tau_reject, 6),
            "argmax_or_rejected_by_rank": rule_argmax_rank,
            "sum05_or_rejected_by_rank": rule_sum05_rank,
            "argmax_or_below_tau": rule_argmax_tau,
        })

    # 見逃し詳細（argmax FN および p_high_risk_sum < 0.50 の全高機密文書）
    fn_cases = [
        {
            "id": r["id"],
            "gold_type": r["gold_type"],
            "style": r["style"],
            "gold_domain": r["gold_domain"],
            "pred_type": r["pred_type"],
            "pred_domain": r["pred_domain"],
            "conf_type": round(r["conf_type"], 6),
            "p_high_risk_sum": r["p_high_risk_sum"],
            "p_high_risk_max": r["p_high_risk_max"],
            "argmax_in_hr": bool(r["pred_high_risk_argmax"]),
            "top3_types": r["top3_types"],
        }
        for r in records
        if r["gold_high_risk"] == 1 and (r["pred_high_risk_argmax"] == 0 or r["p_high_risk_sum"] < 0.50)
    ]
    fn_cases.sort(key=lambda x: x["p_high_risk_sum"], reverse=True)

    # FN 種別・文体別集計（argmax 基準）
    argmax_fn_by_type = dict(Counter(c["gold_type"] for c in fn_cases if not c["argmax_in_hr"]))
    argmax_fn_by_style = dict(Counter(c["style"] for c in fn_cases if not c["argmax_in_hr"]))

    return {
        "hr_rows": int((gold == 1).sum()),
        "std_rows": int((gold == 0).sum()),
        "baseline_argmax": baseline_argmax,
        "roc_auc_sum": roc_auc_sum,
        "pr_auc_sum": pr_auc_sum,
        "roc_auc_max": roc_auc_max,
        "pr_auc_max": pr_auc_max,
        "p_high_risk_sum_curve": sum_curve,
        "p_high_risk_max_curve": max_curve,
        "target_recall_operating_points": target_points,
        "rejection_coupled_fallback": rejection_fallback,
        "fn_details_argmax_or_sum_lt_05": fn_cases,
        "argmax_fn_by_type": argmax_fn_by_type,
        "argmax_fn_by_style": argmax_fn_by_style,
    }


def main() -> int:
    root = Path(".")  # project root (run from the model folder)
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sbintuitions/modernbert-ja-130m")
    ap.add_argument("--state", default=str(root / "runs" / "prod-modernbert-v2" / "model_state.pt"))
    ap.add_argument("--tokenizer-dir", default=str(root / "runs" / "prod-modernbert-v2" / "final"))
    ap.add_argument("--dev-jsonl", default=str(root / "data" / "pilot" / "dev.jsonl"))
    ap.add_argument("--eval-dev", default=str(root / "runs" / "prod-modernbert-v2" / "eval_dev.json"))
    ap.add_argument("--ood-v1", default=str(root / "data" / "ood" / "ood-200.jsonl"))
    ap.add_argument("--ood-v2", default=str(root / "data" / "ood_v2" / "ood-v2-200.jsonl"))
    ap.add_argument("--ref-ood-v1", default=str(root / "runs" / "prod-modernbert-v2" / "eval_ood_v1.json"))
    ap.add_argument("--ref-ood-v2", default=str(root / "runs" / "prod-modernbert-v2" / "eval_ood_v2.json"))
    ap.add_argument("--out-dir", default=str(root / "runs" / "prod-modernbert-v2"))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    # 1. 43種別ラベル順序を eval_dev.json から取得し、dev.jsonl と完全一致・43種であることを検証（sealed-test は開かない）
    eval_dev = json.loads(Path(args.eval_dev).read_text(encoding="utf-8"))
    types = sorted(eval_dev["per_type"].keys())
    dev_rows = load_jsonl(Path(args.dev_jsonl))
    dev_types = sorted({r["document_type"] for r in dev_rows})
    assert len(types) == 43, f"Expected 43 types, got {len(types)}"
    assert types == dev_types, "Label order mismatch between eval_dev.json and dev.jsonl"
    t2i = {t: i for i, t in enumerate(types)}
    hr_idx = [t2i[t] for t in sorted(HIGH_RISK) if t in t2i]
    assert len(hr_idx) == 22, f"Expected 22 HIGH_RISK types, got {len(hr_idx)}"

    # 2. モデルとトークナイザをロード
    device, device_note = select_device(args.device)
    print(f"[INFO] device={device_note}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model = MultiHead(args.base, len(types))
    max_len = min(1024, getattr(model.encoder.config, "max_position_embeddings", 1024))
    assert max_len == 1024, f"Unexpected max_len={max_len}"
    state_dict = torch.load(args.state, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # 3. OOD v1 / OOD v2 推論
    rows_v1 = load_jsonl(Path(args.ood_v1))
    rows_v2 = load_jsonl(Path(args.ood_v2))
    sec_v1, rec_v1 = run_inference(model, tok, rows_v1, types, hr_idx, device, max_len)
    sec_v2, rec_v2 = run_inference(model, tok, rows_v2, types, hr_idx, device, max_len)
    print(f"[INFO] inference finished: ood_v1={sec_v1:.2f}s ({len(rec_v1)} docs), ood_v2={sec_v2:.2f}s ({len(rec_v2)} docs)")

    # 4. パリティ検証（0% 棄却時と既存 Colab L4 評価の一致を確認）
    parity_v1 = verify_parity("OOD v1", rec_v1, Path(args.ref_ood_v1))
    parity_v2 = verify_parity("OOD v2", rec_v2, Path(args.ref_ood_v2))
    print("[OK] Parity assertions passed for OOD v1 and OOD v2.")

    # 5. 保存 & 分析
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_probs_jsonl(out_dir / "ood_v1_probs.jsonl", rec_v1)
    save_probs_jsonl(out_dir / "ood_v2_probs.jsonl", rec_v2)

    analysis = {
        "metadata": {
            "model_base": args.base,
            "model_state": str(args.state),
            "model_state_bytes": Path(args.state).stat().st_size,
            "tokenizer_dir": str(args.tokenizer_dir),
            "device_used": device_note,
            "max_len": max_len,
            "n_types": len(types),
            "n_domains": len(DOMAINS),
            "n_high_risk_types": len(hr_idx),
            "inference_sec": {"ood_v1": round(sec_v1, 2), "ood_v2": round(sec_v2, 2)},
            "parity_verified": {"ood_v1": parity_v1, "ood_v2": parity_v2},
            "sealed_test_opened": False,
            "retrained": False,
        },
        "coverage_vs_accuracy": {
            "ood_v1": analyze_coverage_vs_accuracy(rec_v1),
            "ood_v2": analyze_coverage_vs_accuracy(rec_v2),
        },
        "high_risk_threshold_analysis": {
            "ood_v2": analyze_high_risk(rec_v2),
            "ood_v1": analyze_high_risk(rec_v1),
        },
    }

    out_json = out_dir / "routing_analysis.json"
    out_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved analysis to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
