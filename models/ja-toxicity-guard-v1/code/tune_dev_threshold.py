#!/usr/bin/env python3
"""第8号 ja-toxicity-guard-v1 dev のみでのラベル別閾値検討・固定スクリプト。

原則 0.50 を基準とし、過学習を防ぐ明確な改善条件を満たす場合のみ dev 上で
ラベル別閾値を採用・固定する。封印テストは一切参照しない。
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toxicity_prod_templates as pt  # noqa: E402
import train_prod_tox as tpt  # noqa: E402

CANDIDATES: List[float] = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    for p_str in (args.dev, args.preds):
        if "sealed" in Path(p_str).name.lower() or "sealed" in str(p_str).lower():
            raise RuntimeError(f"封印テストファイルは入力禁止です: {p_str}")

    dev_rows = [json.loads(l) for l in open(args.dev, encoding="utf-8")]
    pred_rows = [json.loads(l) for l in open(args.preds, encoding="utf-8")]
    assert len(dev_rows) == len(pred_rows)
    for dr, pr in zip(dev_rows, pred_rows):
        assert dr["id"] == pr["id"]

    probs = np.array([[float(pr["probs"][lb]) for lb in pt.LABELS] for pr in pred_rows], dtype=float)
    y_true = np.array([[int(r["labels"][lb]) for lb in pt.LABELS] for r in dev_rows], dtype=int)
    benign_mask = y_true.sum(axis=1) == 0
    crit_news_mask = np.array([r.get("benign_subtype") in ("criticism", "news_quote") for r in dev_rows])

    sweep: Dict[str, List[dict]] = {}
    adopted: Dict[str, float] = {}
    rationale: Dict[str, dict] = {}

    for j, lb in enumerate(pt.LABELS):
        yt = y_true[:, j]
        p_col = probs[:, j]
        rows_j: List[dict] = []
        base_row = None

        for thr in CANDIDATES:
            yp = (p_col >= thr).astype(int)
            tp = int(((yp == 1) & (yt == 1)).sum())
            fp = int(((yp == 1) & (yt == 0)).sum())
            fn = int(((yp == 0) & (yt == 1)).sum())
            m = tpt._prf(tp, fp, fn)
            m["threshold"] = thr
            m["crit_news_fp"] = int(yp[crit_news_mask].sum())
            m["all_benign_fp"] = int(yp[benign_mask].sum())
            rows_j.append(m)
            if abs(thr - 0.50) < 1e-9:
                base_row = m

        assert base_row is not None
        sweep[lb] = rows_j

        eligible = []
        for m in rows_j:
            if abs(m["threshold"] - 0.50) < 1e-9:
                continue
            cond_a = (
                m["crit_news_fp"] <= base_row["crit_news_fp"]
                and m["all_benign_fp"] <= base_row["all_benign_fp"]
                and m["f1"] >= base_row["f1"] + 0.0050
            )
            cond_b = (
                (m["crit_news_fp"] <= base_row["crit_news_fp"] - 3 or m["all_benign_fp"] <= base_row["all_benign_fp"] - 3)
                and m["crit_news_fp"] <= base_row["crit_news_fp"]
                and m["f1"] >= base_row["f1"] - 0.0030
            )
            if cond_a or cond_b:
                score = (m["f1"] - base_row["f1"]) + 0.001 * (base_row["all_benign_fp"] - m["all_benign_fp"])
                eligible.append((score, -abs(m["threshold"] - 0.50), m, cond_a, cond_b))

        if eligible:
            eligible.sort(key=lambda x: (x[0], x[1]), reverse=True)
            _, _, best_m, ca, cb = eligible[0]
            adopted[lb] = best_m["threshold"]
            rationale[lb] = {
                "adopted": best_m["threshold"],
                "changed_from_default": True,
                "reason": ("cond_a_f1_gain" if ca else "") + ("+" if ca and cb else "") + ("cond_b_fp_cut" if cb else ""),
                "at_0.5": base_row,
                "at_adopted": best_m,
            }
        else:
            adopted[lb] = 0.50
            rationale[lb] = {
                "adopted": 0.50,
                "changed_from_default": False,
                "reason": "keep_default_0.50",
                "at_0.5": base_row,
                "at_adopted": base_row,
            }

    default_thrs = {lb: 0.50 for lb in pt.LABELS}
    eval_default = tpt.evaluate_with_thresholds(dev_rows, probs, default_thrs)
    eval_adopted = tpt.evaluate_with_thresholds(dev_rows, probs, adopted)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adopted_thresholds.json").write_text(
        json.dumps({"thresholds": adopted, "rationale": rationale}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "dev_threshold_tuning.json").write_text(
        json.dumps(
            {
                "candidates": CANDIDATES,
                "adopted_thresholds": adopted,
                "rationale": rationale,
                "eval_at_default_0.50": eval_default,
                "eval_at_adopted": eval_adopted,
                "sweep": sweep,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "adopted_thresholds": adopted,
                "default_macro_f1": eval_default["macro_f1"],
                "adopted_macro_f1": eval_adopted["macro_f1"],
                "default_exact_match": eval_default["exact_match_accuracy"],
                "adopted_exact_match": eval_adopted["exact_match_accuracy"],
                "default_crit_news_fpr": eval_default["benign_fpr"]["criticism_and_news_quote"]["fpr"],
                "adopted_crit_news_fpr": eval_adopted["benign_fpr"]["criticism_and_news_quote"]["fpr"],
                "default_all_benign_fpr": eval_default["benign_fpr"]["all_benign"]["fpr"],
                "adopted_all_benign_fpr": eval_adopted["benign_fpr"]["all_benign"]["fpr"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
