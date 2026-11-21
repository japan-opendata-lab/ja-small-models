#!/usr/bin/env python3
"""第12号 ja-sentiment-v1 無作為200件監査結果集計スクリプト（Wilson 95%信頼区間付き）。"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent


def wilson_ci(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def rate_obj(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {
        "pass": k,
        "total": n,
        "rate": round(k / n, 4) if n else 0.0,
        "ci95": [lo, hi],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(ROOT / "data" / "audit" / "audit_sample_200.jsonl"))
    ap.add_argument("--results", default=str(ROOT / "data" / "audit" / "audit_results_200.jsonl"))
    ap.add_argument("--out-json", default=str(ROOT / "data" / "audit" / "audit_summary.json"))
    ap.add_argument("--out-md", default=str(ROOT / "data" / "audit" / "audit_review_200.md"))
    args = ap.parse_args()

    samples = {
        r["audit_idx"]: r
        for r in (json.loads(ln) for ln in Path(args.sample).read_text(encoding="utf-8").splitlines() if ln.strip())
    }
    results = [
        json.loads(ln)
        for ln in Path(args.results).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(results) == len(samples), f"Expected {len(samples)} results, got {len(results)}"

    merged = []
    for res in sorted(results, key=lambda x: x["audit_idx"]):
        s = samples[res["audit_idx"]]
        lv = bool(res["label_valid"])
        np_ = bool(res["naturalness_pass"])
        bp = lv and np_
        merged.append({**s, **res, "both_pass": bp})

    n = len(merged)
    lv_cnt = sum(1 for r in merged if r["label_valid"])
    np_cnt = sum(1 for r in merged if r["naturalness_pass"])
    bp_cnt = sum(1 for r in merged if r["both_pass"])

    by_label = defaultdict(list)
    by_domain = defaultdict(list)
    by_diff = defaultdict(list)
    by_sub = defaultdict(list)
    fail_cats = Counter()
    failures = []

    for r in merged:
        by_label[r["label"]].append(r)
        by_domain[r["domain"]].append(r)
        by_diff[r.get("difficulty", "standard")].append(r)
        by_sub[r.get("subtype", "unknown")].append(r)
        if not r["both_pass"]:
            cat = r.get("failure_category") or "unspecified"
            fail_cats[cat] += 1
            failures.append(r)

    def grp_stats(grp_map):
        out = {}
        for k, items in sorted(grp_map.items()):
            out[k] = {
                "label_valid": rate_obj(sum(1 for x in items if x["label_valid"]), len(items)),
                "naturalness_pass": rate_obj(sum(1 for x in items if x["naturalness_pass"]), len(items)),
                "both_pass": rate_obj(sum(1 for x in items if x["both_pass"]), len(items)),
            }
        return out

    summary = {
        "n_audited": n,
        "overall": {
            "label_valid": rate_obj(lv_cnt, n),
            "naturalness_pass": rate_obj(np_cnt, n),
            "both_pass": rate_obj(bp_cnt, n),
        },
        "by_label": grp_stats(by_label),
        "by_domain": grp_stats(by_domain),
        "by_difficulty": grp_stats(by_diff),
        "by_subtype": grp_stats(by_sub),
        "failure_categories": dict(fail_cats),
        "failure_examples": [
            {
                "audit_idx": r["audit_idx"],
                "id": r["id"],
                "label": r["label"],
                "domain": r["domain"],
                "subtype": r.get("subtype", ""),
                "label_valid": r["label_valid"],
                "naturalness_pass": r["naturalness_pass"],
                "failure_category": r.get("failure_category", ""),
                "note": r.get("note", ""),
                "text": r["text"],
            }
            for r in failures
        ],
    }
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# 無作為抽出200件 監査レビュー台帳\n",
        f"- **ラベル妥当性合格率**: {lv_cnt}/{n} ({lv_cnt/n*100:.2f}%), 95%CI {summary['overall']['label_valid']['ci95']}",
        f"- **自然さ合格率**: {np_cnt}/{n} ({np_cnt/n*100:.2f}%), 95%CI {summary['overall']['naturalness_pass']['ci95']}",
        f"- **両方合格率**: {bp_cnt}/{n} ({bp_cnt/n*100:.2f}%), 95%CI {summary['overall']['both_pass']['ci95']}\n",
        "## 不合格事例一覧\n",
    ]
    for f in failures:
        md.append(
            f"- **[{f['audit_idx']:03d}] {f['id']}** (`{f['label']}`, `{f['domain']}`, `{f.get('subtype','')}`) "
            f"| label_valid={f['label_valid']}, nat_pass={f['naturalness_pass']} "
            f"| cat=`{f.get('failure_category','')}` | 理由: {f.get('note','')}\n"
            f"  - Text: {f['text']}"
        )
    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
    print(f"Audit summary written -> {args.out_json} | label_valid={lv_cnt}/{n}, nat={np_cnt}/{n}, both={bp_cnt}/{n}")


if __name__ == "__main__":
    main()
