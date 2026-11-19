#!/usr/bin/env python3
"""第8号 ja-toxicity-guard-v1 パイロットデータ構築・リーク検査・安全監査スクリプト。

使い方:
  py code/build_pilot_data.py --out data/pilot --seed 42

出力:
  data/pilot/train.jsonl (20,000件)
  data/pilot/dev.jsonl (2,000件)
  data/pilot/sealed-test.jsonl (2,000件, 生成後未開封)
  data/pilot/build-stats.json
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toxicity_synth as ts  # noqa: E402
import toxicity_templates as tt  # noqa: E402


def _summarize_split(rows: List[dict]) -> dict:
    lengths = [len(r["text"]) for r in rows]
    label_counts = {lb: 0 for lb in tt.LABELS}
    num_labels_dist = Counter()
    source_kind_dist = Counter()
    benign_sub_dist = Counter()
    pattern_dist = Counter()
    style_dist = Counter()
    used_templates: Set[str] = set()

    for r in rows:
        cnt = 0
        for lb in tt.LABELS:
            v = int(r["labels"][lb])
            label_counts[lb] += v
            cnt += v
        num_labels_dist[str(cnt)] += 1
        source_kind_dist[r["source_kind"]] += 1
        if r["benign_subtype"] is not None:
            benign_sub_dist[r["benign_subtype"]] += 1
        pattern_dist[r["pattern"]] += 1
        style_dist[r["style"]] += 1
        used_templates.update(r["template_ids"])

    n = len(rows)
    return {
        "total_rows": n,
        "toxic_rows": sum(1 for r in rows if r["is_toxic"]),
        "benign_rows": sum(1 for r in rows if not r["is_toxic"]),
        "label_counts": label_counts,
        "label_rates": {k: round(v / max(1, n), 4) for k, v in label_counts.items()},
        "num_active_labels_dist": dict(sorted(num_labels_dist.items())),
        "source_kind_dist": dict(source_kind_dist),
        "benign_subtype_dist": dict(benign_sub_dist),
        "pattern_dist": dict(pattern_dist),
        "unique_templates_used": len(used_templates),
        "char_len": {
            "min": min(lengths),
            "median": round(float(statistics.median(lengths)), 1),
            "mean": round(float(statistics.mean(lengths)), 2),
            "max": max(lengths),
        },
        "top_styles": dict(style_dist.most_common(12)),
    }


def _pair_leak_check(rows_a: List[dict], rows_b: List[dict]) -> dict:
    tpl_a = {tid for r in rows_a for tid in r["template_ids"]}
    tpl_b = {tid for r in rows_b for tid in r["template_ids"]}
    tpl_ov = len(tpl_a & tpl_b)

    exact_a = {r["text"] for r in rows_a}
    exact_b = {r["text"] for r in rows_b}
    exact_ov = len(exact_a & exact_b)

    norm_a = {ts.normalize_text_for_dedup(r["text"]) for r in rows_a}
    norm_b = {ts.normalize_text_for_dedup(r["text"]) for r in rows_b}
    norm_ov = len(norm_a & norm_b)

    lsh_080 = ts.count_lsh_leaks(rows_a, rows_b, thr=0.80)
    lsh_070 = ts.count_lsh_leaks(rows_a, rows_b, thr=0.70)

    return {
        "template_id_overlap": tpl_ov,
        "exact_text_overlap": exact_ov,
        "exact_norm_text_overlap": norm_ov,
        "near_dup_lsh_080": lsh_080,
        "near_dup_lsh_070_ref": lsh_070,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pilot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--dev-n", type=int, default=2000)
    ap.add_argument("--sealed-n", type=int, default=2000)
    args = ap.parse_args()

    t0 = time.time()
    tt.validate_template_registry()

    lsh_index = ts.LSHIndex()
    seen_exact: Set[str] = set()
    seen_norm: Set[str] = set()

    splits_spec = [
        ("train", args.train_n, args.seed),
        ("dev", args.dev_n, args.seed + 101),
        ("sealed-test", args.sealed_n, args.seed + 202),
    ]
    splits_rows: Dict[str, List[dict]] = {}
    gen_meta: Dict[str, dict] = {}

    for split_name, target_n, sp_seed in splits_spec:
        rows, meta = ts.generate_split(
            split_name=split_name,
            total_n=target_n,
            seed=sp_seed,
            forbidden_lsh=lsh_index,
            forbidden_exact=seen_exact,
            forbidden_norm=seen_norm,
        )
        splits_rows[split_name] = rows
        gen_meta[split_name] = meta
        for r in rows:
            eh = hashlib.md5(r["text"].encode("utf-8")).hexdigest()
            nh = hashlib.md5(ts.normalize_text_for_dedup(r["text"]).encode("utf-8")).hexdigest()
            seen_exact.add(eh)
            seen_norm.add(nh)
            sh = ts.char_shingles(r["text"])
            sig = ts.compute_minhash(sh)
            lsh_index.add(sh, sig)

    # 分割間リーク検査（全3ペア）
    leaks = {
        "train__dev": _pair_leak_check(splits_rows["train"], splits_rows["dev"]),
        "train__sealed-test": _pair_leak_check(splits_rows["train"], splits_rows["sealed-test"]),
        "dev__sealed-test": _pair_leak_check(splits_rows["dev"], splits_rows["sealed-test"]),
    }
    for pair_name, res in leaks.items():
        assert res["template_id_overlap"] == 0, f"{pair_name} template_id_overlap={res['template_id_overlap']}"
        assert res["exact_text_overlap"] == 0, f"{pair_name} exact_text_overlap={res['exact_text_overlap']}"
        assert res["exact_norm_text_overlap"] == 0, f"{pair_name} exact_norm_text_overlap={res['exact_norm_text_overlap']}"
        assert res["near_dup_lsh_080"] == 0, f"{pair_name} near_dup_lsh_080={res['near_dup_lsh_080']}"

    # 安全・倫理自動監査（全3分割）
    safety = {}
    for split_name, rows in splits_rows.items():
        sres = ts.audit_safety(rows)
        safety[split_name] = sres
        assert sres["violation_count"] == 0, f"{split_name} safety violations: {sres}"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits_rows.items():
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "seed": args.seed,
        "labels": tt.LABELS,
        "benign_subtypes": tt.BENIGN_SUBTYPES,
        "elapsed_sec": round(time.time() - t0, 2),
        "splits": {name: _summarize_split(rows) for name, rows in splits_rows.items()},
        "generation_meta": gen_meta,
        "leaks": leaks,
        "safety_audit": safety,
    }
    (out_dir / "build-stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "elapsed_sec": stats["elapsed_sec"],
        "counts": {k: v["total_rows"] for k, v in stats["splits"].items()},
        "leaks": {
            k: {
                "tpl": v["template_id_overlap"],
                "exact": v["exact_text_overlap"],
                "norm": v["exact_norm_text_overlap"],
                "lsh080": v["near_dup_lsh_080"],
                "lsh070_ref": v["near_dup_lsh_070_ref"],
            }
            for k, v in leaks.items()
        },
        "safety_violations": {k: v["violation_count"] for k, v in safety.items()},
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
