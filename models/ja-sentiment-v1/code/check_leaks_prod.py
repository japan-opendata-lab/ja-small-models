#!/usr/bin/env python3
"""第12号 ja-sentiment-v1 本番4分割全6ペア 高速リーク検査スクリプト。

対象: train (100,000), dev (5,000), sealed-test (5,000), human-eval (600)
※ sealed-test.jsonl はテキスト本文を一切出力せず、数値指標のみを出力する。
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_lsh as fl  # noqa: E402
import sentiment_rules as sr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(p: Path) -> List[dict]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def inspect_pair_fast(
    rows_a: List[dict],
    rows_b: List[dict],
    idx_a_80: fl.LSHIndex,
) -> dict:
    sub_a = {f"{r['domain']}::{r['subtopic']}" for r in rows_a}
    sub_b = {f"{r['domain']}::{r['subtopic']}" for r in rows_b}
    ex_a = {r["text"] for r in rows_a}
    ex_b = {r["text"] for r in rows_b}
    norm_a = {sr.normalize_for_dedup(r["text"]) for r in rows_a}
    norm_b = {sr.normalize_for_dedup(r["text"]) for r in rows_b}
    texts_b = [r["text"] for r in rows_b]

    return {
        "subtopic_overlap": len(sub_a & sub_b),
        "exact_text_overlap": len(ex_a & ex_b),
        "exact_norm_text_overlap": len(norm_a & norm_b),
        "near_dup_lsh_080": idx_a_80.count_cross_overlap_pairs(texts_b, jaccard_thresh=0.80),
        "near_dup_lsh_070": idx_a_80.count_cross_overlap_pairs(texts_b, jaccard_thresh=0.70),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data" / "prod" / "train.jsonl"))
    ap.add_argument("--dev", default=str(ROOT / "data" / "prod" / "dev.jsonl"))
    ap.add_argument("--sealed", default=str(ROOT / "data" / "prod" / "sealed-test.jsonl"))
    ap.add_argument("--human-eval", default=str(ROOT / "data" / "eval" / "human-eval.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "prod" / "leak-report.json"))
    args = ap.parse_args()

    t0 = time.time()
    splits: Dict[str, List[dict]] = {
        "train": load_jsonl(Path(args.train)),
        "dev": load_jsonl(Path(args.dev)),
        "sealed-test": load_jsonl(Path(args.sealed)),
        "human-eval": load_jsonl(Path(args.human_eval)),
    }
    names = ["train", "dev", "sealed-test", "human-eval"]

    # 小さい方をLSHインデックス側にして全ペア高速検査（対称比較）
    lsh_indices: Dict[str, fl.LSHIndex] = {}
    for nm in ("dev", "sealed-test", "human-eval"):
        idx = fl.LSHIndex(ngram_n=5, num_perm=64, bands=16, seed=42)
        idx.add_texts([r["text"] for r in splits[nm]])
        lsh_indices[nm] = idx

    pairs = {}
    all_zero = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            # lsh_indices に入っている小さい方をインデックス側に渡す
            if b in lsh_indices and (a not in lsh_indices or len(splits[b]) <= len(splits[a])):
                res = inspect_pair_fast(splits[b], splits[a], lsh_indices[b])
            else:
                res = inspect_pair_fast(splits[a], splits[b], lsh_indices[a])
            pairs[f"{a}__{b}"] = res
            if any(v != 0 for v in res.values()):
                all_zero = False

    report = {
        "elapsed_sec": round(time.time() - t0, 2),
        "counts": {k: len(v) for k, v in splits.items()},
        "all_zero": all_zero,
        "pairs": pairs,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all_zero:
        sys.exit(1)


if __name__ == "__main__":
    main()
