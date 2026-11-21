#!/usr/bin/env python
"""第11号 ja-nli-v1 4分割全6ペア リーク検査スクリプト。

対象: train (20,000), dev (2,000), sealed-test (2,000), human-eval (600)
※ sealed-test.jsonl は文書ID/ハッシュ/MinHash の数値比較のみ行い、中身テキストは一切出力しない。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nli_rules as nr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(p: Path) -> List[dict]:
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def build_lsh_cache(texts: Sequence[str], sig_cache: Dict[str, Tuple[Tuple[int, ...], Set[str]]], ngram_n: int = 5, num_perm: int = 64, bands: int = 16) -> dict:
    """テキスト集合のシグネチャ・5-gram・LSHバケットを1回だけ計算して保持する。"""
    rows_per_band = num_perm // bands
    sigs: List[Tuple[int, ...]] = []
    grams_list: List[Set[str]] = []
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
    for idx, t in enumerate(texts):
        if t not in sig_cache:
            norm_t = nr.normalize_for_dedup(t) or t
            sig_cache[t] = (nr.compute_minhash_sig(t, ngram_n=ngram_n, num_perm=num_perm), nr.char_ngrams(norm_t, n=ngram_n))
        sig, gr = sig_cache[t]
        sigs.append(sig)
        grams_list.append(gr)
        for b in range(bands):
            st = b * rows_per_band
            buckets[(b, sig[st : st + rows_per_band])].append(idx)
    return {"sigs": sigs, "grams": grams_list, "buckets": buckets, "bands": bands, "rows_per_band": rows_per_band}


def count_lsh_leaks_cached(cache_a: dict, cache_b: dict, jaccard_thresh: float) -> int:
    """キャッシュ済み2分割間の LSH 近傍重複ペア数を高速に数える。"""
    buckets_a = cache_a["buckets"]
    grams_a = cache_a["grams"]
    bands = cache_b["bands"]
    rpb = cache_b["rows_per_band"]

    seen_pairs: Set[Tuple[int, int]] = set()
    for idx_b, (sig_b, sb) in enumerate(zip(cache_b["sigs"], cache_b["grams"])):
        if not sb:
            continue
        for b in range(bands):
            st = b * rpb
            for idx_a in buckets_a.get((b, sig_b[st : st + rpb]), ()):
                pair = (idx_a, idx_b)
                if pair in seen_pairs:
                    continue
                sa = grams_a[idx_a]
                if not sa:
                    continue
                inter = len(sa & sb)
                jac = inter / (len(sa) + len(sb) - inter)
                if jac >= jaccard_thresh:
                    seen_pairs.add(pair)
    return len(seen_pairs)


def prepare_split_profile(rows: List[dict], sig_cache: Dict[str, Tuple[Tuple[int, ...], Set[str]]]) -> dict:
    docs = {r["doc_id"] for r in rows}
    cls = {r["cluster_id"] for r in rows}
    ep = {r["premise"] for r in rows}
    np_set = {nr.normalize_for_dedup(r["premise"]) for r in rows}
    nh_set = {nr.normalize_for_dedup(r["hypothesis"]) for r in rows}
    prem_list = sorted(ep)
    hyp_list = [r["hypothesis"] for r in rows]
    return {
        "docs": docs,
        "clusters": cls,
        "exact_prems": ep,
        "norm_prems": np_set,
        "norm_hyps": nh_set,
        "prem_lsh": build_lsh_cache(prem_list, sig_cache, ngram_n=5),
        "hyp_lsh": build_lsh_cache(hyp_list, sig_cache, ngram_n=5),
    }


def inspect_pair(prof_a: dict, prof_b: dict) -> dict:
    return {
        "doc_id_overlap": len(prof_a["docs"] & prof_b["docs"]),
        "cluster_id_overlap": len(prof_a["clusters"] & prof_b["clusters"]),
        "exact_premise_overlap": len(prof_a["exact_prems"] & prof_b["exact_prems"]),
        "exact_norm_premise_overlap": len(prof_a["norm_prems"] & prof_b["norm_prems"]),
        "exact_norm_hypothesis_overlap": len(prof_a["norm_hyps"] & prof_b["norm_hyps"]),
        "premise_lsh_080": count_lsh_leaks_cached(prof_a["prem_lsh"], prof_b["prem_lsh"], jaccard_thresh=0.80),
        "hypothesis_lsh_085": count_lsh_leaks_cached(prof_a["hyp_lsh"], prof_b["hyp_lsh"], jaccard_thresh=0.85),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data" / "pilot" / "train.jsonl"))
    ap.add_argument("--dev", default=str(ROOT / "data" / "pilot" / "dev.jsonl"))
    ap.add_argument("--sealed", default=str(ROOT / "data" / "pilot" / "sealed-test.jsonl"))
    ap.add_argument("--human-eval", default=str(ROOT / "data" / "eval" / "human-eval.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "pilot" / "leak-report.json"))
    args = ap.parse_args()

    splits = {
        "train": load_jsonl(Path(args.train)),
        "dev": load_jsonl(Path(args.dev)),
        "sealed-test": load_jsonl(Path(args.sealed)),
        "human-eval": load_jsonl(Path(args.human_eval)),
    }
    sig_cache: Dict[str, Tuple[Tuple[int, ...], Set[str]]] = {}
    profiles = {k: prepare_split_profile(v, sig_cache) for k, v in splits.items()}
    names = ["train", "dev", "sealed-test", "human-eval"]
    pairs = {}
    all_zero = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            res = inspect_pair(profiles[a], profiles[b])
            pairs[f"{a}__{b}"] = res
            if any(v != 0 for v in res.values()):
                all_zero = False

    report = {
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
