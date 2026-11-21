#!/usr/bin/env python3
"""第12号 ja-sentiment-v1 パイロットデータ構築スクリプト。

data/synth/synth_verified.jsonl から規則再検証・重複除去（プール内 LSH>=0.75、プール間 LSH>=0.70）・
ドメイン/サブタイプ/長さ帯の層化均衡サンプリングを行い、
train (20,000) / dev (2,000) / sealed-test (2,000) を構築する。
※ sealed-test.jsonl のテキスト本文は一切出力しない。
"""
import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sentiment_rules as sr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

TARGET_COUNTS = {
    "train": {"negative": 6666, "neutral": 6667, "positive": 6667},
    "dev": {"negative": 666, "neutral": 667, "positive": 667},
    "sealed_test": {"negative": 666, "neutral": 667, "positive": 667},
}


def load_jsonl(p: Path) -> List[dict]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def dedup_within_pool(rows: List[dict], jaccard_thresh: float = 0.75) -> Tuple[List[dict], dict]:
    """単一プール内の完全・正規化重複および LSH 近傍重複を除去する。"""
    seen_norm: Set[str] = set()
    uniq_rows: List[dict] = []
    exact_drop = 0
    rule_drop = 0
    for r in rows:
        txt = sr.clean_surface(r["text"])
        lbl = r["intended_label"]
        ok, _ = sr.check_text_rules(txt, lbl, r.get("domain", ""), r.get("length_band", ""))
        if not ok:
            rule_drop += 1
            continue
        nt = sr.normalize_for_dedup(txt)
        if not nt or nt in seen_norm:
            exact_drop += 1
            continue
        seen_norm.add(nt)
        uniq_rows.append({**r, "text": txt})

    hits = sr.lsh_find_near_duplicates(
        [r["text"] for r in uniq_rows], ngram_n=5, jaccard_thresh=jaccard_thresh
    )
    drop_idx: Set[int] = set()
    for u, v, _ in hits:
        if u not in drop_idx:
            drop_idx.add(v)
    kept = [r for i, r in enumerate(uniq_rows) if i not in drop_idx]
    return kept, {"rule_drop": rule_drop, "exact_drop": exact_drop, "lsh_drop": len(drop_idx)}


def filter_against_existing(rows: List[dict], existing_rows: List[dict], jaccard_thresh: float = 0.70) -> Tuple[List[dict], int]:
    """他プールと重複（正規化一致または LSH>=0.70）する行を除去する。"""
    if not existing_rows:
        return rows, 0
    ex_norm = {sr.normalize_for_dedup(r["text"]) for r in existing_rows}
    step1 = [r for r in rows if sr.normalize_for_dedup(r["text"]) not in ex_norm]

    # 高速 LSH チェック（existing_rows をバケット化）
    ngram_n = 5
    num_perm = 64
    bands = 16
    band_rows = num_perm // bands
    ex_grams = [sr.char_ngrams(sr.normalize_for_dedup(r["text"]) or r["text"], n=ngram_n) for r in existing_rows]
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
    for idx, r in enumerate(existing_rows):
        sig = sr.compute_minhash_sig(r["text"], ngram_n=ngram_n, num_perm=num_perm, seed=42)
        for b in range(bands):
            buckets[(b, sig[b * band_rows : (b + 1) * band_rows])].append(idx)

    kept: List[dict] = []
    lsh_drop = 0
    for r in step1:
        t = r["text"]
        sg = sr.char_ngrams(sr.normalize_for_dedup(t) or t, n=ngram_n)
        sig = sr.compute_minhash_sig(t, ngram_n=ngram_n, num_perm=num_perm, seed=42)
        near = False
        if sg:
            cand_ids: Set[int] = set()
            for b in range(bands):
                cand_ids.update(buckets.get((b, sig[b * band_rows : (b + 1) * band_rows]), ()))
            for cid in cand_ids:
                eg = ex_grams[cid]
                if not eg:
                    continue
                inter = len(sg & eg)
                jac = inter / (len(sg) + len(eg) - inter)
                if jac >= jaccard_thresh:
                    near = True
                    break
        if near:
            lsh_drop += 1
        else:
            kept.append(r)
    return kept, (len(rows) - len(step1)) + lsh_drop


def stratified_sample_for_class(rows: List[dict], n_target: int, seed: int) -> List[dict]:
    """(domain, subtype, length_band) で層化し、極力均等に n_target 件を抽出する。"""
    if len(rows) < n_target:
        raise RuntimeError(f"Insufficient rows: need {n_target}, got {len(rows)}")
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["domain"], r.get("subtype", ""), r.get("length_band", ""))].append(r)
    keys = sorted(buckets.keys())
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(buckets[k])
        # 自然さスコア 5 を先頭寄りにする（ただし完全排除はせずバリエーション保持）
        buckets[k].sort(key=lambda x: int(x.get("verifier_naturalness") or 0), reverse=True)

    picked: List[dict] = []
    while len(picked) < n_target:
        progressed = False
        for k in keys:
            if buckets[k]:
                picked.append(buckets[k].pop(0))
                progressed = True
                if len(picked) >= n_target:
                    break
        if not progressed:
            break
    return picked


def format_record(r: dict, idx: int, prefix: str) -> dict:
    lbl = r["intended_label"]
    return {
        "id": f"{prefix}_{idx:05d}",
        "sample_id": r["sample_id"],
        "split": prefix,
        "domain": r["domain"],
        "subtopic": r["subtopic"],
        "aspect": r.get("aspect", ""),
        "style": r.get("style", ""),
        "length_band": r.get("length_band", ""),
        "subtype": r.get("subtype", ""),
        "difficulty": r.get("difficulty", "standard"),
        "label": lbl,
        "label_id": sr.LABEL2ID[lbl],
        "text": r["text"],
        "generator_model": r.get("generator_model", ""),
        "verifier_model": r.get("verifier_model", ""),
        "verifier_naturalness": r.get("verifier_naturalness", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "synth" / "synth_verified.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "pilot"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_verified = load_jsonl(Path(args.input))
    by_pool: Dict[str, List[dict]] = defaultdict(list)
    for r in raw_verified:
        if r.get("accepted"):
            by_pool[r["split_pool"]].append(r)

    dedup_stats = {}
    cleaned: Dict[str, List[dict]] = {}
    # dev, sealed_test, train の順でプール内重複除去・プール間重複除去を実施
    seen_across: List[dict] = []
    for pool in ("dev", "sealed_test", "train"):
        in_rows = by_pool[pool]
        u_rows, st_in = dedup_within_pool(in_rows, jaccard_thresh=0.75)
        u_rows2, cross_drop = filter_against_existing(u_rows, seen_across, jaccard_thresh=0.70)
        cleaned[pool] = u_rows2
        seen_across.extend(u_rows2)
        dedup_stats[pool] = {
            "raw_accepted": len(in_rows),
            **st_in,
            "cross_pool_drop": cross_drop,
            "available_after_dedup": len(u_rows2),
            "available_by_label": dict(Counter(x["intended_label"] for x in u_rows2)),
        }

    print("[DEDUP STATS]", json.dumps(dedup_stats, ensure_ascii=False, indent=2))

    # 各分割の層化抽出
    out_files = {
        "train": out_dir / "train.jsonl",
        "dev": out_dir / "dev.jsonl",
        "sealed_test": out_dir / "sealed-test.jsonl",
    }
    split_stats = {}
    rng = random.Random(args.seed)

    for pool, out_path in out_files.items():
        targets = TARGET_COUNTS[pool]
        pool_rows = cleaned[pool]
        selected: List[dict] = []
        for lbl, n_t in targets.items():
            cls_rows = [r for r in pool_rows if r["intended_label"] == lbl]
            picked = stratified_sample_for_class(cls_rows, n_t, seed=args.seed + sr.LABEL2ID[lbl])
            selected.extend(picked)
        rng.shuffle(selected)

        prefix = "sealed" if pool == "sealed_test" else pool
        formatted = [format_record(r, i, prefix) for i, r in enumerate(selected)]
        out_path.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in formatted) + "\n",
            encoding="utf-8",
        )
        lens = [len(x["text"]) for x in formatted]
        raw_bytes = out_path.read_bytes()
        split_stats[pool] = {
            "file": out_path.name,
            "rows": len(formatted),
            "bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "by_label": dict(Counter(x["label"] for x in formatted)),
            "by_domain": dict(Counter(x["domain"] for x in formatted)),
            "by_difficulty": dict(Counter(x["difficulty"] for x in formatted)),
            "by_subtype": dict(Counter(x["subtype"] for x in formatted)),
            "by_length_band": dict(Counter(x["length_band"] for x in formatted)),
            "unique_subtopics": len({x["subtopic"] for x in formatted}),
            "text_chars": {
                "min": min(lens),
                "median": statistics.median(lens),
                "mean": round(statistics.mean(lens), 2),
                "max": max(lens),
            },
        }

    build_stats = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "build_sec": round(time.time() - t0, 2),
        "dedup_stats": dedup_stats,
        "splits": split_stats,
    }
    (out_dir / "build-stats.json").write_text(
        json.dumps(build_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[BUILD DONE] elapsed={build_stats['build_sec']}s")
    for pool, st in split_stats.items():
        print(f"  {pool:12s}: rows={st['rows']}, by_label={st['by_label']}, sha256={st['sha256'][:12]}...")


if __name__ == "__main__":
    main()
