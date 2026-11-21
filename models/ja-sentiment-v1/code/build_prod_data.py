#!/usr/bin/env python3
"""第12号 ja-sentiment-v1 本番データセット構築スクリプト。

パイロット検証済み (data/synth/synth_verified.jsonl) と本番追加検証済み
(data/prod_synth/synth_verified.jsonl) を合流し、human-eval 600件および
分割間の重複除去（プール内 LSH>=0.75、他分割・human-eval間 LSH>=0.70）を実施したうえで、
3クラス均等・ドメイン/難例/中立厚みを保った
train (100,000) / dev (5,000) / sealed-test (5,000) を構築する。
※ sealed-test.jsonl のテキスト本文は一切出力・表示しない。
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
import fast_lsh as fl  # noqa: E402
import sentiment_rules as sr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

TARGET_COUNTS = {
    "train": {"negative": 33333, "neutral": 33334, "positive": 33333},
    "dev": {"negative": 1666, "neutral": 1667, "positive": 1667},
    "sealed_test": {"negative": 1666, "neutral": 1667, "positive": 1667},
}

# クラス別の目標サブタイプ比率（中立 U1+U2>=64%、U1+U2+U4>=85% の厚みを確保）
SUBTYPE_WEIGHTS = {
    "negative": {
        "N1_direct_specific": 0.30,
        "N2_understated_complaint": 0.22,
        "N3_mixed_neg_dominant": 0.20,
        "N4_irony_sarcasm": 0.16,
        "N5_negated_positive": 0.12,
    },
    "neutral": {
        "U1_factual_report": 0.38,
        "U2_balanced_mixed": 0.26,
        "U4_emotion_word_factual": 0.21,
        "U3_question_inquiry": 0.15,
    },
    "positive": {
        "P1_direct_specific": 0.35,
        "P2_litotes_negation": 0.20,
        "P3_mixed_pos_dominant": 0.20,
        "P4_understated_pos": 0.15,
        "P5_relief_turnaround": 0.10,
    },
}


def load_jsonl(p: Path) -> List[dict]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def dedup_within_pool_fast(rows: List[dict], jaccard_thresh: float = 0.75) -> Tuple[List[dict], dict]:
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

    hits = fl.find_near_duplicates_fast(
        [r["text"] for r in uniq_rows], ngram_n=5, jaccard_thresh=jaccard_thresh
    )
    drop_idx: Set[int] = set()
    for u, v, _ in hits:
        if u not in drop_idx:
            drop_idx.add(v)
    kept = [r for i, r in enumerate(uniq_rows) if i not in drop_idx]
    return kept, {"rule_drop": rule_drop, "exact_drop": exact_drop, "lsh_drop": len(drop_idx)}


def filter_against_index(
    rows: List[dict],
    seen_norms: Set[str],
    lsh_idx: fl.LSHIndex,
    jaccard_thresh: float = 0.70,
) -> Tuple[List[dict], dict]:
    step1 = []
    norm_drop = 0
    for r in rows:
        nt = sr.normalize_for_dedup(r["text"])
        if nt in seen_norms:
            norm_drop += 1
        else:
            step1.append(r)

    matched, _ = lsh_idx.query_matches([r["text"] for r in step1], jaccard_thresh=jaccard_thresh)
    kept = [r for r, m in zip(step1, matched) if not m]
    lsh_drop = len(step1) - len(kept)
    return kept, {"cross_norm_drop": norm_drop, "cross_lsh_drop": lsh_drop, "cross_total_drop": norm_drop + lsh_drop}


def allocate_subtype_quotas(
    avail_by_sub: Dict[str, int],
    n_target: int,
    weights: Dict[str, float],
) -> Tuple[Dict[str, int], int]:
    """Largest-remainder 配分＋欠損サブタイプの自動補填を行い、合計 n_target 件の割当を返す。"""
    total_avail = sum(avail_by_sub.get(s, 0) for s in weights)
    if total_avail < n_target:
        # 想定外サブタイプがあれば含めて再確認
        total_all = sum(avail_by_sub.values())
        if total_all < n_target:
            raise RuntimeError(f"Insufficient rows: need {n_target}, got {total_all}")

    raw = {s: n_target * w for s, w in weights.items()}
    quotas = {s: int(v) for s, v in raw.items()}
    rem = n_target - sum(quotas.values())
    for s in sorted(weights.keys(), key=lambda k: raw[k] - quotas[k], reverse=True):
        if rem <= 0:
            break
        quotas[s] += 1
        rem -= 1

    # 利用可能上限でキャップし、不足分を残余サブタイプへ再配分
    shortfall = 0
    for s in list(quotas.keys()):
        cap = avail_by_sub.get(s, 0)
        if quotas[s] > cap:
            shortfall += quotas[s] - cap
            quotas[s] = cap

    shortfall_orig = shortfall
    if shortfall > 0:
        slack_order = sorted(
            avail_by_sub.keys(),
            key=lambda s: avail_by_sub[s] - quotas.get(s, 0),
            reverse=True,
        )
        while shortfall > 0:
            progressed = False
            for s in slack_order:
                slack = avail_by_sub[s] - quotas.get(s, 0)
                if slack > 0:
                    quotas[s] = quotas.get(s, 0) + 1
                    shortfall -= 1
                    progressed = True
                    if shortfall == 0:
                        break
            if not progressed:
                raise RuntimeError(f"Cannot redistribute shortfall={shortfall}")

    assert sum(quotas.values()) == n_target
    return quotas, shortfall_orig


def pick_stratified_within_bucket(rows: List[dict], n_pick: int, seed: int) -> List[dict]:
    if n_pick == 0:
        return []
    if len(rows) < n_pick:
        raise RuntimeError(f"Bucket underflow: need {n_pick}, got {len(rows)}")
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r.get("domain", ""), r.get("length_band", ""))].append(r)
    keys = sorted(buckets.keys())
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(buckets[k])
        buckets[k].sort(key=lambda x: int(x.get("verifier_naturalness") or 0), reverse=True)

    picked: List[dict] = []
    while len(picked) < n_pick:
        progressed = False
        for k in keys:
            if buckets[k]:
                picked.append(buckets[k].pop(0))
                progressed = True
                if len(picked) >= n_pick:
                    break
        if not progressed:
            break
    return picked


def format_record(r: dict, idx: int, prefix: str) -> dict:
    lbl = r["intended_label"]
    src_batch = "prod_add" if str(r.get("sample_id", "")).startswith("prod_") else "pilot"
    return {
        "id": f"prod_{prefix}_{idx:05d}",
        "sample_id": r["sample_id"],
        "split": "sealed" if prefix == "sea" else ("dev" if prefix == "dev" else "train"),
        "source_batch": src_batch,
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
    ap.add_argument(
        "--inputs",
        default=f"{ROOT / 'data' / 'synth' / 'synth_verified.jsonl'},{ROOT / 'data' / 'prod_synth' / 'synth_verified.jsonl'}",
    )
    ap.add_argument("--human-eval", default=str(ROOT / "data" / "eval" / "human-eval.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "prod"))
    ap.add_argument("--seed", type=int, default=20261122)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 入力合流（sample_id 重複排除）
    seen_sids: Set[str] = set()
    by_pool: Dict[str, List[dict]] = defaultdict(list)
    input_counts = {}
    for in_str in [x.strip() for x in args.inputs.split(",") if x.strip()]:
        p = Path(in_str)
        rows = load_jsonl(p)
        kept_cnt = 0
        for r in rows:
            if not r.get("accepted"):
                continue
            sid = r["sample_id"]
            if sid in seen_sids:
                continue
            seen_sids.add(sid)
            by_pool[r["split_pool"]].append(r)
            kept_cnt += 1
        input_counts[str(p)] = kept_cnt

    # human-eval 600件を交差除外インデックスへ先登録
    he_rows = load_jsonl(Path(args.human_eval))
    seen_norms: Set[str] = {sr.normalize_for_dedup(r["text"]) for r in he_rows if sr.normalize_for_dedup(r["text"])}
    lsh_idx = fl.LSHIndex(ngram_n=5, num_perm=64, bands=16, seed=42)
    lsh_idx.add_texts([r["text"] for r in he_rows])

    dedup_stats = {}
    cleaned: Dict[str, List[dict]] = {}
    for pool in ("dev", "sealed_test", "train"):
        t_p = time.time()
        in_rows = by_pool[pool]
        u_rows, st_in = dedup_within_pool_fast(in_rows, jaccard_thresh=0.75)
        u_rows2, st_cross = filter_against_index(u_rows, seen_norms, lsh_idx, jaccard_thresh=0.70)
        for r in u_rows2:
            nt = sr.normalize_for_dedup(r["text"])
            if nt:
                seen_norms.add(nt)
        lsh_idx.add_texts([r["text"] for r in u_rows2])
        cleaned[pool] = u_rows2
        dedup_stats[pool] = {
            "raw_accepted": len(in_rows),
            **st_in,
            **st_cross,
            "available_after_dedup": len(u_rows2),
            "available_by_label": dict(Counter(x["intended_label"] for x in u_rows2)),
            "elapsed_sec": round(time.time() - t_p, 2),
        }

    print("[PROD DEDUP]", json.dumps(dedup_stats, ensure_ascii=False, indent=2), flush=True)

    out_files = {
        "train": (out_dir / "train.jsonl", "tra"),
        "dev": (out_dir / "dev.jsonl", "dev"),
        "sealed_test": (out_dir / "sealed-test.jsonl", "sea"),
    }
    split_stats = {}
    rng = random.Random(args.seed)

    for pool, (out_path, prefix) in out_files.items():
        targets = TARGET_COUNTS[pool]
        pool_rows = cleaned[pool]
        selected: List[dict] = []
        quota_logs = {}
        for lbl, n_t in targets.items():
            cls_rows = [r for r in pool_rows if r["intended_label"] == lbl]
            by_sub: Dict[str, List[dict]] = defaultdict(list)
            for r in cls_rows:
                by_sub[r.get("subtype", "unknown")].append(r)
            avail_counts = {s: len(lst) for s, lst in by_sub.items()}
            quotas, shortfall = allocate_subtype_quotas(avail_counts, n_t, SUBTYPE_WEIGHTS[lbl])
            quota_logs[lbl] = {"quotas": quotas, "shortfall_redistributed": shortfall}
            for s, q_s in sorted(quotas.items()):
                picked = pick_stratified_within_bucket(
                    by_sub[s], q_s, seed=args.seed + sr.LABEL2ID[lbl] * 100 + hash(s) % 97
                )
                selected.extend(picked)

        rng.shuffle(selected)
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
            "by_source_batch": dict(Counter(x["source_batch"] for x in formatted)),
            "by_domain": dict(Counter(x["domain"] for x in formatted)),
            "by_difficulty": dict(Counter(x["difficulty"] for x in formatted)),
            "by_subtype": dict(Counter(x["subtype"] for x in formatted)),
            "by_length_band": dict(Counter(x["length_band"] for x in formatted)),
            "unique_subtopics": len({f"{x['domain']}::{x['subtopic']}" for x in formatted}),
            "quota_logs": quota_logs,
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
        "input_accepted_counts": input_counts,
        "human_eval_excluded_count": len(he_rows),
        "dedup_stats": dedup_stats,
        "splits": split_stats,
    }
    (out_dir / "build-stats.json").write_text(
        json.dumps(build_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[PROD BUILD DONE] elapsed={build_stats['build_sec']}s")
    for pool, st in split_stats.items():
        print(f"  {pool:12s}: rows={st['rows']}, by_label={st['by_label']}, sha256={st['sha256'][:12]}...")


if __name__ == "__main__":
    main()
