#!/usr/bin/env python
"""第11号 ja-nli-v1 パイロット分割（train 20,000 / dev 2,000 / sealed-test 2,000）構築スクリプト。

機能:
1. 独立検証済み合成データ (data/synth/synth_verified.jsonl) に対し最終規則フィルタ・正規化重複除去を適用
2. 事前分離済みの split_pool ごとに 3クラス均等 (E/N/C = 1:1:1) で層化サンプリング
   - train.jsonl: 20,000 (6,667 / 6,667 / 6,666)
   - dev.jsonl: 2,000 (667 / 667 / 666)
   - sealed-test.jsonl: 2,000 (667 / 667 / 666) ※構築後未開封厳守
3. 仮説単体バイアス診断 (Hypothesis-Only TF-IDF + LogisticRegression) を train -> dev で実測
4. 分割間リーク検査 (train / dev / sealed-test) を実行し build-stats.json を出力
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nli_rules as nr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


def select_class_balanced(
    rows: List[dict],
    target_counts: Dict[str, int],
    seed: int,
) -> Tuple[List[dict], Dict[str, int]]:
    """ドメイン×長さ帯の多様性を保ちつつ、クラス別に厳密定数をサンプリングする。"""
    rng = random.Random(seed)
    by_lbl: Dict[str, List[dict]] = {lbl: [] for lbl in nr.VALID_LABELS}
    for r in rows:
        by_lbl[r["intended_label"]].append(r)

    avail = {lbl: len(v) for lbl, v in by_lbl.items()}
    picked_all: List[dict] = []

    for lbl, quota in target_counts.items():
        cands = by_lbl[lbl]
        if len(cands) < quota:
            raise RuntimeError(
                f"Insufficient verified samples for label={lbl}: need {quota}, have {len(cands)}"
            )
        # (domain_group, len_bucket) で層化ラウンドロビン抽出
        buckets: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for item in cands:
            buckets[(item["domain_group"], item["len_bucket"])].append(item)
        keys = sorted(buckets.keys())
        for k in keys:
            rng.shuffle(buckets[k])
        rng.shuffle(keys)

        chosen: List[dict] = []
        used_chunk_ids: Set[str] = set()
        # まず同一チャンク内で過度に偏らないようラウンドロビン
        idx = 0
        while len(chosen) < quota:
            progressed = False
            for k in keys:
                if not buckets[k]:
                    continue
                chosen.append(buckets[k].pop())
                progressed = True
                if len(chosen) >= quota:
                    break
            if not progressed:
                break
            idx += 1
        picked_all.extend(chosen)

    rng.shuffle(picked_all)
    return picked_all, avail


def format_pilot_row(r: dict, row_idx: int, split_name: str) -> dict:
    lbl = r["intended_label"]
    return {
        "id": f"{split_name}_{row_idx:05d}",
        "pair_id": r["pair_id"],
        "chunk_id": r["chunk_id"],
        "doc_id": r["doc_id"],
        "cluster_id": r["cluster_id"],
        "source": r["source"],
        "domain_group": r["domain_group"],
        "license": r["license"],
        "len_bucket": r["len_bucket"],
        "subtype": r.get("subtype", ""),
        "premise": r["premise"],
        "hypothesis": r["hypothesis"],
        "label": lbl,
        "label_id": LABEL2ID[lbl],
        "verifier_naturalness": r.get("verifier_naturalness"),
        "generator_model": r.get("generator_model"),
        "verifier_model": r.get("verifier_model"),
    }


def compute_split_stats(rows: List[dict]) -> dict:
    p_lens = [len(r["premise"]) for r in rows]
    h_lens = [len(r["hypothesis"]) for r in rows]
    return {
        "total_rows": len(rows),
        "by_label": dict(Counter(r["label"] for r in rows)),
        "by_domain": dict(Counter(r["domain_group"] for r in rows)),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_len_bucket": dict(Counter(r["len_bucket"] for r in rows)),
        "by_subtype": dict(Counter(r["subtype"] for r in rows)),
        "unique_docs": len({r["doc_id"] for r in rows}),
        "unique_clusters": len({r["cluster_id"] for r in rows}),
        "premise_chars": {
            "min": min(p_lens),
            "median": statistics.median(p_lens),
            "mean": round(statistics.mean(p_lens), 2),
            "max": max(p_lens),
        },
        "hypothesis_chars": {
            "min": min(h_lens),
            "median": statistics.median(h_lens),
            "mean": round(statistics.mean(h_lens), 2),
            "max": max(h_lens),
        },
    }


def hypothesis_only_diagnostic(train_rows: List[dict], dev_rows: List[dict]) -> dict:
    """仮説文のみを使った TF-IDF + LogisticRegression でラベルバイアスを診断する。"""
    x_tr = [r["hypothesis"] for r in train_rows]
    y_tr = [r["label_id"] for r in train_rows]
    x_dv = [r["hypothesis"] for r in dev_rows]
    y_dv = [r["label_id"] for r in dev_rows]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=30000, sublinear_tf=True)
    X_tr = vec.fit_transform(x_tr)
    X_dv = vec.transform(x_dv)

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_dv)
    return {
        "model": "char_wb(2-4)_TFIDF + LogisticRegression",
        "random_baseline_acc": round(1.0 / 3.0, 4),
        "dev_accuracy": round(float(accuracy_score(y_dv, pred)), 4),
        "dev_macro_f1": round(float(f1_score(y_dv, pred, average="macro")), 4),
        "dev_per_class_f1": {
            lbl: round(float(f1), 4)
            for lbl, f1 in zip(nr.VALID_LABELS, f1_score(y_dv, pred, average=None, labels=[0, 1, 2]))
        },
    }


def check_pair_leaks(name_a: str, rows_a: List[dict], name_b: str, rows_b: List[dict]) -> dict:
    docs_a = {r["doc_id"] for r in rows_a}
    docs_b = {r["doc_id"] for r in rows_b}
    cl_a = {r["cluster_id"] for r in rows_a}
    cl_b = {r["cluster_id"] for r in rows_b}
    np_a = {nr.normalize_for_dedup(r["premise"]) for r in rows_a}
    np_b = {nr.normalize_for_dedup(r["premise"]) for r in rows_b}
    nh_a = {nr.normalize_for_dedup(r["hypothesis"]) for r in rows_a}
    nh_b = {nr.normalize_for_dedup(r["hypothesis"]) for r in rows_b}

    prems_a = list({r["premise"] for r in rows_a})
    prems_b = list({r["premise"] for r in rows_b})
    hyps_a = [r["hypothesis"] for r in rows_a]
    hyps_b = [r["hypothesis"] for r in rows_b]

    prem_lsh_080 = nr.lsh_cross_split_leaks(prems_a, prems_b, ngram_n=5, jaccard_thresh=0.80)
    hyp_lsh_085 = nr.lsh_cross_split_leaks(hyps_a, hyps_b, ngram_n=5, jaccard_thresh=0.85)
    return {
        "doc_id_overlap": len(docs_a & docs_b),
        "cluster_id_overlap": len(cl_a & cl_b),
        "exact_norm_premise_overlap": len(np_a & np_b),
        "exact_norm_hypothesis_overlap": len(nh_a & nh_b),
        "premise_lsh_080": prem_lsh_080,
        "hypothesis_lsh_085": hyp_lsh_085,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", default=str(ROOT / "data" / "synth" / "synth_verified.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "pilot"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = [json.loads(ln) for ln in Path(args.verified).read_text(encoding="utf-8").splitlines() if ln.strip()]

    # 最終規則再検 & 重複除去
    seen_pair_norm: Set[Tuple[str, str]] = set()
    seen_hyp_global: Set[str] = set()
    by_pool: Dict[str, List[dict]] = defaultdict(list)
    drop_counts = Counter()

    for r in raw_rows:
        ok, reasons = nr.check_pair_rules(r["premise"], r["hypothesis"], r["intended_label"])
        if not ok:
            drop_counts["rule_recheck_fail"] += 1
            continue
        np = nr.normalize_for_dedup(r["premise"])
        nh = nr.normalize_for_dedup(r["hypothesis"])
        if (np, nh) in seen_pair_norm:
            drop_counts["dup_pair_norm"] += 1
            continue
        if nh in seen_hyp_global:
            drop_counts["dup_hyp_norm"] += 1
            continue
        seen_pair_norm.add((np, nh))
        seen_hyp_global.add(nh)
        by_pool[r["split_pool"]].append(r)

    quotas = {
        "train": {"entailment": 6667, "neutral": 6667, "contradiction": 6666},
        "dev": {"entailment": 667, "neutral": 667, "contradiction": 666},
        "sealed-test": {"entailment": 667, "neutral": 667, "contradiction": 666},
    }
    pool_map = {"train": "train", "dev": "dev", "sealed-test": "sealed_test"}

    splits_formatted: Dict[str, List[dict]] = {}
    avail_map: Dict[str, Dict[str, int]] = {}
    file_hashes: Dict[str, str] = {}

    for sp_name, q in quotas.items():
        pool_key = pool_map[sp_name]
        picked, avail = select_class_balanced(by_pool[pool_key], q, seed=args.seed + hash(sp_name) % 1000)
        avail_map[sp_name] = avail
        formatted = [format_pilot_row(r, i, sp_name) for i, r in enumerate(picked)]
        splits_formatted[sp_name] = formatted

        out_path = out_dir / f"{sp_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as fout:
            for row in formatted:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        file_hashes[sp_name] = hashlib.sha256(out_path.read_bytes()).hexdigest()

    # 仮説単体バイアス診断（train -> dev のみ。sealed-test には一切触れない）
    hyp_diag = hypothesis_only_diagnostic(splits_formatted["train"], splits_formatted["dev"])

    # 3分割リーク検査
    leaks = {
        "train__dev": check_pair_leaks("train", splits_formatted["train"], "dev", splits_formatted["dev"]),
        "train__sealed-test": check_pair_leaks("train", splits_formatted["train"], "sealed-test", splits_formatted["sealed-test"]),
        "dev__sealed-test": check_pair_leaks("dev", splits_formatted["dev"], "sealed-test", splits_formatted["sealed-test"]),
    }

    build_stats = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "elapsed_sec": round(time.time() - t0, 2),
        "seed": args.seed,
        "input_verified_rows": len(raw_rows),
        "dedup_drop_counts": dict(drop_counts),
        "available_by_pool_class": avail_map,
        "splits": {k: compute_split_stats(v) for k, v in splits_formatted.items()},
        "file_sha256": file_hashes,
        "hypothesis_only_diagnostic": hyp_diag,
        "leaks": leaks,
        "sealed_test_policy": "sealed-test.jsonl constructed, audited for counts/leaks only, and sealed without opening or evaluation.",
    }
    stats_path = out_dir / "build-stats.json"
    stats_path.write_text(json.dumps(build_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Built pilot splits in {build_stats['elapsed_sec']}s:")
    for sp_name, st in build_stats["splits"].items():
        print(f"  {sp_name}: total={st['total_rows']}, labels={st['by_label']}, docs={st['unique_docs']}, clusters={st['unique_clusters']}")
    print("Hypothesis-only diagnostic (dev):", hyp_diag)
    print("Leaks:", leaks)


if __name__ == "__main__":
    main()
