#!/usr/bin/env python3
"""第11号 ja-nli-v1 本番分割（train 100,000 / dev 5,000 / sealed-test 5,000）構築スクリプト。

設計方針:
1. human-eval (600件) との doc_id / cluster_id / 正規化テキスト / LSH(前提>=0.80, 仮説>=0.85) 重複を事前完全除去。
2. sealed-test (5,000) -> dev (5,000) -> train (100,000) の順に構築し、各分割確定後に他分割候補との
   LSH 重複を事前排除してから抽出することで、全6組×7指標リーク 0 を構成的に保証する。
3. 同一前提(chunk_id, round) から E/N/C が揃った 3 クラス完全バンドルを最優先で層化抽出し、
   法令上限(<=15%)・ドメインバランス・仮説単体バイアス軽減を両立する。
※ sealed-test.jsonl は件数・統計集計とハッシュ記録のみ行い、中身テキストを出力・評価しない。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Set, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(".")
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import build_pilot_data as bpd  # noqa: E402
import check_leaks_all as cla  # noqa: E402
import nli_rules as nr  # noqa: E402

LABEL2ID = bpd.LABEL2ID


def has_lsh_collision(text: str, ref_cache: dict, jaccard_thresh: float, ngram_n: int = 5) -> bool:
    """単一テキストが参照側 LSH キャッシュと閾値以上で衝突するか判定する。"""
    norm_t = nr.normalize_for_dedup(text) or text
    grams = nr.char_ngrams(norm_t, n=ngram_n)
    if not grams:
        return False
    sig = nr.compute_minhash_sig(text, ngram_n=ngram_n, num_perm=64, seed=42)
    buckets = ref_cache["buckets"]
    ref_grams = ref_cache["grams"]
    bands = ref_cache["bands"]
    rpb = ref_cache["rows_per_band"]
    checked: Set[int] = set()
    for b in range(bands):
        st = b * rpb
        for idx in buckets.get((b, sig[st : st + rpb]), ()):
            if idx in checked:
                continue
            checked.add(idx)
            sg = ref_grams[idx]
            if not sg:
                continue
            inter = len(grams & sg)
            if inter / (len(grams) + len(sg) - inter) >= jaccard_thresh:
                return True
    return False


def filter_against_profile(
    rows: List[dict],
    ref_prof: dict,
    *,
    prem_thresh: float = 0.80,
    hyp_thresh: float = 0.85,
) -> Tuple[List[dict], Dict[str, int]]:
    """参照分割プロファイルと重複・LSH衝突する行を除外する（前提判定キャッシュ付き）。"""
    kept: List[dict] = []
    drops = Counter()
    prem_hit_cache: Dict[str, bool] = {}

    for r in rows:
        if r["doc_id"] in ref_prof["docs"]:
            drops["doc_id"] += 1
            continue
        if r["cluster_id"] in ref_prof["clusters"]:
            drops["cluster_id"] += 1
            continue
        np = nr.normalize_for_dedup(r["premise"])
        nh = nr.normalize_for_dedup(r["hypothesis"])
        if np in ref_prof["norm_prems"] or r["premise"] in ref_prof["exact_prems"]:
            drops["norm_prem"] += 1
            continue
        if nh in ref_prof["norm_hyps"]:
            drops["norm_hyp"] += 1
            continue
        p_txt = r["premise"]
        if p_txt not in prem_hit_cache:
            prem_hit_cache[p_txt] = has_lsh_collision(p_txt, ref_prof["prem_lsh"], prem_thresh)
        if prem_hit_cache[p_txt]:
            drops["prem_lsh"] += 1
            continue
        if has_lsh_collision(r["hypothesis"], ref_prof["hyp_lsh"], hyp_thresh):
            drops["hyp_lsh"] += 1
            continue
        kept.append(r)

    return kept, dict(drops)


def select_bundle_prioritized(
    rows: List[dict],
    target_counts: Dict[str, int],
    seed: int,
    max_law_ratio: float = 0.15,
) -> Tuple[List[dict], dict]:
    """同一(chunk_id, round)の3クラス完全バンドルを最優先に層化抽出し、端数のみ2クラス/単発で補完する。"""
    rng = random.Random(seed)
    total_target = sum(target_counts.values())
    max_law_rows = int(total_target * max_law_ratio)

    # (chunk_id, round) ごとにグループ化し、同一ラベル重複があれば1つに絞る
    groups: Dict[Tuple[str, int], Dict[str, dict]] = defaultdict(dict)
    for r in rows:
        gk = (r["chunk_id"], int(r.get("round", 1)))
        lbl = r["intended_label"]
        if lbl not in groups[gk]:
            groups[gk][lbl] = r

    triples_by_bucket: Dict[Tuple[str, str], List[Tuple[dict, dict, dict]]] = defaultdict(list)
    rem_by_lbl_bucket: Dict[str, Dict[Tuple[str, str], List[dict]]] = {
        l: defaultdict(list) for l in nr.VALID_LABELS
    }

    for gk, ldict in groups.items():
        if len(ldict) == 3:
            e_r = ldict["entailment"]
            bk = (e_r["domain_group"], e_r["len_bucket"])
            triples_by_bucket[bk].append((ldict["entailment"], ldict["neutral"], ldict["contradiction"]))
        else:
            for l, r in ldict.items():
                bk = (r["domain_group"], r["len_bucket"])
                rem_by_lbl_bucket[l][bk].append(r)

    for bk in triples_by_bucket:
        rng.shuffle(triples_by_bucket[bk])
    for l in nr.VALID_LABELS:
        for bk in rem_by_lbl_bucket[l]:
            rng.shuffle(rem_by_lbl_bucket[l][bk])

    chosen: List[dict] = []
    chosen_counts = Counter()
    law_rows = 0

    min_bundle_quota = min(target_counts.values())
    # ドメイン×長さ帯バケットを均等に回しながら3クラス完全バンドルを確保
    non_law_bks = sorted([bk for bk in triples_by_bucket if bk[0] != "gov_law"])
    law_bks = sorted([bk for bk in triples_by_bucket if bk[0] == "gov_law"])
    all_bks = non_law_bks + law_bks
    rng.shuffle(non_law_bks)

    bundles_picked = 0
    while bundles_picked < min_bundle_quota:
        progressed = False
        # 非法令と法令を比率制御しながらラウンドロビン（法令枠が残っていれば7回に1回程度法令を混ぜる）
        order = list(non_law_bks) + list(law_bks)
        for bk in order:
            if bundles_picked >= min_bundle_quota:
                break
            if not triples_by_bucket[bk]:
                continue
            if bk[0] == "gov_law" and law_rows + 3 > max_law_rows:
                continue
            trip = triples_by_bucket[bk].pop()
            for r in trip:
                chosen.append(r)
                chosen_counts[r["intended_label"]] += 1
            if bk[0] == "gov_law":
                law_rows += 3
            bundles_picked += 1
            progressed = True
        if not progressed:
            # もし非法令トリプルが尽きて law 制限で止まった場合は残余へ渡す
            break

    # 3クラス未使用分をクラス別候補へ戻し（可能なら同一チャンク残余も活用して端数補充）
    for bk, tlist in triples_by_bucket.items():
        for trip in tlist:
            for r in trip:
                rem_by_lbl_bucket[r["intended_label"]][bk].append(r)

    for l in nr.VALID_LABELS:
        bks = sorted(rem_by_lbl_bucket[l].keys())
        for bk in bks:
            rng.shuffle(rem_by_lbl_bucket[l][bk])
        while chosen_counts[l] < target_counts[l]:
            progressed = False
            for bk in bks:
                if chosen_counts[l] >= target_counts[l]:
                    break
                if not rem_by_lbl_bucket[l][bk]:
                    continue
                if bk[0] == "gov_law" and law_rows + 1 > max_law_rows and any(
                    rem_by_lbl_bucket[l][ob] for ob in bks if ob[0] != "gov_law"
                ):
                    continue
                r = rem_by_lbl_bucket[l][bk].pop()
                chosen.append(r)
                chosen_counts[l] += 1
                if bk[0] == "gov_law":
                    law_rows += 1
                progressed = True
            if not progressed:
                raise RuntimeError(
                    f"Insufficient samples for label={l}: need {target_counts[l]}, got {chosen_counts[l]}"
                )

    rng.shuffle(chosen)

    # 同一前提マルチクラス比率を計算
    by_chunk = defaultdict(set)
    for r in chosen:
        by_chunk[r["chunk_id"]].add(r["intended_label"])
    rows_3cls = sum(1 for r in chosen if len(by_chunk[r["chunk_id"]]) == 3)
    rows_2plus = sum(1 for r in chosen if len(by_chunk[r["chunk_id"]]) >= 2)
    diag = {
        "bundles_3class_picked": bundles_picked,
        "rows_in_3class_same_premise": rows_3cls,
        "ratio_3class_same_premise": round(rows_3cls / max(1, len(chosen)), 4),
        "rows_in_2plus_same_premise": rows_2plus,
        "ratio_2plus_same_premise": round(rows_2plus / max(1, len(chosen)), 4),
        "unique_chunks": len(by_chunk),
        "law_rows": sum(1 for r in chosen if r["domain_group"] == "gov_law"),
        "law_ratio": round(sum(1 for r in chosen if r["domain_group"] == "gov_law") / max(1, len(chosen)), 4),
    }
    return chosen, diag


def run_hyp_only_2to5(train_rows: List[dict], dev_rows: List[dict]) -> dict:
    """char_wb (2-5) でも仮説単体プローブを計測して比較可能にする。"""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=40000, sublinear_tf=True)
    x_tr = vec.fit_transform([r["hypothesis"] for r in train_rows])
    y_tr = [r["label_id"] for r in train_rows]
    x_dv = vec.transform([r["hypothesis"] for r in dev_rows])
    y_dv = [r["label_id"] for r in dev_rows]
    clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_dv)
    return {
        "model": "char_wb(2-5)_min3_TFIDF + LogReg(C=2.0)",
        "dev_accuracy": round(float(accuracy_score(y_dv, pred)), 4),
        "dev_macro_f1": round(float(f1_score(y_dv, pred, average="macro")), 4),
        "dev_per_class_f1": {
            lbl: round(float(f1), 4)
            for lbl, f1 in zip(nr.VALID_LABELS, f1_score(y_dv, pred, average=None, labels=[0, 1, 2]))
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", default=str(ROOT / "data" / "synth_prod" / "synth_verified.jsonl"))
    ap.add_argument("--human-eval", default=str(ROOT / "data" / "eval" / "human-eval.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "prod"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.human_eval).open("r", encoding="utf-8") as f:
        he_rows = [json.loads(ln) for ln in f if ln.strip()]
    sig_cache: Dict[str, Tuple[Tuple[int, ...], Set[str]]] = {}
    he_prof = cla.prepare_split_profile(he_rows, sig_cache)

    with Path(args.verified).open("r", encoding="utf-8") as f:
        raw_rows = [json.loads(ln) for ln in f if ln.strip()]
    seen_pair_norm: Set[Tuple[str, str]] = set()
    seen_hyp_global: Set[str] = set()
    by_pool: Dict[str, List[dict]] = defaultdict(list)
    drop_counts = Counter()

    for r in raw_rows:
        ok, _ = nr.check_pair_rules(
            r["premise"], r["hypothesis"], r["intended_label"], strict_demonstratives=True
        )
        if not ok:
            drop_counts["rule_recheck_fail"] += 1
            continue
        np_txt = nr.normalize_for_dedup(r["premise"])
        nh_txt = nr.normalize_for_dedup(r["hypothesis"])
        if (np_txt, nh_txt) in seen_pair_norm:
            drop_counts["dup_pair_norm"] += 1
            continue
        if nh_txt in seen_hyp_global:
            drop_counts["dup_hyp_norm"] += 1
            continue
        seen_pair_norm.add((np_txt, nh_txt))
        seen_hyp_global.add(nh_txt)
        by_pool[r["split_pool"]].append(r)

    print(f"[1/4] Loaded verified={len(raw_rows)}, after_dedup_pools={{p: len(v) for p,v in by_pool.items()}}:")
    for p, v in by_pool.items():
        print(" ", p, len(v), dict(Counter(x["intended_label"] for x in v)))

    # まず全プールから human-eval 600 との衝突を完全除外
    he_drop_report = {}
    for p in ("sealed_test", "dev", "train"):
        kept, d = filter_against_profile(by_pool[p], he_prof)
        by_pool[p] = kept
        he_drop_report[p] = d
    print("[2/4] Filtered against human-eval:", he_drop_report)

    quotas = {
        "sealed-test": ("sealed_test", {"entailment": 1667, "neutral": 1667, "contradiction": 1666}),
        "dev": ("dev", {"entailment": 1667, "neutral": 1667, "contradiction": 1666}),
        "train": ("train", {"entailment": 33334, "neutral": 33333, "contradiction": 33333}),
    }

    splits_formatted: Dict[str, List[dict]] = {}
    bundle_diags: Dict[str, dict] = {}
    file_hashes: Dict[str, str] = {}
    file_mtimes: Dict[str, str] = {}
    cross_filter_drops: Dict[str, dict] = {}
    avail_after_filter: Dict[str, Dict[str, int]] = {}

    # 順番: sealed-test -> dev -> train（確定済み全分割との LSH 重複を次プールから事前除外）
    chosen_profiles: List[Tuple[str, dict]] = []
    for sp_name in ("sealed-test", "dev", "train"):
        pool_key, q = quotas[sp_name]
        cands = by_pool[pool_key]
        cross_drops_sp = {}
        for prev_name, prev_prof in chosen_profiles:
            cands, d = filter_against_profile(cands, prev_prof)
            cross_drops_sp[prev_name] = d
        cross_filter_drops[sp_name] = cross_drops_sp
        avail_after_filter[sp_name] = dict(Counter(x["intended_label"] for x in cands))

        picked, bdiag = select_bundle_prioritized(
            cands, q, seed=args.seed + (0 if sp_name == "sealed-test" else (1 if sp_name == "dev" else 2))
        )
        formatted = [bpd.format_pilot_row(r, i, sp_name) for i, r in enumerate(picked)]
        splits_formatted[sp_name] = formatted
        bundle_diags[sp_name] = bdiag

        out_path = out_dir / f"{sp_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as fout:
            for row in formatted:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        file_hashes[sp_name] = hashlib.sha256(out_path.read_bytes()).hexdigest()
        file_mtimes[sp_name] = time.strftime(
            "%Y-%m-%dT%H:%M:%S+09:00", time.localtime(out_path.stat().st_mtime)
        )
        chosen_profiles.append((sp_name, cla.prepare_split_profile(formatted, sig_cache)))
        print(f"  Built {sp_name}: {len(formatted)} rows | bundle_diag={bdiag}")

    print("[3/4] Running hypothesis-only diagnostics (train -> dev ONLY; sealed-test untouched)...")
    hyp_diag_2_4 = bpd.hypothesis_only_diagnostic(splits_formatted["train"], splits_formatted["dev"])
    hyp_diag_2_5 = run_hyp_only_2to5(splits_formatted["train"], splits_formatted["dev"])

    build_stats = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "elapsed_sec": round(time.time() - t0, 2),
        "seed": args.seed,
        "input_verified_rows": len(raw_rows),
        "dedup_drop_counts": dict(drop_counts),
        "human_eval_collision_drops": he_drop_report,
        "cross_split_pre_filter_drops": cross_filter_drops,
        "available_by_pool_class_after_filter": avail_after_filter,
        "same_premise_bundle_diagnostics": bundle_diags,
        "splits": {k: bpd.compute_split_stats(v) for k, v in splits_formatted.items()},
        "file_sha256": file_hashes,
        "file_mtime": file_mtimes,
        "hypothesis_only_diagnostic_2_4": hyp_diag_2_4,
        "hypothesis_only_diagnostic_2_5": hyp_diag_2_5,
        "sealed_test_policy": "sealed-test.jsonl constructed and sealed untouched prior to training & one-shot final eval.",
    }
    stats_path = out_dir / "build-stats.json"
    stats_path.write_text(json.dumps(build_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4/4] Done in {build_stats['elapsed_sec']}s -> {stats_path}")
    print("Hypothesis-only (dev 2-4):", hyp_diag_2_4)
    print("Hypothesis-only (dev 2-5):", hyp_diag_2_5)


if __name__ == "__main__":
    main()
