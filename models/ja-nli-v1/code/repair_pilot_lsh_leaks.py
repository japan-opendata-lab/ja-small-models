#!/usr/bin/env python
"""第11号 ja-nli-v1 train/dev LSH 衝突局所差し替えスクリプト（sealed-test 完全非改変・未開封）。

機能:
1. data/pilot/sealed-test.jsonl の SHA256 不変を事前事後に検証（テキスト出力は一切行わない）。
2. Phase 1 (dev): sealed-test と LSH 衝突（premise>=0.80, hypothesis>=0.85）する dev 行のみを、
   synth_verified.jsonl (split_pool=="dev") の同一ラベル未使用・無衝突候補で差し替える
   （audit_sample_200 との衝突も禁止）。
3. Phase 2 (train): sealed-test または更新後 dev と LSH 衝突する train 行を特定し、
   audit_sample_200.jsonl と重複 0 件であることを assert したうえで、
   synth_verified.jsonl (split_pool=="train") の同一ラベル未使用・無衝突候補で差し替える。
4. dev.jsonl / train.jsonl / build-stats.json を更新し、全3分割間リーク 0 を検証、
   human_eval_candidates.jsonl との LSH 衝突候補インデックス一覧を出力する。
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pilot_data as bpd  # noqa: E402
import check_leaks_all as cla  # noqa: E402
import nli_rules as nr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SEALED_SHA256 = "462fd3f1524a1590e7efc8a818aac1dc1e36f6068cecb741c30834c58e47c7ea"


def load_jsonl(p: Path) -> List[dict]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def save_jsonl(p: Path, rows: List[dict]) -> None:
    with p.open("w", encoding="utf-8") as fout:
        for r in rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_ref_lsh(
    texts: List[str],
    sig_cache: Dict[str, Tuple[Tuple[int, ...], Set[str]]],
    ngram_n: int = 5,
) -> dict:
    unique_texts = list(dict.fromkeys(texts))
    return cla.build_lsh_cache(unique_texts, sig_cache, ngram_n=ngram_n, num_perm=64, bands=16)


def has_lsh_collision(
    text: str,
    ref_lsh: dict,
    sig_cache: Dict[str, Tuple[Tuple[int, ...], Set[str]]],
    thresh: float,
    ngram_n: int = 5,
) -> bool:
    if text not in sig_cache:
        norm_t = nr.normalize_for_dedup(text) or text
        sig_cache[text] = (
            nr.compute_minhash_sig(text, ngram_n=ngram_n, num_perm=64),
            nr.char_ngrams(norm_t, n=ngram_n),
        )
    sig, gr = sig_cache[text]
    if not gr:
        return False
    bands = ref_lsh["bands"]
    rpb = ref_lsh["rows_per_band"]
    buckets = ref_lsh["buckets"]
    grams_ref = ref_lsh["grams"]

    checked: Set[int] = set()
    for b in range(bands):
        st = b * rpb
        for idx in buckets.get((b, sig[st : st + rpb]), ()):
            if idx in checked:
                continue
            checked.add(idx)
            gr_ref = grams_ref[idx]
            if not gr_ref:
                continue
            inter = len(gr & gr_ref)
            jac = inter / (len(gr) + len(gr_ref) - inter)
            if jac >= thresh:
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", default=str(ROOT / "data" / "synth" / "synth_verified.jsonl"))
    ap.add_argument("--pilot-dir", default=str(ROOT / "data" / "pilot"))
    ap.add_argument("--audit-sample", default=str(ROOT / "data" / "audit" / "audit_sample_200.jsonl"))
    ap.add_argument("--he-cands", default=str(ROOT / "data" / "eval" / "human_eval_candidates.jsonl"))
    args = ap.parse_args()

    t0 = time.time()
    pilot_dir = Path(args.pilot_dir)
    sealed_path = pilot_dir / "sealed-test.jsonl"
    dev_path = pilot_dir / "dev.jsonl"
    train_path = pilot_dir / "train.jsonl"
    stats_path = pilot_dir / "build-stats.json"

    sealed_sha_before = sha256_file(sealed_path)
    assert sealed_sha_before == EXPECTED_SEALED_SHA256, (
        f"sealed-test.jsonl SHA256 mismatch before repair: {sealed_sha_before}"
    )

    sealed_rows = load_jsonl(sealed_path)
    dev_rows = load_jsonl(dev_path)
    train_rows = load_jsonl(train_path)
    audit_rows = load_jsonl(Path(args.audit_sample))
    audit_pair_ids = {r["pair_id"] for r in audit_rows}
    prev_stats = json.loads(stats_path.read_text(encoding="utf-8"))

    verified_rows = load_jsonl(Path(args.verified))
    sig_cache: Dict[str, Tuple[Tuple[int, ...], Set[str]]] = {}

    # 参照インデックス構築 (sealed-test + audit_200)
    sealed_prem_lsh = build_ref_lsh([r["premise"] for r in sealed_rows], sig_cache)
    sealed_hyp_lsh = build_ref_lsh([r["hypothesis"] for r in sealed_rows], sig_cache)
    audit_prem_lsh = build_ref_lsh([r["premise"] for r in audit_rows], sig_cache)
    audit_hyp_lsh = build_ref_lsh([r["hypothesis"] for r in audit_rows], sig_cache)

    sealed_norm_prems = {nr.normalize_for_dedup(r["premise"]) for r in sealed_rows}
    sealed_norm_hyps = {nr.normalize_for_dedup(r["hypothesis"]) for r in sealed_rows}
    sealed_docs = {r["doc_id"] for r in sealed_rows}
    sealed_clusters = {r["cluster_id"] for r in sealed_rows}

    used_pair_ids = {r["pair_id"] for r in sealed_rows + dev_rows + train_rows}
    used_norm_hyps = {nr.normalize_for_dedup(r["hypothesis"]) for r in sealed_rows + dev_rows + train_rows}
    used_pair_norms = {
        (nr.normalize_for_dedup(r["premise"]), nr.normalize_for_dedup(r["hypothesis"]))
        for r in sealed_rows + dev_rows + train_rows
    }

    # --- Phase 1: dev の LSH 衝突修正 (against sealed-test) ---
    dev_bad_idxs: List[int] = []
    for idx, r in enumerate(dev_rows):
        if (
            has_lsh_collision(r["premise"], sealed_prem_lsh, sig_cache, 0.80)
            or has_lsh_collision(r["hypothesis"], sealed_hyp_lsh, sig_cache, 0.85)
        ):
            dev_bad_idxs.append(idx)

    train_norm_prems_init = {nr.normalize_for_dedup(r["premise"]) for r in train_rows}
    dev_pool_cands = [
        r for r in verified_rows
        if r["split_pool"] == "dev"
        and r["pair_id"] not in used_pair_ids
        and r["doc_id"] not in sealed_docs
        and r["cluster_id"] not in sealed_clusters
    ]

    dev_replacements: List[dict] = []
    for idx in dev_bad_idxs:
        old = dev_rows[idx]
        lbl = old["label"]
        chosen_raw = None
        # 同ドメイン・同長さ帯を優先し、なければ同ラベルの任意候補を使う
        ordered_cands = sorted(
            [c for c in dev_pool_cands if c["intended_label"] == lbl],
            key=lambda c: (
                0 if (c["domain_group"] == old["domain_group"] and c["len_bucket"] == old["len_bucket"]) else 1,
                0 if c["domain_group"] == old["domain_group"] else 1,
                c["pair_id"],
            ),
        )
        for cand in ordered_cands:
            if cand["pair_id"] in used_pair_ids:
                continue
            ok, _ = nr.check_pair_rules(cand["premise"], cand["hypothesis"], lbl)
            if not ok:
                continue
            np_c = nr.normalize_for_dedup(cand["premise"])
            nh_c = nr.normalize_for_dedup(cand["hypothesis"])
            if np_c in sealed_norm_prems or np_c in train_norm_prems_init or nh_c in used_norm_hyps:
                continue
            if (np_c, nh_c) in used_pair_norms:
                continue
            if (
                has_lsh_collision(cand["premise"], sealed_prem_lsh, sig_cache, 0.80)
                or has_lsh_collision(cand["hypothesis"], sealed_hyp_lsh, sig_cache, 0.85)
                or has_lsh_collision(cand["premise"], audit_prem_lsh, sig_cache, 0.80)
                or has_lsh_collision(cand["hypothesis"], audit_hyp_lsh, sig_cache, 0.85)
            ):
                continue
            chosen_raw = cand
            break
        if chosen_raw is None:
            raise RuntimeError(f"Insufficient replacement candidates in dev pool for idx={idx}, label={lbl}")

        formatted = bpd.format_pilot_row(chosen_raw, idx, "dev")
        formatted["id"] = old["id"]
        dev_rows[idx] = formatted
        used_pair_ids.add(chosen_raw["pair_id"])
        used_norm_hyps.add(nr.normalize_for_dedup(chosen_raw["hypothesis"]))
        used_pair_norms.add((nr.normalize_for_dedup(chosen_raw["premise"]), nr.normalize_for_dedup(chosen_raw["hypothesis"])))
        dev_replacements.append({
            "id": old["id"],
            "label": lbl,
            "old_pair_id": old["pair_id"],
            "new_pair_id": chosen_raw["pair_id"],
        })

    # --- Phase 2: train の LSH 衝突修正 (against sealed-test ∪ updated dev) ---
    dev_prem_lsh = build_ref_lsh([r["premise"] for r in dev_rows], sig_cache)
    dev_hyp_lsh = build_ref_lsh([r["hypothesis"] for r in dev_rows], sig_cache)
    dev_norm_prems = {nr.normalize_for_dedup(r["premise"]) for r in dev_rows}

    train_bad_idxs: List[int] = []
    for idx, r in enumerate(train_rows):
        p, h = r["premise"], r["hypothesis"]
        if (
            has_lsh_collision(p, sealed_prem_lsh, sig_cache, 0.80)
            or has_lsh_collision(h, sealed_hyp_lsh, sig_cache, 0.85)
            or has_lsh_collision(p, dev_prem_lsh, sig_cache, 0.80)
            or has_lsh_collision(h, dev_hyp_lsh, sig_cache, 0.85)
        ):
            train_bad_idxs.append(idx)

    colliding_train_pids = {train_rows[i]["pair_id"] for i in train_bad_idxs}
    overlap_with_audit = colliding_train_pids & audit_pair_ids
    assert len(overlap_with_audit) == 0, f"Audit 200 overlap detected in colliding train rows: {overlap_with_audit}"

    train_pool_cands = [
        r for r in verified_rows
        if r["split_pool"] == "train"
        and r["pair_id"] not in used_pair_ids
        and r["doc_id"] not in sealed_docs
        and r["cluster_id"] not in sealed_clusters
    ]

    train_replacements: List[dict] = []
    for idx in train_bad_idxs:
        old = train_rows[idx]
        lbl = old["label"]
        chosen_raw = None
        ordered_cands = sorted(
            [c for c in train_pool_cands if c["intended_label"] == lbl],
            key=lambda c: (
                0 if (c["domain_group"] == old["domain_group"] and c["len_bucket"] == old["len_bucket"]) else 1,
                0 if c["domain_group"] == old["domain_group"] else 1,
                c["pair_id"],
            ),
        )
        for cand in ordered_cands:
            if cand["pair_id"] in used_pair_ids:
                continue
            ok, _ = nr.check_pair_rules(cand["premise"], cand["hypothesis"], lbl)
            if not ok:
                continue
            np_c = nr.normalize_for_dedup(cand["premise"])
            nh_c = nr.normalize_for_dedup(cand["hypothesis"])
            if np_c in sealed_norm_prems or np_c in dev_norm_prems or nh_c in used_norm_hyps:
                continue
            if (np_c, nh_c) in used_pair_norms:
                continue
            if (
                has_lsh_collision(cand["premise"], sealed_prem_lsh, sig_cache, 0.80)
                or has_lsh_collision(cand["hypothesis"], sealed_hyp_lsh, sig_cache, 0.85)
                or has_lsh_collision(cand["premise"], dev_prem_lsh, sig_cache, 0.80)
                or has_lsh_collision(cand["hypothesis"], dev_hyp_lsh, sig_cache, 0.85)
            ):
                continue
            chosen_raw = cand
            break
        if chosen_raw is None:
            raise RuntimeError(f"Insufficient replacement candidates in train pool for idx={idx}, label={lbl}")

        formatted = bpd.format_pilot_row(chosen_raw, idx, "train")
        formatted["id"] = old["id"]
        train_rows[idx] = formatted
        used_pair_ids.add(chosen_raw["pair_id"])
        used_norm_hyps.add(nr.normalize_for_dedup(chosen_raw["hypothesis"]))
        used_pair_norms.add((nr.normalize_for_dedup(chosen_raw["premise"]), nr.normalize_for_dedup(chosen_raw["hypothesis"])))
        train_replacements.append({
            "id": old["id"],
            "label": lbl,
            "old_pair_id": old["pair_id"],
            "new_pair_id": chosen_raw["pair_id"],
        })

    # 保存と検証
    save_jsonl(dev_path, dev_rows)
    save_jsonl(train_path, train_rows)

    sealed_sha_after = sha256_file(sealed_path)
    assert sealed_sha_after == EXPECTED_SEALED_SHA256, "sealed-test.jsonl was altered unexpectedly!"

    # 3分割キャッシュ済み全ペア検査
    prof_tr = cla.prepare_split_profile(train_rows, sig_cache)
    prof_dv = cla.prepare_split_profile(dev_rows, sig_cache)
    prof_st = cla.prepare_split_profile(sealed_rows, sig_cache)

    leaks = {
        "train__dev": cla.inspect_pair(prof_tr, prof_dv),
        "train__sealed-test": cla.inspect_pair(prof_tr, prof_st),
        "dev__sealed-test": cla.inspect_pair(prof_dv, prof_st),
    }
    for pair_name, metrics in leaks.items():
        for k, v in metrics.items():
            assert v == 0, f"Leak still non-zero after repair: {pair_name}.{k}={v}"

    # hypothesis_only_diagnostic 再計算 & build-stats.json 更新
    hyp_diag = bpd.hypothesis_only_diagnostic(train_rows, dev_rows)
    updated_stats = {
        "created_at": prev_stats.get("created_at"),
        "repaired_at": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "repair_elapsed_sec": round(time.time() - t0, 2),
        "seed": prev_stats.get("seed", 42),
        "input_verified_rows": prev_stats.get("input_verified_rows"),
        "dedup_drop_counts": prev_stats.get("dedup_drop_counts"),
        "available_by_pool_class": prev_stats.get("available_by_pool_class"),
        "splits": {
            "train": bpd.compute_split_stats(train_rows),
            "dev": bpd.compute_split_stats(dev_rows),
            "sealed-test": prev_stats["splits"]["sealed-test"],
        },
        "file_sha256": {
            "train": sha256_file(train_path),
            "dev": sha256_file(dev_path),
            "sealed-test": sealed_sha_after,
        },
        "hypothesis_only_diagnostic": hyp_diag,
        "leaks": leaks,
        "lsh_repair_summary": {
            "dev_replaced": len(dev_replacements),
            "train_replaced": len(train_replacements),
            "dev_replaced_ids": [x["id"] for x in dev_replacements],
            "train_replaced_by_label": dict(Counter(x["label"] for x in train_replacements)),
            "audit_200_affected": 0,
            "sealed_test_untouched": True,
        },
        "sealed_test_policy": "sealed-test.jsonl constructed, audited for counts/leaks only, and sealed without opening or evaluation.",
    }
    stats_path.write_text(json.dumps(updated_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # human_eval_candidates 780件に対する LSH 衝突インデックス抽出
    he_cands_path = Path(args.he_cands)
    he_hit_idxs: List[int] = []
    if he_cands_path.exists():
        he_cands = load_jsonl(he_cands_path)
        tr_prem_lsh = build_ref_lsh([r["premise"] for r in train_rows], sig_cache)
        tr_hyp_lsh = build_ref_lsh([r["hypothesis"] for r in train_rows], sig_cache)
        for r in he_cands:
            p, h = r["premise"], r["hypothesis"]
            if (
                has_lsh_collision(p, sealed_prem_lsh, sig_cache, 0.80)
                or has_lsh_collision(h, sealed_hyp_lsh, sig_cache, 0.85)
                or has_lsh_collision(p, dev_prem_lsh, sig_cache, 0.80)
                or has_lsh_collision(h, dev_hyp_lsh, sig_cache, 0.85)
                or has_lsh_collision(p, tr_prem_lsh, sig_cache, 0.80)
                or has_lsh_collision(h, tr_hyp_lsh, sig_cache, 0.85)
            ):
                he_hit_idxs.append(r["cand_idx"])

    print(f"LSH repair finished in {updated_stats['repair_elapsed_sec']}s")
    print(f"  dev_replaced={len(dev_replacements)} ({[x['id'] for x in dev_replacements]})")
    print(f"  train_replaced={len(train_replacements)} ({dict(Counter(x['label'] for x in train_replacements))})")
    print(f"  audit_200_affected=0, sealed_test_sha256_verified={sealed_sha_after}")
    print(f"  3-split leaks={leaks}")
    print(f"  hypothesis_only_diagnostic={hyp_diag}")
    print(f"  human_eval_candidates LSH hits ({len(he_hit_idxs)}): {he_hit_idxs}")


if __name__ == "__main__":
    main()
