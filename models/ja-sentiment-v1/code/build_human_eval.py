#!/usr/bin/env python3
"""第12号 ja-sentiment-v1 人手確認評価セット（600件）候補抽出＆確定台帳生成スクリプト。

モード:
  --mode prepare  : data/synth/he_verified.jsonl からリークなし・多様層化候補をバッチ展開
  --mode finalize : 全件目視確認ログ (human_eval_reviews.jsonl) に基づき、neg/neu/pos 各200=600件の
                    human-eval.jsonl / human-eval-review.md / human-eval-review.csv / human-eval-stats.json を生成
"""
import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sentiment_rules as sr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def prepare_candidates(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(Path(args.he_verified))
    assert rows, f"No rows in {args.he_verified}"

    # train / dev / sealed-test との重複セット構築（テキストは外部出力しない）
    excl_texts: List[str] = []
    excl_norms: Set[str] = set()
    excl_subs: Set[str] = set()
    for sp in ("train.jsonl", "dev.jsonl", "sealed-test.jsonl"):
        for r in load_jsonl(ROOT / "data" / "pilot" / sp):
            excl_texts.append(r["text"])
            excl_norms.add(sr.normalize_for_dedup(r["text"]))
            excl_subs.add(f"{r['domain']}::{r['subtopic']}")

    # LSH 高速フィルタ構築
    clean_cands: List[dict] = []
    seen_norm: Set[str] = set()
    skipped = Counter()

    for r in rows:
        sub_key = f"{r['domain']}::{r['subtopic']}"
        if sub_key in excl_subs:
            skipped["subtopic_overlap"] += 1
            continue
        txt = sr.clean_surface(r["text"])
        lbl = r["intended_label"]
        ok, _ = sr.check_text_rules(txt, lbl, r.get("domain", ""), r.get("length_band", ""))
        if not ok:
            skipped["rule_fail"] += 1
            continue
        nt = sr.normalize_for_dedup(txt)
        if nt in excl_norms:
            skipped["cross_split_norm_overlap"] += 1
            continue
        if nt in seen_norm:
            skipped["internal_dup"] += 1
            continue
        seen_norm.add(nt)
        clean_cands.append({**r, "text": txt})

    # LSH による pilot 分割との近傍重複除去
    if excl_texts and clean_cands:
        ngram_n, num_perm, bands = 5, 64, 16
        band_rows = num_perm // bands
        ex_grams = [sr.char_ngrams(sr.normalize_for_dedup(t) or t, n=ngram_n) for t in excl_texts]
        buckets: Dict[tuple, List[int]] = defaultdict(list)
        for idx, t in enumerate(excl_texts):
            sig = sr.compute_minhash_sig(t, ngram_n=ngram_n, num_perm=num_perm, seed=42)
            for b in range(bands):
                buckets[(b, sig[b * band_rows : (b + 1) * band_rows])].append(idx)
        filtered: List[dict] = []
        for r in clean_cands:
            t = r["text"]
            sg = sr.char_ngrams(sr.normalize_for_dedup(t) or t, n=ngram_n)
            sig = sr.compute_minhash_sig(t, ngram_n=ngram_n, num_perm=num_perm, seed=42)
            near = False
            cand_ids: Set[int] = set()
            for b in range(bands):
                cand_ids.update(buckets.get((b, sig[b * band_rows : (b + 1) * band_rows]), ()))
            for cid in cand_ids:
                eg = ex_grams[cid]
                if not eg:
                    continue
                inter = len(sg & eg)
                if inter / (len(sg) + len(eg) - inter) >= 0.70:
                    near = True
                    break
            if near:
                skipped["lsh_cross_overlap"] += 1
            else:
                filtered.append(r)
        clean_cands = filtered

    # 候補内部 LSH>=0.75 重複除去
    hits = sr.lsh_find_near_duplicates([r["text"] for r in clean_cands], ngram_n=5, jaccard_thresh=0.75)
    drop_idx = set()
    for u, v, _ in hits:
        if u not in drop_idx:
            drop_idx.add(v)
    if drop_idx:
        skipped["internal_lsh"] = len(drop_idx)
        clean_cands = [r for i, r in enumerate(clean_cands) if i not in drop_idx]

    # クラスごとに V1/V2 両一致 (both_verifiers_pass) と自然さを優先しつつ (domain, subtype) 多様層化
    rng = random.Random(args.seed)
    picked_cands: List[dict] = []
    avail_by_cls = {}
    for lbl in sr.VALID_LABELS:
        cls_rows = [r for r in clean_cands if r["intended_label"] == lbl]
        avail_by_cls[lbl] = len(cls_rows)
        rng.shuffle(cls_rows)
        cls_rows.sort(
            key=lambda x: (
                1 if x.get("both_verifiers_pass") else 0,
                1 if x.get("v1_agree") and x.get("v2_agree") else 0,
                int(x.get("v1_naturalness") or 0) + int(x.get("v2_naturalness") or 0),
            ),
            reverse=True,
        )
        buckets = defaultdict(list)
        for item in cls_rows:
            buckets[(item["domain"], item.get("subtype", ""))].append(item)
        keys = sorted(buckets.keys())
        ordered = []
        while len(ordered) < args.cands_per_class and any(buckets.values()):
            for k in keys:
                if buckets[k]:
                    ordered.append(buckets[k].pop(0))
                    if len(ordered) >= args.cands_per_class:
                        break
        picked_cands.extend(ordered)

    cands_path = out_dir / "human_eval_candidates.jsonl"
    with cands_path.open("w", encoding="utf-8") as fout:
        for idx, r in enumerate(picked_cands):
            rec = {
                "cand_idx": idx,
                "sample_id": r["sample_id"],
                "domain": r["domain"],
                "subtopic": r["subtopic"],
                "aspect": r.get("aspect", ""),
                "style": r.get("style", ""),
                "length_band": r.get("length_band", ""),
                "subtype": r.get("subtype", ""),
                "difficulty": r.get("difficulty", "standard"),
                "label": r["intended_label"],
                "label_id": sr.LABEL2ID[r["intended_label"]],
                "text": r["text"],
                "generator_model": r.get("generator_model"),
                "v1_model": r.get("v1_model"),
                "v1_label": r.get("v1_label"),
                "v1_naturalness": r.get("v1_naturalness"),
                "v1_ambiguity": r.get("v1_ambiguity"),
                "v1_reason": r.get("v1_reason", ""),
                "v2_model": r.get("v2_model"),
                "v2_label": r.get("v2_label"),
                "v2_naturalness": r.get("v2_naturalness"),
                "v2_ambiguity": r.get("v2_ambiguity"),
                "v2_reason": r.get("v2_reason", ""),
                "both_verifiers_pass": r.get("both_verifiers_pass", False),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for b in range(0, len(picked_cands), 60):
        b_no = b // 60 + 1
        lines = [f"# 人手確認評価候補 バッチ {b_no} ({b}〜{min(b+60, len(picked_cands))-1})\n"]
        for i in range(b, min(b + 60, len(picked_cands))):
            r = picked_cands[i]
            lines.append(
                f"## [C{i:03d}] label=**{r['intended_label']}** | dom={r['domain']} ({r['subtopic']}) | sub={r.get('subtype','')}\n"
                f"- **V1({r.get('v1_model','')}, nat={r.get('v1_naturalness')})**: {r.get('v1_label')} ({r.get('v1_reason','')})\n"
                f"- **V2({r.get('v2_model','')}, nat={r.get('v2_naturalness')})**: {r.get('v2_label')} ({r.get('v2_reason','')})\n"
                f"- **Text ({len(r['text'])}字)**: {r['text']}\n"
            )
        (out_dir / f"he_batch_{b_no:02d}.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        f"[PREPARE HE] clean_avail={avail_by_cls}, skipped={dict(skipped)}, "
        f"prepared_cands={len(picked_cands)} -> {cands_path}"
    )


def finalize_eval(args) -> None:
    out_dir = Path(args.out_dir)
    cands = {r["cand_idx"]: r for r in load_jsonl(out_dir / "human_eval_candidates.jsonl")}
    reviews = load_jsonl(out_dir / "human_eval_reviews.jsonl")
    assert reviews, "No human_eval_reviews.jsonl found!"

    approved_by_cls: Dict[str, List[dict]] = defaultdict(list)
    rejected_list: List[dict] = []

    for rev in sorted(reviews, key=lambda x: x["cand_idx"]):
        c = cands[rev["cand_idx"]]
        if rev["status"] == "approved":
            lbl = rev.get("override_label") or c["label"]
            merged = {
                **c,
                "label": lbl,
                "label_id": sr.LABEL2ID[lbl],
                "human_review_status": "approved",
                "human_review_note": rev.get("note", ""),
            }
            approved_by_cls[lbl].append(merged)
        else:
            rejected_list.append({
                **c,
                "human_review_status": "rejected",
                "human_review_note": rev.get("note", ""),
            })

    final_rows: List[dict] = []
    for lbl in sr.VALID_LABELS:
        items = approved_by_cls[lbl]
        if len(items) < 200:
            raise RuntimeError(f"Approved count for {lbl} is {len(items)} < 200!")
        final_rows.extend(items[:200])

    rng = random.Random(args.seed)
    rng.shuffle(final_rows)

    out_jsonl = out_dir / "human-eval.jsonl"
    out_csv = out_dir / "human-eval-review.csv"
    out_md = out_dir / "human-eval-review.md"

    with out_jsonl.open("w", encoding="utf-8") as fj:
        for i, r in enumerate(final_rows):
            r["id"] = f"he_{i:04d}"
            fj.write(json.dumps(r, ensure_ascii=False) + "\n")

    with out_csv.open("w", encoding="utf-8-sig", newline="") as fc:
        w = csv.writer(fc)
        w.writerow([
            "id", "cand_idx", "label", "domain", "subtopic", "style", "length_band", "subtype", "difficulty",
            "text", "v1_label", "v1_naturalness", "v1_reason", "v2_label", "v2_naturalness", "v2_reason",
            "human_review_status", "human_review_note",
        ])
        for r in final_rows:
            w.writerow([
                r["id"], r["cand_idx"], r["label"], r["domain"], r["subtopic"], r.get("style", ""),
                r.get("length_band", ""), r.get("subtype", ""), r.get("difficulty", ""), r["text"],
                r.get("v1_label"), r.get("v1_naturalness"), r.get("v1_reason", ""),
                r.get("v2_label"), r.get("v2_naturalness"), r.get("v2_reason", ""),
                r["human_review_status"], r["human_review_note"],
            ])

    md_lines = [
        "# 第12号 `ja-sentiment-v1` 人手確認評価セット（600件）全件台帳\n",
        f"- 確定件数: **{len(final_rows)}** 件 (`negative`: 200, `neutral`: 200, `positive`: 200)",
        f"- 目視精査件数: 承認 {sum(len(v) for v in approved_by_cls.values())} 件 / 却下・差替 {len(rejected_list)} 件",
        "- 学習・dev・封印テストとはモデルファミリー・サブトピック・生成プロンプト構造を完全分離済み。\n",
    ]
    for r in final_rows:
        md_lines.append(
            f"### {r['id']} (`{r['label']}`, `{r['domain']}`, `{r['subtopic']}`, `{r.get('subtype','')}`)\n"
            f"- **Text ({len(r['text'])}字)**: {r['text']}\n"
            f"- **検証役照合**: V1=`{r.get('v1_label')}` (nat={r.get('v1_naturalness')}, {r.get('v1_reason','')}) / V2=`{r.get('v2_label')}` (nat={r.get('v2_naturalness')}, {r.get('v2_reason','')})\n"
            f"- **目視確認**: `{r['human_review_status']}` — {r['human_review_note']}\n"
        )
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    lens = [len(r["text"]) for r in final_rows]
    stats = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "total_rows": len(final_rows),
        "reviewed_total": len(reviews),
        "approved_total": sum(len(v) for v in approved_by_cls.values()),
        "rejected_total": len(rejected_list),
        "rejected_reasons": dict(Counter(r["human_review_note"].split(":")[0] for r in rejected_list)),
        "by_label": dict(Counter(r["label"] for r in final_rows)),
        "by_domain": dict(Counter(r["domain"] for r in final_rows)),
        "by_difficulty": dict(Counter(r.get("difficulty", "standard") for r in final_rows)),
        "by_subtype": dict(Counter(r.get("subtype", "") for r in final_rows)),
        "by_length_band": dict(Counter(r.get("length_band", "") for r in final_rows)),
        "both_verifiers_pass_count": sum(1 for r in final_rows if r.get("both_verifiers_pass")),
        "unique_subtopics": len({f"{r['domain']}::{r['subtopic']}" for r in final_rows}),
        "text_chars": {"min": min(lens), "median": statistics.median(lens), "mean": round(statistics.mean(lens), 2), "max": max(lens)},
        "sha256": hashlib.sha256(out_jsonl.read_bytes()).hexdigest(),
    }
    (out_dir / "human-eval-stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FINALIZE HE] 600 rows saved -> {out_jsonl} | rejected={len(rejected_list)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prepare", "finalize"], required=True)
    ap.add_argument("--he-verified", default=str(ROOT / "data" / "synth" / "he_verified.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "eval"))
    ap.add_argument("--cands-per-class", type=int, default=260)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "prepare":
        prepare_candidates(args)
    else:
        finalize_eval(args)


if __name__ == "__main__":
    main()
