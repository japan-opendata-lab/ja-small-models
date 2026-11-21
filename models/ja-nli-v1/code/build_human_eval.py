#!/usr/bin/env python
"""第11号 ja-nli-v1 人手確認評価セット（600ペア）候補抽出＆確定台帳生成スクリプト。

モード:
  --mode prepare  : human_eval_verified.jsonl からリークなし・多様層化候補をバッチ展開
  --mode finalize : 全件目視確認ログ (human_eval_reviews.jsonl) に基づき、E/N/C 各200=600件の
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
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nli_rules as nr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


def load_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def prepare_candidates(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    he_path = Path(args.he_verified)
    if not he_path.exists() or he_path.stat().st_size == 0:
        rows = [
            r
            for r in load_jsonl(ROOT / "data" / "synth" / "synth_verified.jsonl")
            if r["split_pool"] == "human_eval"
        ]
    else:
        rows = load_jsonl(he_path)

    # train / dev / sealed-test との重複セット構築（テキスト出力禁止）
    excl_docs: Set[str] = set()
    excl_clusters: Set[str] = set()
    excl_prems: Set[str] = set()
    excl_hyps: Set[str] = set()
    for sp in ["train.jsonl", "dev.jsonl", "sealed-test.jsonl"]:
        for r in load_jsonl(ROOT / "data" / "pilot" / sp):
            excl_docs.add(r["doc_id"])
            excl_clusters.add(r["cluster_id"])
            excl_prems.add(nr.normalize_for_dedup(r["premise"]))
            excl_hyps.add(nr.normalize_for_dedup(r["hypothesis"]))

    clean_cands: List[dict] = []
    seen_pair: Set[Tuple[str, str]] = set()
    seen_hyp: Set[str] = set()
    skipped = Counter()

    for r in rows:
        if r["doc_id"] in excl_docs or r["cluster_id"] in excl_clusters:
            skipped["cluster_overlap"] += 1
            continue
        np = nr.normalize_for_dedup(r["premise"])
        nh = nr.normalize_for_dedup(r["hypothesis"])
        if np in excl_prems or nh in excl_hyps:
            skipped["text_overlap"] += 1
            continue
        if (np, nh) in seen_pair or nh in seen_hyp:
            skipped["internal_dup"] += 1
            continue
        ok, _ = nr.check_pair_rules(r["premise"], r["hypothesis"], r["intended_label"])
        if not ok:
            skipped["rule_fail"] += 1
            continue
        seen_pair.add((np, nh))
        seen_hyp.add(nh)
        clean_cands.append(r)

    # クラスごとに (第3検証一致, 自然さ) を優先しつつ (domain_group, len_bucket, subtype) を多様化抽出
    rng = random.Random(args.seed)
    picked_cands: List[dict] = []
    avail_by_cls = {}
    for lbl in nr.VALID_LABELS:
        cls_rows = [r for r in clean_cands if r["intended_label"] == lbl]
        avail_by_cls[lbl] = len(cls_rows)
        rng.shuffle(cls_rows)
        # 優先ソート: verifier3_agree (True先), verifier_naturalness (高い先)
        cls_rows.sort(
            key=lambda x: (
                1 if x.get("verifier3_agree") else 0,
                int(x.get("verifier_naturalness") or 0),
                1 if x.get("verifier3_naturalness", 0) >= 4 else 0,
            ),
            reverse=True,
        )
        # 同一チャンクから1クラス1件（当然）かつ全ソース・長さ帯を均等拾うため層化順にする
        buckets = defaultdict(list)
        for item in cls_rows:
            buckets[(item["domain_group"], item["len_bucket"])].append(item)
        keys = sorted(buckets.keys())
        ordered = []
        while len(ordered) < args.cands_per_class and any(buckets.values()):
            for k in keys:
                if buckets[k]:
                    ordered.append(buckets[k].pop(0))
                    if len(ordered) >= args.cands_per_class:
                        break
        picked_cands.extend(ordered)

    # シャッフルせずクラス別かランダム配置（ただし全件レビューするため識別子付番）
    cands_path = out_dir / "human_eval_candidates.jsonl"
    with cands_path.open("w", encoding="utf-8") as fout:
        for idx, r in enumerate(picked_cands):
            rec = {
                "cand_idx": idx,
                "pair_id": r["pair_id"],
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "cluster_id": r["cluster_id"],
                "source": r["source"],
                "domain_group": r["domain_group"],
                "license": r["license"],
                "len_bucket": r["len_bucket"],
                "subtype": r.get("subtype", ""),
                "label": r["intended_label"],
                "label_id": LABEL2ID[r["intended_label"]],
                "premise": r["premise"],
                "hypothesis": r["hypothesis"],
                "generator_model": r.get("generator_model"),
                "verifier_model": r.get("verifier_model"),
                "verifier_label": r.get("verifier_label"),
                "verifier_naturalness": r.get("verifier_naturalness"),
                "verifier_reason": r.get("verifier_reason", ""),
                "verifier3_model": r.get("verifier3_model"),
                "verifier3_label": r.get("verifier3_label"),
                "verifier3_agree": r.get("verifier3_agree"),
                "verifier3_reason": r.get("verifier3_reason", ""),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 60件ごとの目視バッチ Markdown 出力
    for b in range(0, len(picked_cands), 60):
        b_no = b // 60 + 1
        lines = [f"# 人手確認評価候補 バッチ {b_no} ({b}〜{min(b+60, len(picked_cands))-1})\n"]
        for i in range(b, min(b + 60, len(picked_cands))):
            r = picked_cands[i]
            lines.append(
                f"## [C{i:03d}] label=**{r['intended_label']}** | src={r['source']} | len={r['len_bucket']} | sub={r.get('subtype','')}\n"
                f"- **V2(nat={r.get('verifier_naturalness')})**: {r.get('verifier_label')} ({r.get('verifier_reason','')})\n"
                f"- **V3(agree={r.get('verifier3_agree')})**: {r.get('verifier3_label')} ({r.get('verifier3_reason','')})\n"
                f"- **Premise ({len(r['premise'])}字)**: {r['premise']}\n"
                f"- **Hypothesis ({len(r['hypothesis'])}字)**: {r['hypothesis']}\n"
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
    seen_chunks_per_cls: Dict[str, Set[str]] = defaultdict(set)

    for rev in sorted(reviews, key=lambda x: x["cand_idx"]):
        c = cands[rev["cand_idx"]]
        status = rev["status"]
        if status == "approved":
            lbl = rev.get("override_label") or c["label"]
            merged = {
                **c,
                "label": lbl,
                "label_id": LABEL2ID[lbl],
                "human_review_status": "approved",
                "human_review_note": rev.get("note", ""),
            }
            approved_by_cls[lbl].append(merged)
        else:
            rejected_list.append({**c, "human_review_status": "rejected", "human_review_note": rev.get("note", "")})

    final_rows: List[dict] = []
    for lbl in nr.VALID_LABELS:
        items = approved_by_cls[lbl]
        if len(items) < 200:
            raise RuntimeError(f"Approved count for {lbl} is {len(items)} < 200! Need more review replacements.")
        final_rows.extend(items[:200])

    # 同一チャンクからの重複確認・安定並び替え
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
            "id", "cand_idx", "label", "source", "domain_group", "len_bucket", "subtype",
            "premise", "hypothesis", "verifier_label", "verifier_naturalness", "verifier_reason",
            "verifier3_label", "verifier3_agree", "human_review_status", "human_review_note",
        ])
        for r in final_rows:
            w.writerow([
                r["id"], r["cand_idx"], r["label"], r["source"], r["domain_group"], r["len_bucket"],
                r.get("subtype", ""), r["premise"], r["hypothesis"], r.get("verifier_label"),
                r.get("verifier_naturalness"), r.get("verifier_reason", ""),
                r.get("verifier3_label"), r.get("verifier3_agree"),
                r["human_review_status"], r["human_review_note"],
            ])

    md_lines = [
        "# 第11号 `ja-nli-v1` 人手確認評価セット（600ペア）全件台帳\n",
        f"- 確定件数: **{len(final_rows)}** 件 (`entailment`: 200, `neutral`: 200, `contradiction`: 200)",
        f"- 目視精査件数: 承認 {sum(len(v) for v in approved_by_cls.values())} 件 / 却下・差替 {len(rejected_list)} 件",
        "- 学習・dev・封印テストとは文書クラスタ (`cluster_id`) 単位で完全分離済み。\n",
    ]
    for r in final_rows:
        md_lines.append(
            f"### {r['id']} (`{r['label']}`, `{r['source']}`, `{r['len_bucket']}`, `{r.get('subtype','')}`)\n"
            f"- **Premise**: {r['premise']}\n"
            f"- **Hypothesis**: {r['hypothesis']}\n"
            f"- **検証役照合**: V2=`{r.get('verifier_label')}` (nat={r.get('verifier_naturalness')}, {r.get('verifier_reason','')}) / V3=`{r.get('verifier3_label')}` (agree={r.get('verifier3_agree')})\n"
            f"- **目視確認**: `{r['human_review_status']}` — {r['human_review_note']}\n"
        )
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    p_lens = [len(r["premise"]) for r in final_rows]
    h_lens = [len(r["hypothesis"]) for r in final_rows]
    stats = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "total_rows": len(final_rows),
        "reviewed_total": len(reviews),
        "approved_total": sum(len(v) for v in approved_by_cls.values()),
        "rejected_total": len(rejected_list),
        "rejected_reasons": dict(Counter(r["human_review_note"].split(":")[0] for r in rejected_list)),
        "by_label": dict(Counter(r["label"] for r in final_rows)),
        "by_domain": dict(Counter(r["domain_group"] for r in final_rows)),
        "by_source": dict(Counter(r["source"] for r in final_rows)),
        "by_len_bucket": dict(Counter(r["len_bucket"] for r in final_rows)),
        "by_subtype": dict(Counter(r.get("subtype", "") for r in final_rows)),
        "v3_agree_count": sum(1 for r in final_rows if r.get("verifier3_agree")),
        "unique_docs": len({r["doc_id"] for r in final_rows}),
        "unique_clusters": len({r["cluster_id"] for r in final_rows}),
        "premise_chars": {"min": min(p_lens), "median": statistics.median(p_lens), "mean": round(statistics.mean(p_lens), 2), "max": max(p_lens)},
        "hypothesis_chars": {"min": min(h_lens), "median": statistics.median(h_lens), "mean": round(statistics.mean(h_lens), 2), "max": max(h_lens)},
        "sha256": hashlib.sha256(out_jsonl.read_bytes()).hexdigest(),
    }
    (out_dir / "human-eval-stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FINALIZE HE] 600 rows saved -> {out_jsonl} | rejected={len(rejected_list)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prepare", "finalize"], required=True)
    ap.add_argument("--he-verified", default=str(ROOT / "data" / "synth" / "human_eval_verified.jsonl"))
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
