#!/usr/bin/env python3
"""第4号 ja-doc-classify-v1 OOD 評価セット統合・機械監査スクリプト。

4 つの文体別 JSONL（part_formal / part_bullet_mixed / part_email / part_short_memo）を
統合し、ラベル・件数・文体バランス・文字長・メタ発言・内部重複・Nemotron (train+dev) 独立性
を検証した上で data/ood/ood-200.jsonl、ood-200-blind.jsonl、ood-audit-stats.json を出力する。
※ sealed-test.jsonl には一切アクセスしない。
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

META_PHRASES = [
    "以下は",
    "架空の",
    "サンプルとして",
    "作成しました",
    "ダミーデータ",
    "例として",
    "テスト用データ",
]

EXPECTED_TYPES = {
    "クレジットカード利用明細": "金融",
    "振込依頼書": "金融",
    "海外送金依頼書": "金融",
    "業務委託契約書": "法務",
    "国際業務委託契約書": "法務",
    "委任状": "法務",
    "処方箋": "医療",
    "診断書": "医療",
    "検査結果報告書": "医療",
    "診療情報提供書": "医療",
    "成績通知書": "教育",
    "奨学金申請書": "教育",
    "保護者連絡票": "教育",
    "注文確認書": "EC",
    "配送伝票": "EC",
    "返金申請書": "EC",
    "雇用契約書": "雇用",
    "給与連絡票": "雇用",
    "退職手続書類": "雇用",
    "セキュリティインシデント報告書": "通信・IT",
    "パスワード再設定案内": "通信・IT",
    "アクセス申請書": "通信・IT",
    "住民異動届": "行政",
    "給付申請書": "行政",
    "税務申告関連書類": "行政",
}


def shingles(text: str, n: int = 5) -> set:
    return {text[i:i + n] for i in range(max(0, len(text) - n + 1))}


def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def text_stats(lengths: list) -> dict:
    arr = np.array(lengths, dtype=int)
    return {
        "count": int(len(arr)),
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "mean": round(float(arr.mean()), 2),
        "p90": float(np.percentile(arr, 90)),
        "max": int(arr.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-dir", default="data/ood")
    ap.add_argument("--risk-csv", default="risk_tiers_v0.csv")
    ap.add_argument("--train", default="data/pilot/train.jsonl")
    ap.add_argument("--dev", default="data/pilot/dev.jsonl")
    args = ap.parse_args()

    ood_dir = Path(args.ood_dir)
    part_files = [
        ood_dir / "part_formal.jsonl",
        ood_dir / "part_bullet_mixed.jsonl",
        ood_dir / "part_email.jsonl",
        ood_dir / "part_short_memo.jsonl",
    ]
    tiers = {}
    with open(args.risk_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tiers[row["document_type"]] = row["risk_tier"]

    rows = []
    seen_ids = set()
    for pf in part_files:
        for r in load_jsonl(pf):
            rid = r["id"]
            assert rid not in seen_ids, f"Duplicate id: {rid}"
            seen_ids.add(rid)
            ty = r["document_type"]
            assert ty in EXPECTED_TYPES, f"Unexpected type: {ty}"
            assert r["domain"] == EXPECTED_TYPES[ty], (
                f"Domain mismatch for {rid}: {r['domain']} != {EXPECTED_TYPES[ty]}"
            )
            rows.append({
                "id": rid,
                "text": r["text"],
                "document_type": ty,
                "domain": r["domain"],
                "risk_tier": tiers[ty],
                "style": r["style"],
                "writer_id": r["writer_id"],
            })

    assert len(rows) == 200, f"Expected 200 rows, got {len(rows)}"
    type_counts = Counter(r["document_type"] for r in rows)
    assert len(type_counts) == 25 and all(v == 8 for v in type_counts.values()), dict(type_counts)
    style_counts = Counter(r["style"] for r in rows)
    assert len(style_counts) == 4 and all(v == 50 for v in style_counts.values()), dict(style_counts)

    # 書き出し（ood-200.jsonl と 盲検用 ood-200-blind.jsonl）
    out_full = ood_dir / "ood-200.jsonl"
    out_blind = ood_dir / "ood-200-blind.jsonl"
    with open(out_full, "w", encoding="utf-8") as f1, open(out_blind, "w", encoding="utf-8") as f2:
        for r in rows:
            f1.write(json.dumps(r, ensure_ascii=False) + "\n")
            f2.write(json.dumps({"id": r["id"], "text": r["text"]}, ensure_ascii=False) + "\n")

    # 文字長・メタ発言・種別名含有率
    meta_hits = []
    exact_name_in_text = 0
    exact_name_first_line = 0
    by_style_lens = defaultdict(list)
    by_style_exact = Counter()
    for r in rows:
        txt = r["text"]
        by_style_lens[r["style"]].append(len(txt))
        for ph in META_PHRASES:
            if ph in txt:
                meta_hits.append({"id": r["id"], "phrase": ph})
        if r["document_type"] in txt:
            exact_name_in_text += 1
            by_style_exact[r["style"]] += 1
        first_line = txt.splitlines()[0] if txt.splitlines() else ""
        if r["document_type"] in first_line:
            exact_name_first_line += 1

    # OOD 内部ペア非重複検査（5-gram Jaccard）
    ood_sh = [shingles(r["text"]) for r in rows]
    ood_exact_dups = len(rows) - len({r["text"] for r in rows})
    max_internal_jacc = 0.0
    max_internal_pair = None
    internal_ge_05 = 0
    for i in range(len(rows)):
        si = ood_sh[i]
        for j in range(i + 1, len(rows)):
            sj = ood_sh[j]
            u = len(si | sj)
            jacc = (len(si & sj) / u) if u else 0.0
            if jacc >= 0.50:
                internal_ge_05 += 1
            if jacc > max_internal_jacc:
                max_internal_jacc = jacc
                max_internal_pair = [rows[i]["id"], rows[j]["id"]]

    # Nemotron (train + dev のみ。sealed-test は絶対に開かない) との独立性検査
    ood_text_set = {r["text"] for r in rows}
    inv_idx = defaultdict(list)
    ood_sh_lens = [len(s) for s in ood_sh]
    for idx, s in enumerate(ood_sh):
        for sh in s:
            inv_idx[sh].append(idx)

    nem_exact_dups = 0
    nem_ge_05 = 0
    max_nem_jacc = 0.0
    max_nem_info = None
    nem_checked = 0
    for src_path in [Path(args.train), Path(args.dev)]:
        with open(src_path, encoding="utf-8") as f:
            for line in f:
                nem_checked += 1
                txt = json.loads(line)["text"]
                if txt in ood_text_set:
                    nem_exact_dups += 1
                nsh = shingles(txt)
                nlen = len(nsh)
                if not nlen:
                    continue
                overlap = Counter()
                for sh in nsh:
                    if sh in inv_idx:
                        for j in inv_idx[sh]:
                            overlap[j] += 1
                for j, c in overlap.items():
                    jacc = c / (nlen + ood_sh_lens[j] - c)
                    if jacc >= 0.50:
                        nem_ge_05 += 1
                    if jacc > max_nem_jacc:
                        max_nem_jacc = jacc
                        max_nem_info = {"ood_id": rows[j]["id"], "src": src_path.name}

    sha256 = hashlib.sha256(out_full.read_bytes()).hexdigest()
    audit = {
        "total_rows": len(rows),
        "covered_types": len(type_counts),
        "type_counts_min": min(type_counts.values()),
        "type_counts_max": max(type_counts.values()),
        "style_counts": dict(style_counts),
        "domain_counts": dict(Counter(r["domain"] for r in rows)),
        "risk_tier_counts": dict(Counter(r["risk_tier"] for r in rows)),
        "char_len_all": text_stats([len(r["text"]) for r in rows]),
        "char_len_by_style": {k: text_stats(v) for k, v in sorted(by_style_lens.items())},
        "lt_80_chars": sum(1 for r in rows if len(r["text"]) < 80),
        "meta_phrase_hits": meta_hits,
        "exact_name_in_text_count": exact_name_in_text,
        "exact_name_in_text_rate": round(exact_name_in_text / len(rows), 4),
        "exact_name_first_line_count": exact_name_first_line,
        "exact_name_by_style": dict(by_style_exact),
        "ood_internal_exact_dups": ood_exact_dups,
        "ood_internal_ge_050": internal_ge_05,
        "ood_internal_max_jaccard": round(max_internal_jacc, 4),
        "ood_internal_max_pair": max_internal_pair,
        "nemotron_train_dev_checked": nem_checked,
        "nemotron_exact_dups": nem_exact_dups,
        "nemotron_ge_050": nem_ge_05,
        "nemotron_max_jaccard": round(max_nem_jacc, 4),
        "nemotron_max_pair": max_nem_info,
        "ood_200_sha256": sha256,
    }
    (ood_dir / "ood-audit-stats.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
