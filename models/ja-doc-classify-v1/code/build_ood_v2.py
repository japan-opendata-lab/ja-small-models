#!/usr/bin/env python3
"""第4号 ja-doc-classify-v1 OOD v2（全43種・200件）統合・三重非重複機械監査スクリプト。

5 つのスタイル別ファイル（business_email / chat_memo / ocr_form / narrative_long / portal_auto）を統合し、
OOD v1・Nemotron (train+dev)・OOD v2 内部との完全非重複を文字 5-gram で厳密検証して
data/ood_v2/ood-v2-200.jsonl、ood-v2-blind-shuffled.jsonl、ood-v2-audit-stats.json を出力する。
※ sealed-test.jsonl には一切アクセスしない。
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

META_PHRASES = [
    "以下は",
    "架空の",
    "サンプルとして",
    "ダミーデータ",
    "例として",
    "テスト用データ",
]


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


def max_jaccard_against(v2_rows, v2_sh, other_paths, thr):
    """転置索引を用いて other_paths 全文書との厳密 5-gram Jaccard 最大値と閾値超え数を算出。"""
    v2_texts = {r["text"] for r in v2_rows}
    inv = defaultdict(list)
    v2_lens = [len(s) for s in v2_sh]
    for idx, s in enumerate(v2_sh):
        for sh in s:
            inv[sh].append(idx)

    checked = 0
    exact = 0
    ge_thr = 0
    max_j = 0.0
    max_pair = None
    for p in other_paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                checked += 1
                obj = json.loads(line)
                txt = obj["text"]
                if txt in v2_texts:
                    exact += 1
                osh = shingles(txt)
                olen = len(osh)
                if not olen:
                    continue
                ov = Counter()
                for sh in osh:
                    if sh in inv:
                        for j in inv[sh]:
                            ov[j] += 1
                for j, c in ov.items():
                    jacc = c / (olen + v2_lens[j] - c)
                    if jacc >= thr:
                        ge_thr += 1
                    if jacc > max_j:
                        max_j = jacc
                        max_pair = {"v2_id": v2_rows[j]["id"], "other_src": p.name, "other_id": obj.get("id")}
    return checked, exact, ge_thr, round(max_j, 4), max_pair


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", default="data/ood_v2")
    ap.add_argument("--v1-path", default="data/ood/ood-200.jsonl")
    ap.add_argument("--train", default="data/pilot/train.jsonl")
    ap.add_argument("--dev", default="data/pilot/dev.jsonl")
    args = ap.parse_args()

    v2_dir = Path(args.v2_dir)
    assigns = {a["id"]: a for a in json.loads((v2_dir / "v2_assignment.json").read_text(encoding="utf-8"))}
    styles = ["business_email", "chat_memo", "ocr_form", "narrative_long", "portal_auto"]

    rows = []
    seen_ids = set()
    for st in styles:
        for r in load_jsonl(v2_dir / f"part_{st}.jsonl"):
            rid = r["id"]
            assert rid in assigns and rid not in seen_ids, f"Bad or dup id: {rid}"
            seen_ids.add(rid)
            a = assigns[rid]
            assert r["document_type"] == a["document_type"], f"{rid} type mismatch"
            assert r["domain"] == a["domain"], f"{rid} domain mismatch"
            assert r["style"] == st, f"{rid} style mismatch"
            if not a["allow_exact_name"]:
                assert a["document_type"] not in r["text"], f"{rid} leaked exact name"
            rows.append({
                "id": rid,
                "text": r["text"],
                "document_type": a["document_type"],
                "domain": a["domain"],
                "risk_tier": a["risk_tier"],
                "style": st,
                "writer_id": r["writer_id"],
            })

    assert len(rows) == 200, f"Expected 200 rows, got {len(rows)}"
    ty_cnt = Counter(r["document_type"] for r in rows)
    assert len(ty_cnt) == 43 and min(ty_cnt.values()) == 4 and max(ty_cnt.values()) == 5

    out_full = v2_dir / "ood-v2-200.jsonl"
    with open(out_full, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ランダムシャッフル匿名盲検ファイル生成
    rng = random.Random(20261120)
    order = list(range(len(rows)))
    rng.shuffle(order)
    blind_map = {}
    with open(v2_dir / "ood-v2-blind-shuffled.jsonl", "w", encoding="utf-8", newline="\n") as fb:
        for idx, oi in enumerate(order, 1):
            bid = f"v2b{idx:03d}"
            blind_map[bid] = {"orig_id": rows[oi]["id"], "gold": rows[oi]["document_type"], "style": rows[oi]["style"]}
            fb.write(json.dumps({"id": bid, "text": rows[oi]["text"]}, ensure_ascii=False) + "\n")
    (v2_dir / "ood-v2-blind-map.json").write_text(
        json.dumps(blind_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 監査統計
    meta_hits = []
    exact_cnt = 0
    exact_1st = 0
    exact_by_style = Counter()
    by_style_lens = defaultdict(list)
    for r in rows:
        txt = r["text"]
        by_style_lens[r["style"]].append(len(txt))
        for ph in META_PHRASES:
            if ph in txt:
                meta_hits.append({"id": r["id"], "phrase": ph})
        if r["document_type"] in txt:
            exact_cnt += 1
            exact_by_style[r["style"]] += 1
        fl = txt.splitlines()[0] if txt.splitlines() else ""
        if r["document_type"] in fl:
            exact_1st += 1

    v2_sh = [shingles(r["text"]) for r in rows]
    internal_exact = len(rows) - len({r["text"] for r in rows})
    internal_ge05 = 0
    internal_max_j = 0.0
    internal_max_pair = None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            u = len(v2_sh[i] | v2_sh[j])
            jc = (len(v2_sh[i] & v2_sh[j]) / u) if u else 0.0
            if jc >= 0.50:
                internal_ge05 += 1
            if jc > internal_max_j:
                internal_max_j = jc
                internal_max_pair = [rows[i]["id"], rows[j]["id"]]

    v1_chk, v1_ex, v1_ge04, v1_maxj, v1_pair = max_jaccard_against(
        rows, v2_sh, [Path(args.v1_path)], thr=0.40
    )
    nem_chk, nem_ex, nem_ge05, nem_maxj, nem_pair = max_jaccard_against(
        rows, v2_sh, [Path(args.train), Path(args.dev)], thr=0.50
    )

    sha256 = hashlib.sha256(out_full.read_bytes()).hexdigest()
    audit = {
        "total_rows": len(rows),
        "covered_types": len(ty_cnt),
        "type_counts_min": min(ty_cnt.values()),
        "type_counts_max": max(ty_cnt.values()),
        "style_counts": dict(Counter(r["style"] for r in rows)),
        "domain_counts": dict(Counter(r["domain"] for r in rows)),
        "risk_tier_counts": dict(Counter(r["risk_tier"] for r in rows)),
        "char_len_all": text_stats([len(r["text"]) for r in rows]),
        "char_len_by_style": {k: text_stats(v) for k, v in sorted(by_style_lens.items())},
        "lt_80_chars": sum(1 for r in rows if len(r["text"]) < 80),
        "meta_phrase_hits": meta_hits,
        "exact_name_in_text_count": exact_cnt,
        "exact_name_in_text_rate": round(exact_cnt / len(rows), 4),
        "exact_name_first_line_count": exact_1st,
        "exact_name_by_style": dict(exact_by_style),
        "internal_exact_dups": internal_exact,
        "internal_ge_050": internal_ge05,
        "internal_max_jaccard": round(internal_max_j, 4),
        "internal_max_pair": internal_max_pair,
        "ood_v1_checked": v1_chk,
        "ood_v1_exact_dups": v1_ex,
        "ood_v1_ge_040": v1_ge04,
        "ood_v1_max_jaccard": v1_maxj,
        "ood_v1_max_pair": v1_pair,
        "nemotron_train_dev_checked": nem_chk,
        "nemotron_exact_dups": nem_ex,
        "nemotron_ge_050": nem_ge05,
        "nemotron_max_jaccard": nem_maxj,
        "nemotron_max_pair": nem_pair,
        "ood_v2_200_sha256": sha256,
        "ood_v2_200_bytes": out_full.stat().st_size,
    }
    (v2_dir / "ood-v2-audit-stats.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
