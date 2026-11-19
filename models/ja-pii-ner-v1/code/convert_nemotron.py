#!/usr/bin/env python3
"""Nemotron-PII Japan (parquet) -> v1 18ラベル JSONL 変換器。

変換ルールは label-mapping-v1.csv と ANNOTATION-GUIDE-v1.md が正。
- PERSON: first_name / last_name を隣接時に結合
- ADDRESS: street_address / prefecture / city を、文字位置が重なるか隣接 (gap<=1) すれば統合
- POSTAL_CODE: postcode は住所に統合しない
- 落とすラベルは label-mapping-v1.csv の action=drop に従う
出力: {"text","spans":[{"start","end","text","label"}], "source":"nemotron", "domain","document_type"}
分割は --split で train/dev/test を層化 (domain x document_type, seed固定) して別ファイルに出す。
"""
import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

def load_mapping():
    mp = {}
    with open(BASE / "label-mapping-v1.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mp[row["nemotron_label"]] = (row["v1_label"] or None, row["action"])
    return mp

def merge_spans(spans, labels, max_gap=1):
    sel = sorted([s for s in spans if s["label"] in labels], key=lambda s: (s["start"], s["end"]))
    out = []
    for s in sel:
        if out and s["start"] - out[-1]["end"] <= max_gap:
            out[-1]["end"] = max(out[-1]["end"], s["end"])
        else:
            out.append({"start": s["start"], "end": s["end"]})
    return out

def convert_row(row, mp):
    text = row["text"]
    raw = [{"start": int(s["start"]), "end": int(s["end"]), "label": s["label"]} for s in (row["spans"] or [])]
    out = []
    # PERSON 結合
    for m in merge_spans(raw, {"first_name", "last_name"}):
        out.append({**m, "label": "PERSON"})
    # ADDRESS 統合 (postcode は含めない)
    for m in merge_spans(raw, {"street_address", "prefecture", "city"}):
        out.append({**m, "label": "ADDRESS"})
    # 単純マップ
    for s in raw:
        v1, action = mp.get(s["label"], (None, "drop"))
        if action == "map" and v1 not in ("PERSON", "ADDRESS"):
            out.append({"start": s["start"], "end": s["end"], "label": v1})
    # 重なり解消: 長いスパンを優先し、重なる短いものを落とす
    out.sort(key=lambda s: (-(s["end"] - s["start"]), s["start"]))
    kept = []
    for s in out:
        if any(not (s["end"] <= k["start"] or k["end"] <= s["start"]) for k in kept):
            continue
        kept.append(s)
    kept.sort(key=lambda s: s["start"])
    for s in kept:
        s["text"] = text[s["start"]:s["end"]]
    return {"text": text, "spans": kept, "source": "nemotron",
            "domain": row.get("domain"), "document_type": row.get("document_type")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev", type=int, default=10000)
    ap.add_argument("--test", type=int, default=10000)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    t = pq.read_table(args.parquet, columns=["text", "spans", "domain", "document_type"]).to_pylist()
    mp = load_mapping()
    groups = defaultdict(list)
    for r in t:
        groups[(r["domain"], r["document_type"])].append(r)
    rng = random.Random(args.seed)
    splits = {"train": [], "dev": [], "test": []}
    for key, rows in groups.items():
        rng.shuffle(rows)
        n = len(rows)
        n_test = round(n * args.test / len(t)); n_dev = round(n * args.dev / len(t))
        splits["test"] += rows[:n_test]; splits["dev"] += rows[n_test:n_test + n_dev]
        splits["train"] += rows[n_test + n_dev:]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(outdir / f"nemotron-{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(convert_row(r, mp), ensure_ascii=False) + "\n")
        print(name, len(rows))

if __name__ == "__main__":
    main()
