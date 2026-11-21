#!/usr/bin/env python3
"""第12号 ja-sentiment-v1 合成データ無作為200件監査用サンプル抽出スクリプト。"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "pilot" / "train.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "audit"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261121)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(ln) for ln in Path(args.input).read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng = random.Random(args.seed)
    sampled = rng.sample(rows, args.n)

    sample_path = out_dir / "audit_sample_200.jsonl"
    with sample_path.open("w", encoding="utf-8") as fout:
        for i, r in enumerate(sampled):
            rec = {
                "audit_idx": i,
                "id": r["id"],
                "sample_id": r["sample_id"],
                "domain": r["domain"],
                "subtopic": r["subtopic"],
                "style": r.get("style", ""),
                "length_band": r.get("length_band", ""),
                "subtype": r.get("subtype", ""),
                "difficulty": r.get("difficulty", "standard"),
                "label": r["label"],
                "text": r["text"],
                "verifier_naturalness": r.get("verifier_naturalness"),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for b in range(0, len(sampled), 50):
        b_no = b // 50 + 1
        md_lines = [f"# 無作為200件監査 バッチ {b_no} ({b}〜{min(b+50, len(sampled))-1})\n"]
        for i in range(b, min(b + 50, len(sampled))):
            r = sampled[i]
            md_lines.append(
                f"## [{i:03d}] {r['id']} | label=**{r['label']}** | dom={r['domain']} ({r['subtopic']}) | sub={r.get('subtype','')}\n"
                f"- **Text ({len(r['text'])}字)**: {r['text']}\n"
            )
        (out_dir / f"audit_batch_{b_no}.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Prepared {len(sampled)} audit samples -> {sample_path} (and 4 batch markdown files)")


if __name__ == "__main__":
    main()
