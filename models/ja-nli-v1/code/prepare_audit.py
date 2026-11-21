#!/usr/bin/env python
"""第11号 ja-nli-v1 合成データ無作為200件監査用サンプル抽出スクリプト。

train.jsonl から固定シード (20261120) で無作為に200件を抽出し、
目視検査用バッチファイルとテンプレート結果を出力する。
"""
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
    ap.add_argument("--seed", type=int, default=20261120)
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
                "pair_id": r["pair_id"],
                "source": r["source"],
                "domain_group": r["domain_group"],
                "len_bucket": r["len_bucket"],
                "subtype": r.get("subtype", ""),
                "label": r["label"],
                "premise": r["premise"],
                "hypothesis": r["hypothesis"],
                "verifier_naturalness": r.get("verifier_naturalness"),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 50件 x 4バッチの可読テキストを出力（目視全件確認用）
    for b in range(0, len(sampled), 50):
        b_no = b // 50 + 1
        md_lines = [f"# 無作為200件監査 バッチ {b_no} ({b}〜{min(b+50, len(sampled))-1})\n"]
        for i in range(b, min(b + 50, len(sampled))):
            r = sampled[i]
            md_lines.append(
                f"## [{i:03d}] {r['id']} | label=**{r['label']}** | src={r['source']} | sub={r.get('subtype','')}\n"
                f"- **Premise ({len(r['premise'])}字)**: {r['premise']}\n"
                f"- **Hypothesis ({len(r['hypothesis'])}字)**: {r['hypothesis']}\n"
            )
        (out_dir / f"audit_batch_{b_no}.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Prepared {len(sampled)} audit samples -> {sample_path} (and 4 batch markdown files)")


if __name__ == "__main__":
    main()
