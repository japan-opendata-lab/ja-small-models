#!/usr/bin/env python3
"""学習済みモデルの CPU 推論速度を測る。

使い方:
  python bench_cpu.py --model runs/pilot-bert-v3/final --dev data/nemotron-dev.jsonl --n 200

出力: bench_cpu.json (実行環境、1文書あたり平均ミリ秒、中央値、スループット)
- スレッド数は環境の既定のままにし、変更した場合は報告に明記すること
- GPU は使わない (device を cpu に固定する)
"""
import argparse, json, platform, statistics, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    torch.set_num_threads(torch.get_num_threads())  # 既定を明示
    rows = [json.loads(l) for l in open(args.dev, encoding="utf-8")][: args.n]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to("cpu").eval()
    # ウォームアップ
    for r in rows[:5]:
        enc = tok(r["text"], truncation=True, max_length=args.max_len, return_tensors="pt")
        with torch.no_grad(): model(**enc)
    times = []
    for r in rows:
        enc = tok(r["text"], truncation=True, max_length=args.max_len, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad(): model(**enc)
        times.append((time.perf_counter() - t0) * 1000)
    result = {
        "model": args.model, "device": "cpu", "n_docs": len(rows),
        "mean_ms_per_doc": round(statistics.mean(times), 2),
        "median_ms_per_doc": round(statistics.median(times), 2),
        "docs_per_sec": round(1000 / statistics.mean(times), 2),
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(), "processor": platform.processor(),
        "python": platform.python_version(),
    }
    out = args.out or str(Path(args.model).parent / "bench_cpu.json")
    Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))

if __name__ == "__main__":
    main()
