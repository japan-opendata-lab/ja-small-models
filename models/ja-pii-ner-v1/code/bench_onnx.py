#!/usr/bin/env python3
"""ONNX モデルの CPU 推論速度を測る (bench_cpu.py の ONNX 版)。

使い方:
  python bench_onnx.py --onnx onnx/model_int8.onnx --tokenizer runs/full-bert-v3/final \
      --data data/nemotron-test.jsonl --n 200 --out onnx/bench_int8.json
"""
import argparse, json, platform, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_pilot import get_enc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f][:args.n]
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    times = []
    for r in rows:
        enc, _, _ = get_enc(tok, r["text"], 512)
        feed = {k: np.array([v], dtype=np.int64) for k, v in enc.items()}
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    result = {"model": args.onnx, "device": "cpu", "n_docs": len(rows),
              "mean_ms_per_doc": round(statistics.mean(times), 2),
              "median_ms_per_doc": round(statistics.median(times), 2),
              "docs_per_sec": round(1000 / statistics.mean(times), 2),
              "platform": platform.platform(), "processor": platform.processor()}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
