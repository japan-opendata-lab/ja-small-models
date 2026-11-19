#!/usr/bin/env python3
"""ONNX の静的 INT8 量子化 (ONNX Runtime 標準方式、実入力キャリブレーション)。

使い方:
  python onnx_quantize.py --fp32 onnx/model_fp32.onnx --out onnx/model_int8.onnx \
      --tokenizer runs/full-bert-v3/final --calib data/synthetic-train.jsonl --calib-n 300

独自の量子化は行わない。キャリブレーションには合成データの実入力を使う。
量子化の前後で eval_full.py の数字を比べ、変化をそのまま報告する。
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_pilot import get_enc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--calib", required=True, help="キャリブレーション用 JSONL (text を含む)")
    ap.add_argument("--calib-n", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    import numpy as np
    from onnxruntime.quantization import (CalibrationDataReader, QuantType,
                                          quantize_static)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(args.calib, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f][:args.calib_n]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.data = []
            for r in rows:
                enc, _, _ = get_enc(tok, r["text"], args.max_len)
                self.data.append({k: np.array([v], dtype=np.int64) for k, v in enc.items()})
            self.it = iter(self.data)

        def get_next(self):
            return next(self.it, None)

    quantize_static(args.fp32, args.out, Reader(), weight_type=QuantType.QInt8)
    size = Path(args.out).stat().st_size
    print(f"wrote {args.out} ({size:,} bytes) with {len(rows)} calibration samples")


if __name__ == "__main__":
    main()
