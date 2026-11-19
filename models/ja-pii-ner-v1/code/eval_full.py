#!/usr/bin/env python3
"""本評価スクリプト：テスト全件でのラベル別 P/R/F1 と、PIIなし文書の誤検出率を測る。

使い方 (PyTorch 版):
  python eval_full.py --model-dir runs/full-bert-v3/final \
      --test data/nemotron-test.jsonl --no-pii data/no-pii.jsonl \
      --out runs/full-bert-v3/eval_full.json
使い方 (ONNX 版):
  python eval_full.py --onnx onnx/model_int8.onnx --tokenizer runs/full-bert-v3/final \
      --test data/nemotron-test.jsonl --no-pii data/no-pii.jsonl \
      --out onnx/eval_full_int8.json

- テスト側で閾値調整はしない (BIO の argmax 固定)
- PIIなし文書は spans が空の JSONL。文書単位の誤検出率と、誤検出スパンの内訳を出す
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_pilot import get_enc, V1_LABELS  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def decode_spans(offsets, label_ids, id2label):
    spans, cur = [], None
    for (a, b), tid in zip(offsets, label_ids):
        if a == b:
            continue
        lab = id2label[tid]
        if lab.startswith("B-"):
            if cur:
                spans.append(cur)
            cur = {"start": a, "end": b, "label": lab[2:]}
        elif lab.startswith("I-") and cur and cur["label"] == lab[2:]:
            cur["end"] = b
        else:
            if cur:
                spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    return spans


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--tokenizer", default=None, help="ONNX 版のトークナイザの場所")
    ap.add_argument("--test", required=True)
    ap.add_argument("--no-pii", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    tok_src = args.tokenizer or args.model_dir
    tok = AutoTokenizer.from_pretrained(tok_src)

    if args.onnx:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        import json as _json
        cfg = _json.loads(Path(args.model_dir or tok_src, "config.json").read_text(encoding="utf-8")) \
            if args.model_dir else None
        if cfg is None:
            raise SystemExit("ONNX 版は --model-dir (config.json の場所) も必要です")
        id2label = {int(k): v for k, v in cfg["id2label"].items()}

        def predict(text):
            enc, offsets, _ = get_enc(tok, text, args.max_len)
            feed = {k: np.array([v], dtype=np.int64) for k, v in enc.items()}
            logits = sess.run(None, feed)[0][0]
            return decode_spans(offsets, logits.argmax(-1).tolist(), id2label)
    else:
        from transformers import AutoModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
        id2label = model.config.id2label
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()

        def predict(text):
            enc, offsets, _ = get_enc(tok, text, args.max_len)
            enc_pt = {k: torch.tensor([v]).to(device) for k, v in enc.items()}
            with torch.no_grad():
                ids = model(**enc_pt).logits.argmax(-1)[0].tolist()
            return decode_spans(offsets, ids, id2label)

    test_rows = load_jsonl(args.test)
    per = defaultdict(lambda: [0, 0, 0])
    for row in test_rows:
        gold = {(s["start"], s["end"], s["label"]) for s in row["spans"]}
        pred = {(s["start"], s["end"], s["label"]) for s in predict(row["text"])}
        for g in gold & pred:
            per[g[2]][0] += 1
        for p in pred - gold:
            per[p[2]][1] += 1
        for g in gold - pred:
            per[g[2]][2] += 1
    per_label = {}
    TP = FP = FN = 0
    for lab in V1_LABELS:
        tp, fp, fn = per[lab]
        per_label[lab] = prf(tp, fp, fn)
        TP += tp; FP += fp; FN += fn
    result = {"micro": prf(TP, FP, FN), "per_label": per_label,
              "test_docs": len(test_rows)}

    no_rows = load_jsonl(args.no_pii)
    assert all(not r["spans"] for r in no_rows), "no-pii 側にスパン付き文書があります"
    fp_docs, fp_spans = 0, defaultdict(int)
    for row in no_rows:
        preds = predict(row["text"])
        if preds:
            fp_docs += 1
            for s in preds:
                fp_spans[s["label"]] += 1
    result["no_pii"] = {
        "docs": len(no_rows), "docs_with_fp": fp_docs,
        "doc_fpr": round(fp_docs / len(no_rows), 4) if no_rows else 0.0,
        "fp_spans_by_label": dict(fp_spans)}
    result["target"] = args.onnx or args.model_dir
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
