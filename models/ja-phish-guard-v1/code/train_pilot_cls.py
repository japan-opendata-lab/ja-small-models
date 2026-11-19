#!/usr/bin/env python3
"""第5号 ja-phish-guard-v1 パイロット学習（単一ヘッド文分類）。

入力 jsonl: {"text", "label", "family", ...}
評価: 閾値0.5 と FPR<=1% 閾値での発見率、クラス別、未知ファミリー別。
リスクスコア = phishing + fraud + spam の確率合計。
使い方:
  python train_pilot_cls.py --base sbintuitions/modernbert-ja-130m \
      --train data/pilot/train.jsonl --dev data/pilot/dev.jsonl \
      --out runs/pilot-modernbert --epochs 1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments)

ATTACK = {"phishing", "fraud", "spam"}


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


class ClsDataset(Dataset):
    def __init__(self, rows, tok, labels, max_len=512):
        self.rows = []
        l2i = {l: i for i, l in enumerate(labels)}
        for r in rows:
            enc = tok(r["text"], truncation=True, max_length=max_len)
            enc["labels"] = l2i[r["label"]]
            self.rows.append(enc)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": torch.tensor(
               [b["input_ids"] + [pad_id] * (maxlen - len(b["input_ids"]))
                for b in batch]),
           "attention_mask": torch.tensor(
               [b["attention_mask"] + [0] * (maxlen - len(b["attention_mask"]))
                for b in batch]),
           "labels": torch.tensor([b["labels"] for b in batch])}
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = torch.tensor(
            [b["token_type_ids"] + [0] * (maxlen - len(b["token_type_ids"]))
             for b in batch])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    train_rows = load_rows(args.train)
    dev_rows = load_rows(args.dev)
    labels = sorted({r["label"] for r in train_rows + dev_rows})
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(labels))
    trainer = Trainer(
        model=model, args=TrainingArguments(
            output_dir=args.out, per_device_train_batch_size=args.bs,
            num_train_epochs=args.epochs, learning_rate=args.lr,
            logging_steps=100, save_strategy="no", eval_strategy="no",
            report_to=[], fp16=torch.cuda.is_available(), seed=42),
        train_dataset=ClsDataset(train_rows, tok, labels),
        data_collator=lambda b: collate(b, tok.pad_token_id or 0))
    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0
    out = Path(args.out)
    (out / "final").mkdir(parents=True, exist_ok=True)
    trainer.save_model(out / "final")
    tok.save_pretrained(out / "final")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    l2i = {l: i for i, l in enumerate(labels)}
    atk_idx = [l2i[l] for l in labels if l in ATTACK]
    probs = []
    ys = []
    fams = []
    t_eval = time.time()
    with torch.no_grad():
        for r in dev_rows:
            enc = tok(r["text"], truncation=True, max_length=512,
                      return_tensors="pt").to(device)
            p = torch.softmax(model(**enc).logits, -1)[0].cpu().numpy()
            probs.append(p)
            ys.append(r["label"])
            fams.append(r.get("family", ""))
    eval_sec = time.time() - t_eval
    probs = np.array(probs)
    risk = probs[:, atk_idx].sum(axis=1)
    y_atk = np.array([1 if y in ATTACK else 0 for y in ys])

    def at(thr):
        pred = (risk >= thr).astype(int)
        tp = int(((pred == 1) & (y_atk == 1)).sum())
        fp = int(((pred == 1) & (y_atk == 0)).sum())
        fn = int(((pred == 0) & (y_atk == 1)).sum())
        return {"threshold": round(float(thr), 4),
                "detection_rate": round(tp / max(1, tp + fn), 4),
                "fpr": round(fp / max(1, int((y_atk == 0).sum())), 4)}

    neg_sorted = np.sort(risk[y_atk == 0])[::-1]
    k = max(1, int(0.01 * len(neg_sorted)))
    thr_fpr1 = float(neg_sorted[min(k, len(neg_sorted) - 1)]) + 1e-6
    pred_cls = probs.argmax(axis=1)
    per_class = {}
    for l in labels:
        m = np.array([y == l for y in ys])
        if m.any():
            per_class[l] = round(float((pred_cls[m] == l2i[l]).mean()), 4)
    holdout_fams = {"phish_password", "fraud_prize"}
    hm = np.array([f in holdout_fams for f in fams])
    report = {"base": args.base, "train_sec": round(train_sec, 1),
              "dev_rows": len(dev_rows),
              "eval_sec_per_doc": round(eval_sec / len(dev_rows), 4),
              "at_0.5": at(0.5), "at_fpr1": at(thr_fpr1),
              "per_class_accuracy": per_class,
              "holdout_family_detection_at_fpr1":
                  round(float((risk[hm] >= thr_fpr1).mean()), 4)
                  if hm.any() else None,
              "model_bytes": sum(p.stat().st_size for p in
                                 (out / "final").rglob("*") if p.is_file())}
    (out / "eval_pilot.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
