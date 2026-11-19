#!/usr/bin/env python3
"""インジェクション検知パイロット (文分類)。ベース2本を同一条件で比べるための最小実装。

使い方:
  python train_pilot_cls.py --base sbintuitions/modernbert-ja-130m \
      --train train.jsonl --dev dev.jsonl --out runs/pilot-modernbert

評価は攻撃の発見率 (Recall) と通常文の誤検出率 (FPR) を主役にし、
サブカテゴリ別の発見率も eval_cls.json に出す。閾値は 0.5 固定 (調整は内部分割のみ)。
"""
import argparse, json, time
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


class ClsDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.enc = tok([r["text"] for r in rows], truncation=True, max_length=max_len)
        self.labels = [int(r["label"]) for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def metrics_from_logits(logits, rows):
    import numpy as np
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    pred = (probs >= 0.5).astype(int)
    gold = [int(r["label"]) for r in rows]
    tp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 1)
    fn = sum(1 for p, g in zip(pred, gold) if p == 0 and g == 1)
    fp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 0)
    tn = sum(1 for p, g in zip(pred, gold) if p == 0 and g == 0)
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
    per_cat = defaultdict(lambda: [0, 0])
    for p, r in zip(pred, rows):
        per_cat[r.get("subcategory", "?")][1] += 1
        if p == int(r["label"]):
            per_cat[r.get("subcategory", "?")][0] += 1
    return {
        "attack_recall": round(recall, 4), "benign_fpr": round(fpr, 4),
        "precision": round(prec, 4), "f1_binary": round(f1, 4),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "per_subcategory_accuracy": {k: round(c / n, 4) for k, (c, n) in sorted(per_cat.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base)
    train_rows, dev_rows = load_jsonl(args.train), load_jsonl(args.dev)
    train_ds, dev_ds = ClsDataset(train_rows, tok, args.max_len), ClsDataset(dev_rows, tok, args.max_len)

    def collate(feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        pad_id = tok.pad_token_id or 0
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in feats:
            n = maxlen - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * n)
            batch["attention_mask"].append(f["attention_mask"] + [0] * n)
            batch["labels"].append(f["labels"])
        return {k: torch.tensor(v) for k, v in batch.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2, id2label={0: "benign", 1: "injection"},
        label2id={"benign": 0, "injection": 1})
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, logging_steps=50, save_strategy="steps", save_steps=500,
        save_total_limit=2, fp16=torch.cuda.is_available(), report_to=[],
        seed=42, data_seed=42, eval_strategy="no")
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, data_collator=collate)
    trainer.train()

    preds = trainer.predict(dev_ds)
    result = metrics_from_logits(preds.predictions, dev_rows)
    result.update({"base": args.base, "train_n": len(train_rows), "dev_n": len(dev_rows),
                   "epochs": args.epochs, "elapsed_sec": round(time.time() - t0, 1)})
    trainer.save_model(args.out + "/final")
    tok.save_pretrained(args.out + "/final")
    with open(args.out + "/eval_cls.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
