#!/usr/bin/env python3
"""第4号 ja-doc-classify-v1 パイロット学習（共有エンコーダ＋3ヘッド）。

ヘッド：領域8種、種別43種、形式2値。損失は3ヘッドの単純合計。
使い方:
  python train_pilot_multi.py --base sbintuitions/modernbert-ja-130m \
      --train data/pilot/train.jsonl --dev data/pilot/dev.jsonl \
      --out runs/pilot-modernbert --epochs 1
出力: eval_pilot.json（種別 Macro F1、領域・形式 Accuracy、種別ワースト5、
      高機密区分の Recall、学習時間、サイズ、評価速度）
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (AutoModel, AutoTokenizer, Trainer,
                          TrainingArguments)

DOMAINS = ["金融", "法務", "医療", "教育", "EC", "雇用", "通信・IT", "行政"]
FORMATS = ["structured", "unstructured"]
HIGH_RISK = {  # risk_tiers_v0.csv の高機密区分（種別名）
    "給付申請書", "入院同意書", "口座開設申込書", "住民異動届", "本人確認書類一覧",
    "振込依頼書", "学生記録票", "保護者連絡票", "処方箋", "検査結果報告書",
    "税務申告関連書類", "診療情報提供書", "成績通知書", "診断書", "本人確認台帳",
    "クレジットカード利用明細", "給与連絡票", "本人確認記録", "海外送金依頼書",
    "ローン審査書類", "奨学金申請書", "退職手続書類"}


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


class ClsDataset(Dataset):
    def __init__(self, rows, tok, types, max_len=1024):
        self.rows = []
        t2i = {t: i for i, t in enumerate(types)}
        for r in rows:
            enc = tok(r["text"], truncation=True, max_length=max_len)
            enc["labels_domain"] = DOMAINS.index(r["domain"])
            enc["labels_type"] = t2i[r["document_type"]]
            enc["labels_format"] = FORMATS.index(r["document_format"])
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
                for b in batch])}
    for k in ("labels_domain", "labels_type", "labels_format"):
        out[k] = torch.tensor([b[k] for b in batch])
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = torch.tensor(
            [b["token_type_ids"] + [0] * (maxlen - len(b["token_type_ids"]))
             for b in batch])
    return out


class MultiHead(nn.Module):
    def __init__(self, base, n_types):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        h = self.encoder.config.hidden_size
        self.head_domain = nn.Linear(h, len(DOMAINS))
        self.head_type = nn.Linear(h, n_types)
        self.head_format = nn.Linear(h, len(FORMATS))

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, labels_domain=None,
                labels_type=None, labels_format=None, **kw):
        kw_in = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and \
                self.encoder.config.model_type == "bert":
            kw_in["token_type_ids"] = token_type_ids
        out = self.encoder(**kw_in)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]
        ld = self.head_domain(pooled)
        lt = self.head_type(pooled)
        lf = self.head_format(pooled)
        loss = None
        if labels_domain is not None:
            f = nn.CrossEntropyLoss()
            loss = f(ld, labels_domain) + f(lt, labels_type) \
                + f(lf, labels_format)
        return {"loss": loss, "logits_domain": ld, "logits_type": lt,
                "logits_format": lf}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    train_rows = load_rows(args.train)
    dev_rows = load_rows(args.dev)
    types = sorted({r["document_type"] for r in train_rows + dev_rows})
    model = MultiHead(args.base, len(types))
    max_len = min(1024, getattr(model.encoder.config, "max_position_embeddings", 1024))
    train_ds = ClsDataset(train_rows, tok, types, max_len=max_len)
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.bs,
        num_train_epochs=args.epochs, learning_rate=args.lr,
        logging_steps=100, save_strategy="no", eval_strategy="no",
        report_to=[], fp16=torch.cuda.is_available(), seed=42,
        remove_unused_columns=False)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id or 0))
    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model_state.pt")
    tok.save_pretrained(out / "final")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    t2i = {t: i for i, t in enumerate(types)}
    yt, pt_, yd, pd_, yf, pf_ = [], [], [], [], [], []
    t_eval = time.time()
    with torch.no_grad():
        for r in dev_rows:
            enc = tok(r["text"], truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            res = model(**enc)
            pt_.append(int(res["logits_type"].argmax(-1)[0]))
            yt.append(t2i[r["document_type"]])
            pd_.append(int(res["logits_domain"].argmax(-1)[0]))
            yd.append(DOMAINS.index(r["domain"]))
            pf_.append(int(res["logits_format"].argmax(-1)[0]))
            yf.append(FORMATS.index(r["document_format"]))
    eval_sec = time.time() - t_eval
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    yt, pt_ = np.array(yt), np.array(pt_)
    per_f1 = f1_score(yt, pt_, average=None, labels=list(range(len(types))),
                      zero_division=0)
    worst = sorted(range(len(types)), key=lambda i: per_f1[i])[:5]
    hr_idx = [t2i[t] for t in HIGH_RISK if t in t2i]
    hr_mask = np.isin(yt, hr_idx)
    report = {"base": args.base, "train_sec": round(train_sec, 1),
              "dev_rows": len(dev_rows),
              "eval_sec_per_doc": round(eval_sec / len(dev_rows), 4),
              "type_macro_f1": round(float(f1_score(
                  yt, pt_, average="macro", zero_division=0)), 4),
              "type_accuracy": round(float(accuracy_score(yt, pt_)), 4),
              "domain_accuracy": round(float(accuracy_score(yd, pd_)), 4),
              "format_accuracy": round(float(accuracy_score(yf, pf_)), 4),
              "high_risk_recall": round(float(recall_score(
                  np.isin(yt, hr_idx).astype(int),
                  np.isin(pt_, hr_idx).astype(int),
                  zero_division=0)), 4) if hr_mask.any() else None,
              "worst5_types": [{"type": types[i], "f1": round(float(per_f1[i]), 4)}
                               for i in worst],
              "n_types": len(types),
              "state_bytes": (out / "model_state.pt").stat().st_size}
    (out / "eval_pilot.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
