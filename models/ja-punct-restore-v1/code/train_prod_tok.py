#!/usr/bin/env python3
"""第3号 ja-punct-restore-v1 本番学習（トークン分類）＋文字単位評価。

パイロットとの違い：
- 損失にクラス重み（頻度の平方根逆数、上限つき）を入れる
- 評価はトークン単位でなく文字位置へ復元して採点する。長いウィンドウは
  重なりつき分割で推論し、切り詰めで評価対象が欠落しないようにする

使い方:
  python train_prod_tok.py --base sbintuitions/modernbert-ja-130m \
      --train data/prod/train.jsonl --dev data/prod/dev.jsonl \
      --out runs/prod-modernbert --epochs 2
出力: eval_prod.json（記号別 P/R/F1、O を除く micro-F1、MCC、速度、サイズ）
"""
import argparse, json, math, re, time, unicodedata
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

LABELS = ["O", "COMMA", "PERIOD", "QUESTION", "EXCLAIM"]
ID2LABEL = {i: s for i, s in enumerate(LABELS)}
LABEL2ID = {s: i for i, s in enumerate(LABELS)}


def _clusters(text: str):
    """書記素クラスタの近似分割（基底文字＋結合文字・半角濁点）。"""
    clusters, cur = [], ""
    for ch in text:
        if cur and (unicodedata.combining(ch)
                    or unicodedata.category(ch) in ("Mn", "Me", "Mc")
                    or ch in "\uff9e\uff9f"):
            cur += ch
        else:
            if cur:
                clusters.append(cur)
            cur = ch
    if cur:
        clusters.append(cur)
    return clusters


def nfkc_with_map(text: str):
    """NFKC 正規化と元文字オフセット対応表（展開末尾文字のみ有効範囲）を返す。"""
    norm_parts, start_map, end_map = [], [], []
    pos = 0
    for cl in _clusters(text):
        n = unicodedata.normalize("NFKC", cl)
        if not n:
            pos += len(cl)
            continue
        for j in range(len(n)):
            if j == len(n) - 1:
                start_map.append(pos)
                end_map.append(pos + len(cl))
            else:
                start_map.append(0)
                end_map.append(0)
        norm_parts.append(n)
        pos += len(cl)
    return "".join(norm_parts), start_map, end_map


def encode_with_offsets(tok, text: str, max_len: int):
    """Fast / Slow 両トークナイザで (encoding, offset_mapping) を返す。"""
    if tok.is_fast:
        enc = tok(text, return_offsets_mapping=True,
                  truncation=True, max_length=max_len)
        offsets = enc.pop("offset_mapping")
        return enc, offsets
    norm, start_map, end_map = nfkc_with_map(text)
    specials = sorted(tok.all_special_tokens, key=len, reverse=True)
    sp_re = (re.compile("(" + "|".join(re.escape(s) for s in specials) + ")")
             if specials else None)
    segments = sp_re.split(norm) if sp_re else [norm]
    sp_set = set(specials)
    vocab = tok.subword_tokenizer.vocab
    unk = tok.unk_token
    tokens, offsets = [], []
    pos = 0
    for seg in segments:
        if not seg:
            continue
        if seg in sp_set:
            idx = norm.find(seg, pos)
            if idx >= 0:
                pos = idx + len(seg)
            tokens.append(seg)
            offsets.append((0, 0))
            continue
        for word in tok.word_tokenizer.tokenize(
                seg, never_split=tok.all_special_tokens):
            idx = norm.find(word, pos)
            if idx < 0:
                for ch in unicodedata.normalize("NFKC", word):
                    tokens.append(ch if ch in vocab else unk)
                    offsets.append((0, 0))
                continue
            for k, ch in enumerate(word):
                ni = idx + k
                tokens.append(ch if ch in vocab else unk)
                offsets.append((start_map[ni], end_map[ni]))
            pos = idx + len(word)
    max_body = max_len - tok.num_special_tokens_to_add(pair=False)
    tokens = tokens[:max_body]
    offsets = offsets[:max_body]
    ids = tok.build_inputs_with_special_tokens(
        tok.convert_tokens_to_ids(tokens))
    offsets = [(0, 0)] + offsets + [(0, 0)]
    enc = {"input_ids": ids, "attention_mask": [1] * len(ids)}
    if "token_type_ids" in tok.model_input_names:
        enc["token_type_ids"] = [0] * len(ids)
    return enc, offsets


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


class PunctDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows = []
        for idx, r in enumerate(rows):
            enc, offsets = encode_with_offsets(tok, r["text"], max_len)
            if idx < 20:
                ref = tok(r["text"], truncation=True, max_length=max_len)
                assert enc["input_ids"] == ref["input_ids"], (
                    f"input_ids 不一致 (row={idx}): {enc['input_ids'][:20]} != {ref['input_ids'][:20]}"
                )
            lab = [-100] * len(enc["input_ids"])
            for ti, (s, e) in enumerate(offsets):
                if e > s:
                    lab[ti] = r["labels"][e - 1]
            enc["labels"] = lab
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
           "labels": torch.tensor(
               [b["labels"] + [-100] * (maxlen - len(b["labels"]))
                for b in batch])}
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = torch.tensor(
            [b["token_type_ids"] + [0] * (maxlen - len(b["token_type_ids"]))
             for b in batch])
    return out


class WeightedTrainer(Trainer):
    def __init__(self, *a, class_weights=None, **kw):
        super().__init__(*a, **kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device), ignore_index=-100)
        loss = loss_fct(logits.float().view(-1, len(LABELS)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


@torch.no_grad()
def predict_chars(model, tok, text, device, max_len, overlap=128):
    """文字位置ごとの予測ラベルを返す（重なり分割、最初の予測を優先）。"""
    pred = np.full(len(text), -1, dtype=np.int64)
    if tok.is_fast:
        step = max(1, max_len - overlap)
        start = 0
        while start < len(text):
            chunk = text[start:start + max_len]
            enc = tok(chunk, return_offsets_mapping=True)
            ids = torch.tensor([enc["input_ids"]]).to(device)
            kw = {"input_ids": ids,
                  "attention_mask": torch.tensor([enc["attention_mask"]]).to(device)}
            if "token_type_ids" in enc:
                kw["token_type_ids"] = torch.tensor([enc["token_type_ids"]]).to(device)
            logits = model(**kw).logits[0].argmax(-1).cpu().numpy()
            for ti, (s, e) in enumerate(enc["offset_mapping"]):
                if e > s:
                    pos = start + e - 1
                    if pred[pos] == -1:
                        pred[pos] = int(logits[ti])
            if start + max_len >= len(text):
                break
            start += step
        pred[pred == -1] = 0
        return pred

    chunk_chars = max(1, max_len - 32)
    step = max(1, chunk_chars - overlap)
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_chars]
        enc, offsets = encode_with_offsets(tok, chunk, max_len)
        ids = torch.tensor([enc["input_ids"]]).to(device)
        kw = {"input_ids": ids,
              "attention_mask": torch.tensor([enc["attention_mask"]]).to(device)}
        if "token_type_ids" in enc:
            kw["token_type_ids"] = torch.tensor([enc["token_type_ids"]]).to(device)
        logits = model(**kw).logits[0].argmax(-1).cpu().numpy()
        max_e = 0
        for ti, (s, e) in enumerate(offsets):
            if e > s:
                if e > max_e:
                    max_e = e
                pos = start + e - 1
                if pred[pos] == -1:
                    pred[pos] = int(logits[ti])
        if start + chunk_chars >= len(text) and len(enc["input_ids"]) < max_len:
            break
        if len(enc["input_ids"]) >= max_len:
            if max_e == 0:
                adv = step
            elif start + max_e >= len(text):
                break
            else:
                adv = max(1, max_e - overlap if max_e > overlap else max_e)
        else:
            adv = step
        start += adv
    pred[pred == -1] = 0
    return pred


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=640)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    train_rows = load_rows(args.train)
    dev_rows = load_rows(args.dev)
    # クラス重み：頻度の平方根逆数（O は 0.1 に固定、上限 30）
    counts = np.zeros(5, dtype=np.float64)
    for r in train_rows:
        for lb in r["labels"]:
            counts[lb] += 1
    w = 1.0 / np.sqrt(counts / counts.sum())
    w = w / w[1:].mean()
    w[0] = 0.1
    w = np.minimum(w, 30.0)
    class_weights = torch.tensor(w, dtype=torch.float32)
    print("class_weights:", [round(float(x), 3) for x in w],
          "counts:", counts.astype(int).tolist())

    model = AutoModelForTokenClassification.from_pretrained(
        args.base, num_labels=len(LABELS), id2label=ID2LABEL,
        label2id=LABEL2ID)
    max_len = min(args.max_len, getattr(model.config, "max_position_embeddings", args.max_len))
    train_ds = PunctDataset(train_rows, tok, max_len)
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.bs,
        num_train_epochs=args.epochs, learning_rate=args.lr,
        logging_steps=100, save_strategy="no", eval_strategy="no",
        report_to=[], fp16=torch.cuda.is_available(), seed=42)
    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=lambda b: collate(b, tok.pad_token_id or 0),
        class_weights=class_weights)
    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out / "final")
    tok.save_pretrained(out / "final")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    tp = np.zeros(5, dtype=np.int64); fp = np.zeros(5, dtype=np.int64)
    fn = np.zeros(5, dtype=np.int64)
    y_true, y_pred = [], []
    t_eval = time.time()
    total_chars = 0
    for r in dev_rows:
        pred = predict_chars(model, tok, r["text"], device, max_len)
        gold = np.array(r["labels"])
        total_chars += len(gold)
        y_true.extend(gold.tolist()); y_pred.extend(pred.tolist())
        for c in range(1, 5):
            tp[c] += int(((pred == c) & (gold == c)).sum())
            fp[c] += int(((pred == c) & (gold != c)).sum())
            fn[c] += int(((pred != c) & (gold == c)).sum())
    eval_sec = time.time() - t_eval
    from sklearn.metrics import matthews_corrcoef
    per = {}
    for c in range(1, 5):
        p_, r_, f_ = prf(int(tp[c]), int(fp[c]), int(fn[c]))
        per[LABELS[c]] = {"precision": p_, "recall": r_, "f1": f_,
                          "support": int(tp[c] + fn[c])}
    mp, mr, mf = prf(int(tp[1:].sum()), int(fp[1:].sum()), int(fn[1:].sum()))
    size = sum(f.stat().st_size for f in (out / "final").rglob("*")
               if f.is_file())
    report = {"base": args.base, "epochs": args.epochs,
              "class_weights": [round(float(x), 3) for x in w],
              "train_sec": round(train_sec, 1),
              "dev_windows": len(dev_rows), "dev_chars": total_chars,
              "chars_per_sec_eval": round(total_chars / eval_sec, 1),
              "micro_no_O": {"precision": mp, "recall": mr, "f1": mf},
              "mcc_all": round(float(matthews_corrcoef(y_true, y_pred)), 4),
              "per_symbol": per, "final_dir_bytes": size}
    (out / "eval_prod.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
