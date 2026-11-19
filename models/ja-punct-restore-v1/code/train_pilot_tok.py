#!/usr/bin/env python3
"""第3号 ja-punct-restore-v1 パイロット学習（トークン分類、両ベース共通）。

ラベルは文字単位（その文字の直後に挿入する記号）。トークンには末尾文字の
ラベルを持たせる。特殊トークンと継続トークンの先頭側は -100 で無視する。

使い方:
  python train_pilot_tok.py --base sbintuitions/modernbert-ja-130m \
      --train data/pilot/train.jsonl --dev data/pilot/dev.jsonl \
      --out runs/pilot-modernbert --epochs 1
出力: eval_pilot.json（記号別 P/R/F1、O を除く micro-F1、MCC、速度、サイズ）
"""
import argparse, json, re, time, unicodedata
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


class PunctDataset(Dataset):
    def __init__(self, path, tok, max_len=640):
        self.rows = []
        for idx, line in enumerate(open(path, encoding="utf-8")):
            r = json.loads(line)
            enc, offsets = encode_with_offsets(tok, r["text"], max_len)
            if idx < 20:
                ref = tok(r["text"], truncation=True, max_length=max_len)
                assert enc["input_ids"] == ref["input_ids"], (
                    f"input_ids 不一致 (row={idx}): {enc['input_ids'][:20]} != {ref['input_ids'][:20]}"
                )
            lab = [-100] * len(enc["input_ids"])
            for ti, (s, e) in enumerate(offsets):
                if e > s:  # 実文字に触れるトークンだけ、末尾文字のラベル
                    lab[ti] = r["labels"][e - 1]
            enc["labels"] = lab
            self.rows.append(enc)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, labels = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * (maxlen - n))
        attn.append(b["attention_mask"] + [0] * (maxlen - n))
        labels.append(b["labels"] + [-100] * (maxlen - n))
    out = {"input_ids": torch.tensor(input_ids),
           "attention_mask": torch.tensor(attn),
           "labels": torch.tensor(labels)}
    if "token_type_ids" in batch[0]:
        tt = [b["token_type_ids"] + [0] * (maxlen - len(b["token_type_ids"]))
              for b in batch]
        out["token_type_ids"] = torch.tensor(tt)
    return out


def metrics_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


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
    model = AutoModelForTokenClassification.from_pretrained(
        args.base, num_labels=len(LABELS), id2label=ID2LABEL,
        label2id=LABEL2ID)
    max_len = min(640, getattr(model.config, "max_position_embeddings", 640))
    train_ds = PunctDataset(args.train, tok, max_len=max_len)
    dev_ds = PunctDataset(args.dev, tok, max_len=max_len)
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs, num_train_epochs=args.epochs,
        learning_rate=args.lr, logging_steps=50, save_strategy="no",
        eval_strategy="no", report_to=[], fp16=torch.cuda.is_available(),
        seed=42)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id or 0))
    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out / "final")
    tok.save_pretrained(out / "final")

    # dev 評価：文字位置（トークンの末尾文字）単位で記号別に集計する
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    tp = np.zeros(5, dtype=np.int64); fp = np.zeros(5, dtype=np.int64)
    fn = np.zeros(5, dtype=np.int64)
    y_true, y_pred = [], []
    n_chars, t_eval = 0, time.time()
    with torch.no_grad():
        for i in range(len(dev_ds)):
            row = dev_ds[i]
            ids = torch.tensor([row["input_ids"]]).to(device)
            attn = torch.tensor([row["attention_mask"]]).to(device)
            kw = {"input_ids": ids, "attention_mask": attn}
            if "token_type_ids" in row:
                kw["token_type_ids"] = torch.tensor(
                    [row["token_type_ids"]]).to(device)
            pred = model(**kw).logits.argmax(-1)[0].cpu().numpy()
            gold = np.array(row["labels"])
            mask = gold != -100
            g, p = gold[mask], pred[mask]
            n_chars += int(mask.sum())
            y_true.extend(g.tolist()); y_pred.extend(p.tolist())
            for c in range(1, 5):
                tp[c] += int(((p == c) & (g == c)).sum())
                fp[c] += int(((p == c) & (g != c)).sum())
                fn[c] += int(((p != c) & (g == c)).sum())
    eval_sec = time.time() - t_eval
    from sklearn.metrics import matthews_corrcoef
    per = {}
    for c in range(1, 5):
        p_, r_, f_ = metrics_from_counts(int(tp[c]), int(fp[c]), int(fn[c]))
        per[LABELS[c]] = {"precision": round(p_, 4), "recall": round(r_, 4),
                          "f1": round(f_, 4), "support": int(tp[c] + fn[c])}
    mp, mr, mf = metrics_from_counts(int(tp[1:].sum()), int(fp[1:].sum()),
                                     int(fn[1:].sum()))
    size = sum(f.stat().st_size for f in (out / "final").rglob("*")
               if f.is_file())
    report = {"base": args.base, "train_sec": round(train_sec, 1),
              "dev_windows": len(dev_ds), "dev_chars": n_chars,
              "chars_per_sec_eval": round(n_chars / eval_sec, 1),
              "micro_no_O": {"precision": round(mp, 4), "recall": round(mr, 4),
                             "f1": round(mf, 4)},
              "mcc_all": round(float(matthews_corrcoef(y_true, y_pred)), 4),
              "per_symbol": per, "final_dir_bytes": size}
    (out / "eval_pilot.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
