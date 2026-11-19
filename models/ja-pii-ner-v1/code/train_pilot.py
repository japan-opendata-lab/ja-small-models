#!/usr/bin/env python3
"""パイロット学習 (トークン分類)。

使い方:
  python train_pilot.py --base tohoku-nlp/bert-base-japanese-v3 \
      --train data/nemotron-train.jsonl data/synthetic-train.jsonl \
      --dev data/nemotron-dev.jsonl --limit 10000 --epochs 1 \
      --out runs/pilot-bert-v3
  python train_pilot.py --base sbintuitions/modernbert-ja-130m ... (同条件で比較)

設計:
- 系列長 512、1回のバッチ16、勾配累積2 (実効32)、fp16、500ステップごとに保存
- 文字スパンをトークンへ整列する。スパン外の通常トークンには必ず "O" を付ける
  (付け忘れると O が学習されず全文字が PII 予測に崩壊する。パイロットで実際に踏んだ)
- 評価は文字スパン完全一致の micro P/R/F1 を自前で計算し eval_span_f1.json に出す。
  特殊トークン (開始位置=終了位置) は評価ループで読み飛ばす
- スロートークナイザ (bert-base-japanese-v3 の MeCab 系) は offset_mapping を返さないため、
  NFKC 正規化したテキスト上でトークン表層を順に照合してオフセットを補完する。
  スパンの文字列一致を encode 時に全件 assert し、合わないデータは黙って学習しない
"""
import argparse, json, random, unicodedata
from pathlib import Path

V1_LABELS = ["PERSON","ORGANIZATION","ADDRESS","POSTAL_CODE","PHONE","EMAIL",
             "DATE_OF_BIRTH","MY_NUMBER","BANK_ACCOUNT","CREDIT_CARD","DRIVER_LICENSE",
             "PASSPORT","HEALTH_INSURANCE","EMPLOYEE_ID","IP_ADDRESS","URL_PERSONAL",
             "USERNAME","SECRET"]
LABEL2ID = {"O": 0}
for i, lab in enumerate(V1_LABELS):
    LABEL2ID[f"B-{lab}"] = 2 * i + 1
    LABEL2ID[f"I-{lab}"] = 2 * i + 2
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def load_jsonl(paths, limit=None, seed=42):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows

def span_f1(gold_rows, pred_spans_list):
    tp = fp = fn = 0
    for row, preds in zip(gold_rows, pred_spans_list):
        gold = {(s["start"], s["end"], s["label"]) for s in row["spans"]}
        pred = {(s["start"], s["end"], s["label"]) for s in preds}
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}

def _clusters(text):
    """書記素クラスタの近似分割 (基底文字＋結合文字)。"""
    clusters, cur = [], ""
    for ch in text:
        if cur and (unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Mc") or ch in "\uff9e\uff9f"):
            cur += ch
        else:
            if cur:
                clusters.append(cur)
            cur = ch
    if cur:
        clusters.append(cur)
    return clusters

def nfkc_with_map(text):
    """NFKC 正規化と、正規化後の各文字位置から元の位置への対応表を返す。

    戻り値: (norm, start_map, end_map)
    start_map[j] は正規化後 j 文字目の元の開始位置、end_map[j] は元の終了位置。
    全角・半角カナ結合などで文字数が変わっても、元の座標へ逆変換できる。
    """
    norm_parts, start_map, end_map = [], [], []
    pos = 0
    for cl in _clusters(text):
        n = unicodedata.normalize("NFKC", cl)
        for _ in n:
            start_map.append(pos)
            end_map.append(pos + len(cl))
        norm_parts.append(n)
        pos += len(cl)
    return "".join(norm_parts), start_map, end_map

def get_enc(tok, text, max_len):
    """input_ids / attention_mask / offset_mapping を返す。スロートークナイザにも対応する。

    オフセットは常に元の text に対する座標で返す (スロー時は NFKC 位置マップで逆変換)。
    """
    if tok.is_fast:
        enc = tok(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        return enc, offsets, text
    norm, start_map, end_map = nfkc_with_map(text)
    enc = tok(norm, truncation=True, max_length=max_len)
    tokens = tok.convert_ids_to_tokens(enc["input_ids"])
    offsets, pos = [], 0
    for t in tokens:
        if t in tok.all_special_tokens:
            offsets.append((0, 0))
            continue
        surface = t[2:] if t.startswith("##") else t
        idx = norm.find(surface, pos)
        if idx < 0:
            offsets.append((0, 0))
            continue
        offsets.append((start_map[idx], end_map[idx + len(surface) - 1]))
        pos = idx + len(surface)
    return enc, offsets, text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.base)
    train_rows = load_jsonl(args.train, limit=args.limit)
    dev_rows = load_jsonl([args.dev], limit=2000)

    def encode(row):
        enc, offsets, base_text = get_enc(tok, row["text"], args.max_len)
        labels = [LABEL2ID["O"]] * len(enc["input_ids"])
        for i, (a, b) in enumerate(offsets):
            if a == b:  # 特殊トークンは損失対象外
                labels[i] = -100
        for s in row["spans"]:
            if s.get("text") is not None:
                assert base_text[s["start"]:s["end"]] == s["text"], \
                    f"スパン文字列が一致しません: {s}"
            first = True
            for i, (a, b) in enumerate(offsets):
                if a == b:
                    continue
                if a >= s["start"] and b <= s["end"]:
                    labels[i] = LABEL2ID[f"{'B' if first else 'I'}-{s['label']}"]
                    first = False
        enc["labels"] = labels
        return enc

    class DS(Dataset):
        def __init__(self, rows): self.items = [encode(r) for r in rows]
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id or 0
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            n = maxlen - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad] * n)
            mask = b["attention_mask"] if "attention_mask" in b else [1] * len(b["input_ids"])
            out["attention_mask"].append(mask + [0] * n)
            out["labels"].append(b["labels"] + [-100] * n)
        return {k: torch.tensor(v) for k, v in out.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        args.base, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID)
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs,
        learning_rate=args.lr, fp16=True, logging_steps=50, save_steps=500,
        save_total_limit=2, report_to=[], seed=42, dataloader_num_workers=2)
    Trainer(model=model, args=targs, train_dataset=DS(train_rows), data_collator=collate).train()
    model.save_pretrained(args.out + "/final"); tok.save_pretrained(args.out + "/final")

    # 開発データで文字スパンF1を実測する
    model.eval(); device = "cuda" if torch.cuda.is_available() else "cpu"; model.to(device)
    preds_all = []
    for row in dev_rows:
        enc, offsets, _ = get_enc(tok, row["text"], args.max_len)
        enc_pt = {k: torch.tensor([v]).to(device) for k, v in enc.items() if k != "labels"}
        with torch.no_grad():
            ids = model(**enc_pt).logits.argmax(-1)[0].tolist()
        spans, cur = [], None
        for (a, b), tid in zip(offsets, ids):
            if a == b:  # 特殊トークンはスパンを壊すので読み飛ばす
                continue
            lab = ID2LABEL[tid]
            if lab.startswith("B-"):
                if cur: spans.append(cur)
                cur = {"start": a, "end": b, "label": lab[2:]}
            elif lab.startswith("I-") and cur and cur["label"] == lab[2:]:
                cur["end"] = b
            else:
                if cur: spans.append(cur)
                cur = None
        if cur: spans.append(cur)
        preds_all.append(spans)
    result = span_f1(dev_rows, preds_all)
    result.update({"base": args.base, "train_rows": len(train_rows), "dev_rows": len(dev_rows)})
    Path(args.out, "eval_span_f1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
