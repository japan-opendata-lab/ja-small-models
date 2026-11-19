#!/usr/bin/env python3
"""第3号 ja-punct-restore-v1 パイロット用データ構築。

FineWeb-2 日本語の小スライスから、複数文連結ウィンドウを作る。
入力 = 句読点を除いたテキスト、正解 = 各文字の直後に挿入すべき記号。
分割は出典文書単位（同じ文書のウィンドウは同じ分割にしか入れない）。

使い方:
  python build_punct_data.py --docs 3000 --out data/pilot
出力: train.jsonl / dev.jsonl / build-stats.json
  各行: {"text": 入力(句読点なし), "labels": [文字ごとのラベルID], "doc": 文書ID}
  ラベル: 0=O 1=COMMA(、) 2=PERIOD(。) 3=QUESTION(？) 4=EXCLAIM(！)
"""
import argparse, hashlib, json, random, re, sys
from pathlib import Path

PUNCT_MAP = {"，": "、", "．": "。"}  # 中国語式・全角ピリオドを日本語正規約物へ寄せる
TARGETS = {"、": 1, "。": 2, "？": 3, "!": 4, "！": 4}
SENT_END = re.compile(r"(?<=[。？！])")


def normalize(text: str) -> str:
    for k, v in PUNCT_MAP.items():
        text = text.replace(k, v)
    text = text.replace("!", "！")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_labels(text: str):
    """句読点つきテキストから (入力テキスト, 文字ラベル列) を作る。"""
    chars, labels = [], []
    for ch in text:
        if ch in TARGETS:
            if labels:
                labels[-1] = TARGETS[ch]
            continue
        chars.append(ch)
        labels.append(0)
    return "".join(chars), labels


def windows_from_doc(text: str, lo=300, hi=600):
    """文境界で文を連結し、lo〜hi 文字のウィンドウを作る。"""
    sents = [s for s in SENT_END.split(text) if s.strip()]
    wins, buf = [], ""
    for s in sents:
        if len(buf) + len(s) > hi and len(buf) >= lo:
            wins.append(buf)
            buf = s
        else:
            buf += s
    if len(buf) >= lo:
        wins.append(buf)
    return wins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=3000)
    ap.add_argument("--out", default="data/pilot")
    ap.add_argument("--dev-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-2", "jpn_Jpan",
                      split="train", streaming=True)
    rng = random.Random(args.seed)
    rows, doc_count = [], 0
    split_of = {}
    for ex in ds:
        if doc_count >= args.docs:
            break
        text = normalize(ex.get("text") or "")
        if len(text) < 350:
            continue
        doc_id = hashlib.md5(text[:200].encode()).hexdigest()[:12]
        wins = windows_from_doc(text)
        if not wins:
            continue
        doc_count += 1
        # 同一 doc_id（先頭200文字ハッシュが共通する定型文書群）は分割を統一する
        if doc_id not in split_of:
            split_of[doc_id] = "dev" if rng.random() < args.dev_ratio else "train"
        split = split_of[doc_id]
        for w in wins:
            inp, labels = make_labels(w)
            if len(inp) < 100:
                continue
            rows.append({"text": inp, "labels": labels, "doc": doc_id,
                         "split": split})
    # 完全重複の除去（分割をまたぐ重複も落とす）
    seen, uniq = set(), []
    for r in rows:
        h = hashlib.md5(r["text"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        uniq.append(r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"docs": doc_count, "windows": len(uniq), "label_counts": {}}
    for split in ("train", "dev"):
        part = [r for r in uniq if r["split"] == split]
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({"text": r["text"], "labels": r["labels"],
                                    "doc": r["doc"]}, ensure_ascii=False) + "\n")
        stats[f"{split}_windows"] = len(part)
        for r in part:
            for lb in r["labels"]:
                if lb:
                    k = str(lb)
                    stats["label_counts"][k] = stats["label_counts"].get(k, 0) + 1
    # 分割をまたぐ文書の混入チェック
    train_docs = {r["doc"] for r in uniq if r["split"] == "train"}
    dev_docs = {r["doc"] for r in uniq if r["split"] == "dev"}
    stats["doc_overlap"] = len(train_docs & dev_docs)
    stats["unique_rate"] = 1.0
    (out / "build-stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
