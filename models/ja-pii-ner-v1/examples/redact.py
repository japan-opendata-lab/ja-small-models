#!/usr/bin/env python3
"""ja-pii-ner-v1 の単体ヘルパー：日本語テキストから個人情報を検出してマスクする。

依存は transformers と torch のみ。外部通信は一切しない。
長い文書は文字単位の重なりつき分割で処理し、結果を元の座標へ戻す。

使い方:
    from redact import redact
    print(redact("連絡先は taro.yamada@example.com です"))
    # -> 連絡先は <EMAIL> です

    from redact import detect
    spans = detect("担当は山田太郎（090-1234-5678）です")
"""
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from offsets_util import get_enc  # noqa: E402

MODEL_ID = "japan-opendata/ja-pii-ner-v1"
CHUNK_CHARS = 1200
OVERLAP_CHARS = 200

_model = None
_tok = None
_id2label = None


def _load(model_id=MODEL_ID):
    global _model, _tok, _id2label
    if _model is None:
        import torch  # noqa: F401
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForTokenClassification.from_pretrained(model_id)
        _model.eval()
        _id2label = _model.config.id2label
    return _model, _tok


def _decode(offsets, label_ids):
    spans, cur = [], None
    for (a, b), tid in zip(offsets, label_ids):
        if a == b:
            continue
        lab = _id2label[tid]
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


def _predict_chunk(text):
    import torch
    model, tok = _load()
    enc, offsets, _ = get_enc(tok, text, 512)
    # transformers 5.x では token_type_ids が返らないことがあるため、無い場合は 0 で補う
    feed = {k: torch.tensor([v]) for k, v in enc.items()}
    if "token_type_ids" not in feed:
        feed["token_type_ids"] = torch.zeros_like(feed["input_ids"])
    with torch.no_grad():
        ids = model(**feed).logits.argmax(-1)[0].tolist()
    return _decode(offsets, ids)


def detect(text):
    """スパン一覧 [{start, end, label, text}] を返す。座標は元の text 基準。"""
    if len(text) <= CHUNK_CHARS:
        raw = _predict_chunk(text)
    else:
        raw, pos = [], 0
        while pos < len(text):
            chunk = text[pos:pos + CHUNK_CHARS]
            for s in _predict_chunk(chunk):
                raw.append({"start": s["start"] + pos, "end": s["end"] + pos,
                            "label": s["label"]})
            if pos + CHUNK_CHARS >= len(text):
                break
            pos += CHUNK_CHARS - OVERLAP_CHARS
    seen, out = set(), []
    for s in sorted(raw, key=lambda x: (x["start"], x["end"])):
        key = (s["start"], s["end"], s["label"])
        if key in seen:
            continue
        seen.add(key)
        out.append({**s, "text": text[s["start"]:s["end"]]})
    return out


def redact(text, mask=None):
    """検出箇所を <LABEL> (または mask の指定) で置換した文字列を返す。"""
    out, last = [], 0
    for s in detect(text):
        if s["start"] < last:
            continue
        out.append(text[last:s["start"]])
        out.append(mask(s) if callable(mask) else f"<{s['label']}>")
        last = s["end"]
    out.append(text[last:])
    return "".join(out)


if __name__ == "__main__":
    sample = sys.argv[1] if len(sys.argv) > 1 else \
        "担当は山田太郎（電話 090-1234-5678、メール taro.yamada@example.com）です。"
    print(redact(sample))
    for s in detect(sample):
        print(s)
