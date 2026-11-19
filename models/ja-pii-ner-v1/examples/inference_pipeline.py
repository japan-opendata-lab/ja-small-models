#!/usr/bin/env python3
"""ja-pii-ner-v1 の最小の使い方 (transformers pipeline) と、マスク処理まで通す実用例。

前提: pip install transformers torch
モデルは検出のみを行う。伏せ字への置換はこの例のように利用者側で行う。
注意: 本モデルのトークナイザはスロー系 (MeCab) のため、環境によっては pipeline が
オフセットを取れずに止まることがある。その場合は inference_onnx.py の方式
(offsets_util.get_enc でオフセットを補完する) を使ってほしい。
"""
from transformers import pipeline

MODEL = "japan-opendata/ja-pii-ner-v1"

ner = pipeline("token-classification", model=MODEL, aggregation_strategy="simple")

TEXT = "お世話になっております、佐藤健太です。電話は090-1234-5678、住所は兵庫県神戸市灘区城内通に変更になりました。"

def detect(text: str):
    return [
        {"label": e["entity_group"], "text": text[e["start"]:e["end"]],
         "start": e["start"], "end": e["end"], "score": round(float(e["score"]), 3)}
        for e in ner(text)
    ]

def mask(text: str, spans) -> str:
    out, last = [], 0
    for s in sorted(spans, key=lambda x: x["start"]):
        out.append(text[last:s["start"]])
        out.append(f"<{s['label']}>")
        last = s["end"]
    out.append(text[last:])
    return "".join(out)

if __name__ == "__main__":
    spans = detect(TEXT)
    for s in spans:
        print(s)
    print(mask(TEXT, spans))
