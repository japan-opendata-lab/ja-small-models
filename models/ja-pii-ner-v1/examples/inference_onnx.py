#!/usr/bin/env python3
"""ja-pii-ner-v1 の ONNX 版を使う例 (ONNX Runtime, CPU)。

前提: pip install onnxruntime transformers
リポジトリ内の onnx/model_int8.onnx を使う想定。パスは公開時の配置に合わせて調整する。
トークナイザはリポジトリのものをそのまま使う。
"""
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from offsets_util import get_enc

MODEL = "japan-opendata/ja-pii-ner-v1"
ONNX_PATH = "onnx/model_int8.onnx"  # ローカルに落としたリポジトリ内のパス

tok = AutoTokenizer.from_pretrained(MODEL)
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

def load_id2label():
    import json, urllib.request
    with urllib.request.urlopen(f"https://huggingface.co/{MODEL}/resolve/main/config.json") as r:
        cfg = json.load(r)
    return {int(k): v for k, v in cfg["id2label"].items()}

def detect(text: str, labels):
    # スロートークナイザでもオフセットが取れるよう get_enc を使う。
    # スパンの位置は base_text (スロー時は NFKC 正規化後の文字列) 基準で返す
    enc, offsets, base_text = get_enc(tok, text)
    feed = {k: np.array([v]) for k, v in enc.items()}
    logits = sess.run(None, feed)[0][0]
    preds = logits.argmax(-1)
    spans, cur = [], None
    for (s, e), p in zip(offsets, preds):
        if s == e:  # 特殊トークンは読み飛ばす
            continue
        lab = labels[int(p)]
        if lab.startswith("B-"):
            if cur: spans.append(cur)
            cur = {"label": lab[2:], "start": int(s), "end": int(e)}
        elif lab.startswith("I-") and cur and cur["label"] == lab[2:]:
            cur["end"] = int(e)
        else:
            if cur: spans.append(cur)
            cur = None
    if cur: spans.append(cur)
    return [{**sp, "text": base_text[sp["start"]:sp["end"]]}
            for sp in spans if sp["end"] > sp["start"]]

if __name__ == "__main__":
    labels = load_id2label()
    text = "担当は佐藤健太です。連絡先は090-1234-5678までお願いします。"
    for sp in detect(text, labels):
        print(sp)
