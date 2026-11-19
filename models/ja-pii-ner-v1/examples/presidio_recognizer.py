#!/usr/bin/env python3
"""Microsoft Presidio へ ja-pii-ner-v1 を組み込むためのカスタム Recognizer。

Presidio 標準の TransformersNlpEngine は spaCy のトークナイズを前段に挟むため、
日本語では文字スパンの整合が崩れやすい。本実装はモデルから直接文字オフセットを
返すため、その問題を避けられる。

使い方:
    from presidio_analyzer import AnalyzerEngine
    from presidio_recognizer import JaPiiRecognizer

    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(JaPiiRecognizer())
    results = analyzer.analyze(text="担当は山田太郎です", language="ja")
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from redact import detect  # noqa: E402

try:
    from presidio_analyzer import EntityRecognizer, RecognizerResult
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install presidio-analyzer が必要です") from e

# 当方ラベル -> Presidio のエンティティ名
LABEL_MAP = {
    "PERSON": "PERSON",
    "ORGANIZATION": "ORGANIZATION",
    "ADDRESS": "LOCATION",
    "POSTAL_CODE": "LOCATION",
    "PHONE": "PHONE_NUMBER",
    "EMAIL": "EMAIL_ADDRESS",
    "DATE_OF_BIRTH": "DATE_TIME",
    "MY_NUMBER": "JP_MY_NUMBER",
    "BANK_ACCOUNT": "JP_BANK_ACCOUNT",
    "CREDIT_CARD": "CREDIT_CARD",
    "DRIVER_LICENSE": "JP_DRIVER_LICENSE",
    "PASSPORT": "JP_PASSPORT",
    "HEALTH_INSURANCE": "JP_HEALTH_INSURANCE",
    "EMPLOYEE_ID": "EMPLOYEE_ID",
    "IP_ADDRESS": "IP_ADDRESS",
    "URL_PERSONAL": "URL",
    "USERNAME": "USERNAME",
    "SECRET": "SECRET",
}


class JaPiiRecognizer(EntityRecognizer):
    def __init__(self, supported_language="ja"):
        super().__init__(supported_entities=list(set(LABEL_MAP.values())),
                         supported_language=supported_language)

    def load(self):
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        wanted = set(entities) if entities else None
        out = []
        for s in detect(text):
            ent = LABEL_MAP.get(s["label"], s["label"])
            if wanted and ent not in wanted:
                continue
            out.append(RecognizerResult(entity_type=ent, start=s["start"],
                                        end=s["end"], score=0.85))
        return out
