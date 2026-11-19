#!/usr/bin/env python3
"""ja-pii-ner-v1 を Microsoft Presidio の Recognizer として差し込む例。

前提: pip install presidio-analyzer transformers torch
Presidio 標準の日本語認識器は限定的なため、このモデルをカスタム認識器として
登録し、AnalyzerEngine(language="ja") から使えるようにする。
ラベルの対応は v1 の18種を Presidio のエンティティ名へ寄せたもの。
注意: 本モデルのトークナイザはスロー系 (MeCab) のため、環境によっては pipeline が
オフセットを取れずに止まることがある。その場合は inference_onnx.py の方式
(offsets_util.get_enc でオフセットを補完する) で認識器を実装し直してほしい。
"""
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.entity_recognizer import EntityRecognizer
from transformers import pipeline

MODEL = "japan-opendata/ja-pii-ner-v1"

# ja-pii-ner-v1 のラベル -> Presidio のエンティティ名
LABEL_MAP = {
    "PERSON": "PERSON", "ORGANIZATION": "ORGANIZATION", "ADDRESS": "LOCATION",
    "POSTAL_CODE": "LOCATION", "PHONE": "PHONE_NUMBER", "EMAIL": "EMAIL_ADDRESS",
    "DATE_OF_BIRTH": "DATE_TIME", "MY_NUMBER": "JP_MY_NUMBER",
    "BANK_ACCOUNT": "BANK_ACCOUNT", "CREDIT_CARD": "CREDIT_CARD",
    "DRIVER_LICENSE": "JP_DRIVER_LICENSE", "PASSPORT": "PASSPORT",
    "HEALTH_INSURANCE": "JP_HEALTH_INSURANCE", "EMPLOYEE_ID": "ID",
    "IP_ADDRESS": "IP_ADDRESS", "URL_PERSONAL": "URL", "USERNAME": "USERNAME",
    "SECRET": "SECRET",
}

class JaPiiNerRecognizer(EntityRecognizer):
    def __init__(self, model: str = MODEL):
        super().__init__(supported_entities=list(set(LABEL_MAP.values())),
                         supported_language="ja", name="JaPiiNerRecognizer")
        self._ner = pipeline("token-classification", model=model,
                             aggregation_strategy="simple")

    def load(self):
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        wanted = set(entities) if entities else None
        results = []
        for e in self._ner(text):
            ent = LABEL_MAP.get(e["entity_group"])
            if ent is None or (wanted and ent not in wanted):
                continue
            results.append(RecognizerResult(
                entity_type=ent, start=e["start"], end=e["end"],
                score=float(e["score"]), recognition_metadata={
                    RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                    RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                }))
        return results

if __name__ == "__main__":
    analyzer = AnalyzerEngine(supported_languages=["ja"])
    analyzer.registry.add_recognizer(JaPiiNerRecognizer())
    text = "担当は佐藤健太です。連絡先は090-1234-5678までお願いします。"
    for r in analyzer.analyze(text=text, language="ja"):
        print(r.entity_type, text[r.start:r.end], round(r.score, 3))
