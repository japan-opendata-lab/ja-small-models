# ja-pii-ner-v1

Japanese personal-information (PII) detection: token classification over 18 entity types.

- Weights (canonical): https://huggingface.co/japan-opendata/ja-pii-ner-v1
- About / live browser demo: https://jpopendata.jp/models/ja-pii-ner-v1/
- Measured on the sealed test set: overall F1 0.9770, zero false positives on 2,068 documents without PII, ONNX at 224 ms per document. Known weaknesses are listed on the model card and the page above — we publish what we verified and what we haven't.

## Contents

- `code/` — data preparation and generation (`prep_parts.py`, `convert_nemotron.py`, `generate_synthetic.py`), training (`train_pilot.py`), full evaluation (`eval_full.py`), ONNX export/quantization and CPU benchmarks (`onnx_quantize.py`, `bench_onnx.py`, `bench_cpu.py`)
- `examples/` — inference examples (`inference_pipeline.py`, `inference_onnx.py`, `presidio_integration.py`) plus the utilities shipped with the model release: a redaction CLI (`redact.py`), a Presidio recognizer (`presidio_recognizer.py`) and offset utilities (`offsets_util.py`)

Training data itself is not in this repository. The companion synthetic dataset is on Hugging Face: https://huggingface.co/datasets/japan-opendata/ja-pii-synthetic-v1
