# ja-injection-guard-v1

Japanese prompt-injection detection: text classification tuned to keep false positives under 1%.

- Weights (canonical): https://huggingface.co/japan-opendata/ja-injection-guard-v1
- About: https://jpopendata.jp/models/ja-injection-guard-v1/
- Measured on the sealed test set (2,500 items): detection rate 0.9300, false-positive rate 0.0044, F1 0.9599. Known weaknesses are listed on the model card and the page above.

## Contents

- `code/` — data construction (`build_prod_dataset.py`, `rebuild_prod_dataset_v2.py`, `synth_3layer.py`, `benign_long_pool.py`, `generate_injection.py`, `audit_public_data.py`), training (`train_pilot_cls.py` for the base-model comparison, `train_select_cls.py` for base selection and the production run), evaluation (`eval_sealed_test.py`, `eval_diagnostic_v2.py`) and ONNX export / benchmarking (`export_onnx_cls.py`, `bench_onnx_cls.py`)
