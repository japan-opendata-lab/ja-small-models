# ja-error-detect-v1

Japanese error-span detection: a token-classification model that points at the spans of a sentence that look wrong — particle substitutions, homophone mis-conversions, okurigana errors, visually similar kanji substitutions, missing characters and extra characters (6 error types). It does not propose corrections; flagging candidates for a proofreading step or an input warning is its whole job. Dialect, style and colloquial variation are not treated as errors.

- Weights (canonical): https://huggingface.co/japan-opendata/ja-error-detect-v1
- About: https://jpopendata.jp/models/ja-error-detect-v1/
- Measured on the sealed test set (5,000 items, threshold fixed on dev beforehand): exact-span Precision 0.9094, Recall 0.8291, F1 0.8674. The weakest type is extra characters (F1 0.7686). After manual review of every flagged span on clean text, the true false-alarm rate was 1.20% of sentences — see the model card for the breakdown and caveats.

## Contents

- `code/` — error injection and data construction (`error_injector.py`, `build_pilot_data.py`, `build_prod_data.py`, `offsets_util.py`), training (`train_pilot_err.py`, `train_prod_err.py`), evaluation and threshold tuning (`eval_common.py`, `tune_dev_threshold.py`, `eval_sealed_err.py`), the reference inference implementation (`infer_error_detect.py`), and ONNX export / benchmarking (`export_onnx_err.py`, `bench_onnx_err.py`)
