# ja-sentiment-v1

A small Japanese sentiment classifier that returns probabilities for three classes — positive, neutral, negative — for use as a pipeline component: review aggregation, first-pass triage of social posts and inquiries, and trend checks over free-text survey answers. Trained entirely on synthetic data built from commercially clean materials only (open-weight LLMs under Apache 2.0 / MIT, independently cross-verified); no scraped reviews and no research-only datasets.

- Weights (canonical): https://huggingface.co/japan-opendata/ja-sentiment-v1
- About: https://jpopendata.jp/models/ja-sentiment-v1/
- Measured on the sealed test set (5,000 items, split by subtopic): accuracy and Macro F1 both 0.9940, with only 2 positive/negative confusions in 5,000. On a human-checked evaluation set (600 items, all visually verified): accuracy 0.9933 with zero confusions. Calibration error (ECE) is around 0.005, so the probabilities work as scores directly. The model card also publishes a same-conditions comparison against an existing Japanese sentiment model, with an explicit caveat about differing label definitions and distributions.

## Contents

- `code/` — scenario catalogs and rule fallbacks (`scenario_catalog.py`, `prod_scenario_catalog.py`, `sentiment_rules.py`), data building and auditing (`build_pilot_data.py`, `build_prod_data.py`, `prepare_audit.py`, `aggregate_audit.py`, `build_human_eval.py`, `colab_synth_verify.py`, `fast_lsh.py`, `check_leaks_all.py`, `check_leaks_prod.py`), training (`train_pilot_sentiment.py`, `train_prod_sentiment.py`), evaluation (`eval_sentiment.py`, `eval_prod_final.py`), the reference implementation (`infer_sentiment.py`), and ONNX export / benchmarking (`export_onnx_sentiment.py`, `bench_onnx_sentiment.py`)
