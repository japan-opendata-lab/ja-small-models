# ja-phish-guard-v1

Japanese phishing / fraud / spam detection: a five-class text classifier (phishing / fraud / spam / benign_hard / benign) that also returns a risk score — the summed probability of the three attack classes. Positioned as an auxiliary risk scorer alongside sender authentication and URL inspection, not a replacement for them. All training data is synthetic; no real messages, personal information, or real brand names are used.

- Weights (canonical): https://huggingface.co/japan-opendata/ja-phish-guard-v1
- About: https://jpopendata.jp/models/ja-phish-guard-v1/
- Measured on the sealed test set (2,990 items, never used for training or threshold selection): attack detection rate 1.0 (1,640 / 1,640) with a false-positive rate of 0.0 at the 0.5 risk threshold, and detection stayed at 1.0 on attack families held out of training entirely. See the model card for the full breakdown and caveats.

## Contents

- `code/` — data generation (`gen_phish_pilot.py`, `gen_phish_prod.py`, `fetch_fineweb_benign.py`), training (`train_pilot_cls.py`, also used for the production run), sealed-test evaluation (`eval_sealed_cls.py`) and ONNX export / benchmarking (`export_onnx_cls.py`, `bench_onnx_cls.py`)
