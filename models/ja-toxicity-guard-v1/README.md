# ja-toxicity-guard-v1

Japanese multi-label toxicity detection: flags signs of harmful or abusive expression across 7 labels (insult, indirect attack, exposure intent, threat, discrimination, sexual harassment, self-harm signals). Built to prioritise human review and assist warnings — not to make moderation decisions by itself. A design goal was to not mistake legitimate criticism, reporting or quotation for abuse.

- Weights (canonical): https://huggingface.co/japan-opendata/ja-toxicity-guard-v1
- About: https://jpopendata.jp/models/ja-toxicity-guard-v1/
- Measured on the sealed test set (5,000 items, shipped thresholds): Macro F1 0.9747, Micro F1 0.9717, exact-match rate 0.9566; the weakest label is indirect attack (F1 0.9384). On harmless text, the false-flag rate was 0.0000 for legitimate criticism and 0.0083 for news / quotation, but higher for discussion, metaphor and general benign text (0.1317 / 0.1450) — see the model card for the full breakdown and caveats.

## Contents

- `code/` — synthetic data generation (`toxicity_templates.py`, `toxicity_synth.py`, `toxicity_prod_templates.py`, `toxicity_prod_synth.py`, `build_pilot_data.py`, `build_prod_data.py`), training (`train_pilot_tox.py`, `train_prod_tox.py`), threshold tuning and sealed-test evaluation (`tune_dev_threshold.py`, `eval_sealed_tox.py`), the reference inference implementation (`infer_toxicity_guard.py`), and ONNX export / benchmarking (`export_onnx_tox.py`, `bench_onnx_tox.py`)
