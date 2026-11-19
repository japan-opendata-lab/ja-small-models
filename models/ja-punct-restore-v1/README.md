# ja-punct-restore-v1

Japanese punctuation restoration for ASR transcripts: character-level token classification that predicts one of none / 、 / 。 / ？ / ！ for each character. Designed as a post-processing step for speech-recognition output such as Whisper transcripts.

- Weights (canonical): https://huggingface.co/japan-opendata/ja-punct-restore-v1
- About: https://jpopendata.jp/models/ja-punct-restore-v1/
- Measured on the sealed test set (5,003 items, never used for training or tuning): punctuation micro-F1 0.8156, MCC 0.8284. On a real ASR benchmark (FLEURS Japanese, 650 utterances transcribed by Whisper), restored punctuation scores F1 0.8881 versus 0.6062 for Whisper's own punctuated output. Known weakness: exclamation marks are hard (sealed-test F1 0.4884) — see the model card for the full breakdown.

## Contents

- `code/` — data construction (`build_punct_data.py`, `build_prod_data.py`, `generate_spoken.py`), training (`train_pilot_tok.py`, `train_prod_tok.py`), class-bias tuning and baking of the classification head (`tune_logit_bias.py`, `bake_logit_bias.py`), real-ASR benchmark construction (`build_asr_bench.py`), the final sealed-test evaluation (`eval_final.py`), and ONNX export / benchmarking (`export_onnx_tok.py`, `bench_onnx_tok.py`)

The tuned class bias is baked into the published model's classification head, so inference needs no extra correction.
