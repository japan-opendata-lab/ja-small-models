# ja-reading-tagger-v1

A small Japanese reading tagger: for each character span of a sentence it returns the context-appropriate reading — a surface reading (for ruby/furigana) and a pronunciation reading (for text-to-speech). Not generative rewriting, and personal-name readings are out of scope. Its specialty is homographs whose reading depends on context (a built-in catalog of 135 words with 289 readings), where dictionary-only tools cannot decide.

- Weights (canonical): https://huggingface.co/japan-opendata/ja-reading-tagger-v1
- About: https://jpopendata.jp/models/ja-reading-tagger-v1/
- Measured on the sealed test set (5,000 sentences, split by work): sentence-level reading accuracy 0.9354, versus 0.9308 for Sudachi (Mode C) and 0.9276 for UniDic. On a homograph-focused set (1,000 sentences) it scores 0.7230 versus Sudachi's 0.5570. Honest caveat from the model card: on the real-text-only subset, Sudachi is slightly ahead (0.9432 vs 0.9360) — the advantage is concentrated in context-decided homographs.

## Contents

- `code/` — source collection and lexicon building (`collect_aozora.py`, `collect_rohan.py`, `collect_aozora_prod.py`, `build_lexicon.py`, `lexicon_util.py`, `span_extractor.py`, `offsets_util.py`, `phonology.py`, `homograph_catalog.py`), synthetic data generation and top-up (`synth_qwen3.py`, `synth_qwen3_prod.py`, `topup_qwen3_prod.py`, `apply_prod_topup.py`, `build_pilot_data.py`, `build_prod_data.py`), training (`train_pilot_reading.py`, `train_prod_reading.py`), evaluation (`eval_reading.py`, `eval_baselines.py`, `eval_final_prod.py`), the reference implementations (`reading_tagger.py`, `reading_tagger_standalone.py`), and ONNX export / benchmarking (`export_onnx_reading.py`, `bench_onnx_reading.py`)
