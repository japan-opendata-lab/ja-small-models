# ja-nli-v1

A small Japanese natural language inference classifier: given a premise and a hypothesis, it returns probabilities for entailment, neutral, and contradiction. Built as a verification component to place around generation — for example, checking that a RAG answer does not contradict its source text, or that a search result supports a claim. Trained entirely on synthetic data from commercially clean materials; existing Japanese NLI datasets were not used for training (licence terms), only as zero-shot references.

- Weights (canonical): https://huggingface.co/japan-opendata/ja-nli-v1
- About: https://jpopendata.jp/models/ja-nli-v1/
- Measured on the sealed test set (5,000 items): accuracy and Macro F1 both 0.9612. On a human-checked evaluation set (600 items): accuracy 0.9567, with entailment/contradiction confusions at 0.5%. Two multilingual NLI models measured zero-shot under the same conditions scored 0.7920 and 0.8094 on the sealed set. Sanity checks are published too: swapping in an unrelated premise drops accuracy to around 0.38, mostly falling back to neutral.

## Contents

- `code/` — seed collection and rule fallbacks (`collect_seeds.py`, `collect_seeds_prod.py`, `nli_rules.py`), synthetic generation and verification (`colab_synth_prod.py`, `colab_synth_verify.py`), data building and auditing (`build_pilot_data.py`, `build_prod_data.py`, `prepare_audit.py`, `aggregate_audit.py`, `build_human_eval.py`, `check_leaks_all.py`, `repair_pilot_lsh_leaks.py`), shortcut diagnostics (`diagnose_shortcuts.py`, `diagnose_shortcuts_prod.py`), training (`train_pilot_nli.py`, `train_prod_nli.py`), evaluation (`eval_nli.py`, `eval_prod_final.py`), the reference implementation (`infer_nli.py`), and ONNX export / benchmarking (`export_onnx_nli.py`, `bench_onnx_nli.py`)
