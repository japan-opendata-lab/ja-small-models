# ja-small-models

Training, evaluation, data-generation and inference code for the small Japanese AI models built by [Japan Open Data Lab](https://jpopendata.jp).

This repository contains **code only**. Model weights are published on Hugging Face (linked below for each model) — that is the canonical home of the weights. Datasets, where public, also live on Hugging Face.

## Models

- **ja-pii-ner-v1** — Japanese personal-information (PII) detection, token classification over 18 entity types.
  Weights: https://huggingface.co/japan-opendata/ja-pii-ner-v1
  About: https://jpopendata.jp/models/ja-pii-ner-v1/
- **ja-injection-guard-v1** — Japanese prompt-injection detection, text classification tuned to keep false positives low.
  Weights: https://huggingface.co/japan-opendata/ja-injection-guard-v1
  About: https://jpopendata.jp/models/ja-injection-guard-v1/

More models will be added here as they are released. The full model shelf is at https://jpopendata.jp/models/ (pages in Japanese).

## Layout

- `models/<model-id>/code/` — training, evaluation and data-generation scripts
- `models/<model-id>/examples/` — inference examples and small utilities (where a model ships them)

Each model folder has its own short README describing what is included.

## License

Code in this repository: MIT (see `LICENSE`). Model weights are published separately on Hugging Face under the license stated on each model card.
