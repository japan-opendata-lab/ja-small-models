# ja-doc-classify-v1

A document router for the intake side of Japanese business-document processing: it assigns each document to one of 8 domains (finance, medical, legal, government, employment, education, e-commerce, telecom/IT), suggests one of 43 document types only when confidence is high enough (otherwise it returns the domain alone), and flags documents that may be highly sensitive. Designed to sit in front of ja-pii-ner-v1.

- Weights (canonical): https://huggingface.co/japan-opendata/ja-doc-classify-v1
- About: https://jpopendata.jp/models/ja-doc-classify-v1/
- Measured on out-of-distribution sets written by people (2 × 200 documents, independent of the synthetic training data), with the shipped rejection thresholds applied: domain accuracy 0.989 on the documents not rejected (5–6% rejected); document-type accuracy 0.82–0.87 on the documents not rejected (26–30% rejected); high-sensitivity flag recall 0.94–0.975 with an 8–12.5% false-flag rate on ordinary documents. Some similar types are weak (a few scored F1 0 on the harder set) — see the model card for the full breakdown and caveats.

## Contents

- `code/` — data construction (`build_cls_data.py`, `build_train_v2.py`), out-of-distribution evaluation sets (`build_ood_set.py`, `plan_ood_v2.py`, `build_ood_v2.py`), training and evaluation (`train_pilot_multi.py`, `eval_cls_multi.py`), routing-threshold analysis (`analyze_v2_routing.py`), and export / benchmarking (`export_hf_v2.py`, `export_onnx_multi.py`, `bench_onnx_multi.py`)
