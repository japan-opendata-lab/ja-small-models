# ja-address-ner-v1

A two-stage Japanese address component: a token-classification model extracts an address from a sentence and splits it into components (prefecture, municipality, district, street number, building), then a dictionary linker attaches the municipality code. When the code cannot be determined uniquely, it returns "undetermined" instead of guessing. Trained only on fictional addresses — no real personal addresses are used.

- Weights (canonical): https://huggingface.co/japan-opendata/ja-address-ner-v1
- About: https://jpopendata.jp/models/ja-address-ner-v1/
- Measured on the sealed test set (5,000 items): address-span exact-match F1 0.9983, component micro-F1 0.9965, zero false detections on 500 address-free sentences. Through the linker, code assignment was correct on 0.9924 of determinable cases with zero wrong codes asserted anywhere (all linker errors were over-rejection), and undeterminable cases were correctly rejected at 1.0000. See the model card for conditions and limits.

## Contents

- `code/` — address synthesis and master building (`address_synth.py`, `master_builder.py`, `geo_norm.py`, `linker.py`, `offsets_util.py`), data construction (`build_pilot_data.py`, `build_prod_data.py`), training (`train_pilot_addr.py`, `train_prod_addr.py`), evaluation (`eval_linker.py`, `eval_sealed_addr.py`), the reference inference implementation (`infer_address.py`), and ONNX export / benchmarking (`export_onnx_addr.py`, `bench_onnx_addr.py`)
- `code/release_linker/` — the linker package shipped with the model release (`linker.py`, `geo_norm.py`, `build_dictionary.py`, and `DICTIONARY-SETUP.md` explaining how to build the dictionary from public data: Japan Post postal-code data and the Digital Agency Address Base Registry). The dictionary data itself is not included in this repository.
