#!/usr/bin/env python3
"""第8号 ja-toxicity-guard-v1 確定モデル (runs/prod-bert-v3/final) の FP32 ONNX 書き出し。

一時隔離仮想環境 (.venv-onnx) での実行を想定し、
グローバル Python 環境および runs/prod-bert-v3/final は読み取り専用で一切変更しない。
opset 17・FP32 単一ファイル（外部データ分割なし）のみを作成し、INT8 量子化は行わない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# 出力先ディレクトリ onnx/ が onnx パッケージをシャドーイングしないよう CWD を除外
_CWD = str(Path.cwd().resolve())
sys.path = [p for p in sys.path if p not in ("", ".") and str(Path(p).resolve()) != _CWD]

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import onnx
import onnxruntime as ort
import torch
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def prepare_temp_source_dir(src_model_dir: str, tmp_src_dir: str) -> None:
    """元の final/ を一切変更せず、書き出し用の一時読込ディレクトリを作る。"""
    src = Path(src_model_dir)
    dst = Path(tmp_src_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if not item.is_file():
            continue
        if item.name == "config.json":
            cfg = json.loads(item.read_text(encoding="utf-8"))
            cfg["use_cache"] = False
            cfg["attn_implementation"] = "eager"
            (dst / "config.json").write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            shutil.copy2(item, dst / item.name)


def sha256_file(path: Path) -> str:
    """ファイルの SHA-256 ダイジェストを返す。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(1024 * 1024)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="第8号 ja-toxicity-guard-v1 FP32 ONNX 書き出し")
    ap.add_argument("--model-dir", default="runs/prod-bert-v3/final")
    ap.add_argument("--out-dir", default="onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "model_fp32.onnx"

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_src = Path(tmp_root) / "src"
        tmp_exp = Path(tmp_root) / "exp"
        tmp_exp.mkdir(parents=True, exist_ok=True)
        prepare_temp_source_dir(args.model_dir, str(tmp_src))

        cfg = AutoConfig.from_pretrained(str(tmp_src))
        cfg.attn_implementation = "eager"
        cfg.use_cache = False
        assert cfg.model_type == "bert", f"Unexpected model_type: {cfg.model_type}"
        num_labels = int(getattr(cfg, "num_labels", 7))
        assert num_labels == 7, f"Unexpected num_labels: {num_labels}"

        tok = AutoTokenizer.from_pretrained(str(tmp_src))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(tmp_src),
            config=cfg,
            attn_implementation="eager",
            dtype=torch.float32,
        )
        model.eval()

        sample_enc = tok("本日の会議資料を確認しました。", return_tensors="pt")
        input_ids = sample_enc["input_ids"]
        attention_mask = sample_enc["attention_mask"]
        token_type_ids = sample_enc.get("token_type_ids", torch.zeros_like(input_ids))

        raw_onnx = tmp_exp / "model.onnx"
        with torch.no_grad():
            torch.onnx.export(
                model,
                (input_ids, attention_mask, token_type_ids),
                str(raw_onnx),
                dynamo=False,
                opset_version=args.opset,
                do_constant_folding=True,
                export_params=True,
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "token_type_ids": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size"},
                },
            )

        assert raw_onnx.exists(), f"書き出し後の {raw_onnx} が見つかりません"
        onnx_model = onnx.load(str(raw_onnx), load_external_data=True)
        onnx.checker.check_model(onnx_model)
        onnx.save_model(onnx_model, str(fp32_path), save_as_external_data=False)

    assert not (out_dir / "model_fp32.onnx_data").exists()
    assert not (out_dir / "model.onnx_data").exists()

    t_fp32 = time.time() - t0
    fp32_bytes = fp32_path.stat().st_size
    fp32_sha = sha256_file(fp32_path)

    sess_fp32 = ort.InferenceSession(
        str(fp32_path), providers=["CPUExecutionProvider"]
    )
    in_names = [i.name for i in sess_fp32.get_inputs()]
    out_names = [o.name for o in sess_fp32.get_outputs()]

    # 異なるバッチサイズ・シーケンス長（動的軸）で正常推論できることを確認
    for bsz, test_len in ((1, 12), (2, 37)):
        feed = {
            "input_ids": torch.randint(0, 1000, (bsz, test_len), dtype=torch.int64).numpy(),
            "attention_mask": torch.ones((bsz, test_len), dtype=torch.int64).numpy(),
            "token_type_ids": torch.zeros((bsz, test_len), dtype=torch.int64).numpy(),
        }
        out_arr = sess_fp32.run(None, feed)[0]
        assert out_arr.shape == (bsz, num_labels), f"Unexpected ONNX output shape: {out_arr.shape}"
    del sess_fp32

    info = {
        "source_model_dir": str(args.model_dir).replace("\\", "/"),
        "transformers_export_env": transformers.__version__,
        "torch_version": torch.__version__,
        "onnx_version": getattr(onnx, "__version__", "unknown"),
        "onnxruntime_version": ort.__version__,
        "opset": args.opset,
        "task": "text-classification",
        "problem_type": "multi_label_classification",
        "num_labels": num_labels,
        "fp32_path": str(fp32_path).replace("\\", "/"),
        "fp32_bytes": int(fp32_bytes),
        "fp32_sha256": fp32_sha,
        "fp32_export_sec": round(t_fp32, 2),
        "ort_inputs": in_names,
        "ort_outputs": out_names,
        "int8_created": False,
    }

    meta_path = out_dir / "onnx_export_meta.json"
    meta_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
