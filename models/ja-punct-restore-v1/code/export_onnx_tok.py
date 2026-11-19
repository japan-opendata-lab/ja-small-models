#!/usr/bin/env python3
"""確定済み・バイアス焼き込み済みモデル (runs/prod-modernbert/final-baked) を
optimum 経由で単一 FP32 ONNX (onnx/model_fp32.onnx) へ書き出すスクリプト。

実行は transformers 4.57.6 + optimum[onnx] + onnx を導入した一時隔離環境 (.venv-onnx) を想定し、
グローバル環境および runs/prod-modernbert/final(-baked) は読み取り専用で一切変更しない。
INT8 量子化モデルは生成しない。
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import onnx
import onnxruntime as ort
import torch
import transformers
from transformers import AutoConfig, AutoModelForTokenClassification
import optimum.exporters.onnx.model_configs  # noqa: F401 - TasksManager への登録を確定
from optimum.exporters.onnx.convert import onnx_export_from_model


def prepare_temp_source_dir(src_model_dir: str, tmp_src_dir: str) -> None:
    """元モデルを一切変更せず、transformers 4.57.6 互換の config 設定で一時読込ディレクトリを作る。"""
    src = Path(src_model_dir)
    dst = Path(tmp_src_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            if item.name == "config.json":
                cfg = json.loads(item.read_text(encoding="utf-8"))
                # Transformers 5.x の rope_parameters から 4.57.x の属性へ明示補完
                rp = cfg.get("rope_parameters", {})
                if "global_rope_theta" not in cfg:
                    cfg["global_rope_theta"] = float(rp.get("full_attention", {}).get("rope_theta", 160000.0))
                if "local_rope_theta" not in cfg:
                    cfg["local_rope_theta"] = float(rp.get("sliding_attention", {}).get("rope_theta", 10000.0))
                cfg["reference_compile"] = False
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "model_fp32.onnx"

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_src = Path(tmp_root) / "src"
        tmp_exp = Path(tmp_root) / "exp"
        prepare_temp_source_dir(args.model_dir, str(tmp_src))

        cfg = AutoConfig.from_pretrained(str(tmp_src))
        cfg.attn_implementation = "eager"
        cfg.reference_compile = False
        cfg.use_cache = False

        model = AutoModelForTokenClassification.from_pretrained(
            str(tmp_src),
            config=cfg,
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )
        model.eval()

        onnx_export_from_model(
            model=model,
            output=tmp_exp,
            opset=args.opset,
            monolith=True,
            do_validation=True,
            task="token-classification",
            device="cpu",
            no_dynamic_axes=False,
            do_constant_folding=True,
        )

        raw_onnx = tmp_exp / "model.onnx"
        assert raw_onnx.exists(), f"書き出し後の {raw_onnx} が見つかりません"

        # 外部データ分割されていた場合も含め、単一の自己完結モデルとして onnx/model_fp32.onnx に保存
        onnx_model = onnx.load(str(raw_onnx), load_external_data=True)
        onnx.checker.check_model(onnx_model)
        onnx.save_model(onnx_model, str(fp32_path), save_as_external_data=False)

    # 外部データファイルが残っていないことを保証
    assert not (out_dir / "model_fp32.onnx_data").exists()
    assert not (out_dir / "model.onnx_data").exists()

    t_fp32 = time.time() - t0
    fp32_bytes = fp32_path.stat().st_size
    fp32_sha = sha256_file(fp32_path)

    # CPUExecutionProvider でロード疎通確認
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess_fp32.get_inputs()]
    out_names = [o.name for o in sess_fp32.get_outputs()]
    del sess_fp32

    info = {
        "transformers_export_env": transformers.__version__,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "task": "token-classification",
        "opset": args.opset,
        "fp32_path": str(fp32_path).replace("\\", "/"),
        "fp32_bytes": int(fp32_bytes),
        "fp32_sha256": fp32_sha,
        "fp32_export_sec": round(t_fp32, 2),
        "ort_inputs": in_names,
        "ort_outputs": out_names,
        "int8_created": False,
    }

    meta_path = out_dir / "onnx_export_meta.json"
    meta_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
