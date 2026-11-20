#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 確定モデル (runs/prod-modernbert/best) の FP32 ONNX 書き出しスクリプト。

一時隔離仮想環境 (.venv-onnx) での実行を想定し、グローバル Python 環境および
元の runs/prod-modernbert/best/ は読み取り専用で一切変更しない。
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
from typing import Dict

# 出力先ディレクトリ onnx/ が onnx パッケージをシャドーイングしないよう CWD を除外
_CWD = str(Path.cwd().resolve())
sys.path = [p for p in sys.path if p not in ("", ".") and str(Path(p).resolve()) != _CWD]

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import transformers
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer


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


def record_source_hashes(src_dir: Path) -> Dict[str, dict]:
    """元モデルディレクトリ全ファイルのサイズと SHA-256 を読み取り専用で記録する。"""
    records: Dict[str, dict] = {}
    for p in sorted(src_dir.iterdir()):
        if p.is_file():
            records[p.name] = {
                "bytes": int(p.stat().st_size),
                "sha256": sha256_file(p),
            }
    return records


def prepare_temp_source_dir(src_model_dir: str, tmp_src_dir: str) -> None:
    """元モデルを一切変更せず、ONNX 書き出し互換設定を施した一時読込ディレクトリを作成する。"""
    src = Path(src_model_dir)
    dst = Path(tmp_src_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if not item.is_file():
            continue
        if item.name == "config.json":
            cfg = json.loads(item.read_text(encoding="utf-8"))
            rp = cfg.get("rope_parameters", {})
            if "global_rope_theta" not in cfg:
                cfg["global_rope_theta"] = float(
                    rp.get("full_attention", {}).get("rope_theta", 160000.0)
                )
            if "local_rope_theta" not in cfg:
                cfg["local_rope_theta"] = float(
                    rp.get("sliding_attention", {}).get("rope_theta", 10000.0)
                )
            cfg["reference_compile"] = False
            cfg["use_cache"] = False
            cfg["attn_implementation"] = "eager"
            (dst / "config.json").write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            shutil.copy2(item, dst / item.name)


def export_direct(
    model, tok, raw_onnx: Path, opset: int
) -> None:
    """torch.onnx.export (dynamo=False, opset=17) で直接書き出す。"""
    sample_enc = tok(
        "人気のない路地と人気のある店を歩いて回った。",
        return_tensors="pt",
    )
    input_ids = sample_enc["input_ids"]
    attention_mask = sample_enc["attention_mask"]

    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(raw_onnx),
            dynamo=False,
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
        )


def export_optimum(model, tmp_exp: Path, opset: int) -> Path:
    """optimum.exporters.onnx 経由で書き出す（transformers 4.57.x + optimum 環境用フォールバック）。"""
    import optimum.exporters.onnx.model_configs  # noqa: F401
    from optimum.exporters.onnx.convert import onnx_export_from_model

    onnx_export_from_model(
        model=model,
        output=tmp_exp,
        opset=opset,
        monolith=True,
        do_validation=True,
        task="token-classification",
        device="cpu",
        no_dynamic_axes=False,
        do_constant_folding=True,
    )
    return tmp_exp / "model.onnx"


def load_model_fp32(tmp_src: Path):
    """一時ディレクトリから FP32 モデルを読み込む（transformers 4.x/5.x 両対応）。"""
    cfg = AutoConfig.from_pretrained(str(tmp_src))
    cfg.attn_implementation = "eager"
    cfg.reference_compile = False
    cfg.use_cache = False
    assert cfg.model_type == "modernbert", f"想定外の model_type: {cfg.model_type}"

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            str(tmp_src),
            config=cfg,
            attn_implementation="eager",
            dtype=torch.float32,
        )
    except TypeError:
        model = AutoModelForTokenClassification.from_pretrained(
            str(tmp_src),
            config=cfg,
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )
    model.eval()
    return cfg, model


def main() -> int:
    ap = argparse.ArgumentParser(description="第9号 ja-reading-tagger-v1 FP32 ONNX 書き出し")
    ap.add_argument(
        "--model-dir",
        default="runs/prod-modernbert/best",
    )
    ap.add_argument(
        "--out-dir",
        default="onnx",
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--mode", choices=["auto", "direct", "optimum"], default="auto")
    args = ap.parse_args()

    src_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "model_fp32.onnx"

    # 元モデル全ファイルの事前 SHA-256 を記録（以降一切変更しない）
    source_hashes = record_source_hashes(src_dir)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_src = Path(tmp_root) / "src"
        tmp_exp = Path(tmp_root) / "exp"
        tmp_exp.mkdir(parents=True, exist_ok=True)
        prepare_temp_source_dir(str(src_dir), str(tmp_src))

        tok = AutoTokenizer.from_pretrained(str(tmp_src))
        cfg, model = load_model_fp32(tmp_src)

        export_method = args.mode
        raw_onnx = tmp_exp / "model.onnx"
        if args.mode == "direct":
            export_direct(model, tok, raw_onnx, args.opset)
        elif args.mode == "optimum":
            raw_onnx = export_optimum(model, tmp_exp, args.opset)
            export_method = "optimum"
        else:
            try:
                export_direct(model, tok, raw_onnx, args.opset)
                export_method = "direct"
            except Exception as exc:
                print(f"[INFO] direct export failed ({type(exc).__name__}: {exc}), falling back to optimum...", flush=True)
                raw_onnx = export_optimum(model, tmp_exp, args.opset)
                export_method = "optimum"

        assert raw_onnx.exists(), f"書き出し後の {raw_onnx} が見つかりません"
        onnx_model = onnx.load(str(raw_onnx), load_external_data=True)
        onnx.checker.check_model(onnx_model)
        onnx.save_model(onnx_model, str(fp32_path), save_as_external_data=False)

        # バッチ=2・可変長テキストでの PyTorch vs ONNX スモークテスト
        smoke_texts = [
            "工事現場の工夫が工夫を凝らして安全を守る。",
            "重要な会議を行った後に、家族で海へ行った。",
        ]
        enc2 = tok(smoke_texts, padding=True, return_tensors="pt")
        with torch.no_grad():
            pt_logits = model(
                input_ids=enc2["input_ids"],
                attention_mask=enc2["attention_mask"],
            ).logits.numpy()

    assert not (out_dir / "model_fp32.onnx_data").exists()
    assert not (out_dir / "model.onnx_data").exists()

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    assert in_names == ["input_ids", "attention_mask"], f"入力名不一致: {in_names}"
    assert out_names == ["logits"], f"出力名不一致: {out_names}"

    ox_logits = sess.run(
        None,
        {
            "input_ids": enc2["input_ids"].numpy(),
            "attention_mask": enc2["attention_mask"].numpy(),
        },
    )[0]
    del sess

    mask = enc2["attention_mask"].numpy().astype(bool)
    smoke_max_diff = float(np.max(np.abs(pt_logits[mask] - ox_logits[mask])))
    assert smoke_max_diff < 1e-2, f"スモークロジット差過大: {smoke_max_diff}"

    # 元モデルが一切変更されていないことを再アサート
    after_hashes = record_source_hashes(src_dir)
    assert after_hashes == source_hashes, "元モデルファイルに予期せぬ変更を検出しました"

    t_fp32 = time.time() - t0
    fp32_bytes = fp32_path.stat().st_size
    fp32_sha = sha256_file(fp32_path)

    info = {
        "model_type": cfg.model_type,
        "num_labels": int(cfg.num_labels),
        "export_method": export_method,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "opset": args.opset,
        "fp32_path": str(fp32_path).replace("\\", "/"),
        "fp32_bytes": int(fp32_bytes),
        "fp32_sha256": fp32_sha,
        "fp32_export_sec": round(t_fp32, 2),
        "ort_inputs": in_names,
        "ort_outputs": out_names,
        "smoke_batch2_max_logit_diff": smoke_max_diff,
        "int8_created": False,
        "source_model_dir": str(src_dir).replace("\\", "/"),
        "source_model_sha256": source_hashes,
    }
    meta_path = out_dir / "onnx_export_meta.json"
    meta_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
