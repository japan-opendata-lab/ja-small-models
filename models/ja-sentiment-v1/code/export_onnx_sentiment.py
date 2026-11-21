#!/usr/bin/env python3
"""第12号 ja-sentiment-v1 確定モデル (runs/prod-modernbert/best) の FP32 ONNX 書き出しスクリプト。

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
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

QUANTIZED_OP_TYPES = {
    "QuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
    "MatMulInteger",
    "QLinearMatMul",
    "QLinearConv",
    "QAttention",
    "QGemm",
    "DynamicQuantizeLSTM",
    "DynamicQuantizeMatMul",
}


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


def prepare_temp_source_dir(src_model_dir: Path, tmp_src_dir: Path) -> None:
    """元モデルを一切変更せず、ONNX 書き出し互換設定を施した一時読込ディレクトリを作成する。"""
    tmp_src_dir.mkdir(parents=True, exist_ok=True)
    for item in src_model_dir.iterdir():
        if not item.is_file():
            continue
        if item.name == "config.json":
            cfg = json.loads(item.read_text(encoding="utf-8"))
            rp = cfg.get("rope_parameters") or {}
            if "global_rope_theta" not in cfg:
                cfg["global_rope_theta"] = float(
                    rp.get("full_attention", {}).get("rope_theta", 160000.0)
                )
            if "local_rope_theta" not in cfg:
                cfg["local_rope_theta"] = float(
                    rp.get("sliding_attention", {}).get("rope_theta", 10000.0)
                )
            cfg["attn_implementation"] = "eager"
            cfg["reference_compile"] = False
            cfg["use_cache"] = False
            cfg["dtype"] = "float32"
            (tmp_src_dir / "config.json").write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            shutil.copy2(item, tmp_src_dir / item.name)


def load_tokenizer_compat(src_dir: Path):
    """transformers 5.x/4.x 互換でローカルからトークナイザを読み込む。"""
    try:
        return AutoTokenizer.from_pretrained(str(src_dir), local_files_only=True)
    except Exception:
        return PreTrainedTokenizerFast(
            tokenizer_file=str(src_dir / "tokenizer.json"),
            pad_token="<pad>",
            cls_token="<cls>",
            sep_token="<sep>",
            unk_token="<unk>",
            mask_token="<mask>",
            bos_token="<s>",
            eos_token="</s>",
        )


def load_model_fp32(tmp_src: Path):
    """一時ディレクトリから FP32 モデルを読み込み、アーキテクチャ・ラベル対応を検証する。"""
    cfg = AutoConfig.from_pretrained(str(tmp_src), local_files_only=True)
    cfg.attn_implementation = "eager"
    cfg.reference_compile = False
    cfg.use_cache = False

    assert cfg.model_type == "modernbert", f"想定外の model_type: {cfg.model_type}"
    assert int(cfg.num_labels) == 3, f"想定外の num_labels: {cfg.num_labels}"
    id2label_norm = {int(k): str(v) for k, v in cfg.id2label.items()}
    assert id2label_norm == {0: "negative", 1: "neutral", 2: "positive"}, (
        f"想定外の id2label: {id2label_norm}"
    )

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(tmp_src),
            config=cfg,
            attn_implementation="eager",
            dtype=torch.float32,
            local_files_only=True,
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(tmp_src),
            config=cfg,
            attn_implementation="eager",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
    model.cpu().eval()
    return cfg, model


def verify_onnx_graph(onnx_model: onnx.ModelProto, expected_opset: int) -> int:
    """ONNX モデルの opset・量子化ノード不在・仕様適合を検証し、量子化ノード数を返す。"""
    onnx.checker.check_model(onnx_model)

    default_opsets = [
        op.version
        for op in onnx_model.opset_import
        if op.domain in ("", "ai.onnx")
    ]
    assert default_opsets and int(default_opsets[0]) == expected_opset, (
        f"opset 不一致: expected={expected_opset}, got={default_opsets}"
    )

    quant_ops = [
        node.op_type
        for node in onnx_model.graph.node
        if node.op_type in QUANTIZED_OP_TYPES
    ]
    assert len(quant_ops) == 0, f"量子化演算が検出されました: {set(quant_ops)}"
    return len(quant_ops)


def main() -> int:
    ap = argparse.ArgumentParser(description="第12号 ja-sentiment-v1 FP32 ONNX 書き出し")
    ap.add_argument("--model-dir", default="runs/prod-modernbert/best")
    ap.add_argument("--out-dir", default="onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    src_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "model_fp32.onnx"

    # 元モデルの事前ハッシュを記録（以降一切変更しない）
    source_hashes = record_source_hashes(src_dir)
    assert len(source_hashes) >= 4, f"元モデルファイル不足: {list(source_hashes.keys())}"

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_src = Path(tmp_root) / "src"
        tmp_exp = Path(tmp_root) / "exp"
        tmp_exp.mkdir(parents=True, exist_ok=True)
        prepare_temp_source_dir(src_dir, tmp_src)

        tok = load_tokenizer_compat(tmp_src)
        _, model = load_model_fp32(tmp_src)

        # batch=2（パディングなし＋パディングあり混在）でトレース
        # ModernBERT special tokens: cls=6, sep=4, pad=3
        dummy_ids = torch.tensor(
            [
                [6, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 4],
                [6, 101, 202, 303, 404, 505, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            ],
            dtype=torch.long,
        )
        dummy_mask = torch.tensor(
            [
                [1] * 16,
                [1] * 7 + [0] * 9,
            ],
            dtype=torch.long,
        )

        raw_onnx = tmp_exp / "raw.onnx"
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_ids, dummy_mask),
                str(raw_onnx),
                dynamo=False,
                opset_version=args.opset,
                do_constant_folding=True,
                export_params=True,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size"},
                },
            )

        assert raw_onnx.exists(), f"書き出し後の {raw_onnx} が見つかりません"
        onnx_model = onnx.load(str(raw_onnx), load_external_data=True)
        quant_ops_count = verify_onnx_graph(onnx_model, args.opset)
        onnx.save_model(onnx_model, str(fp32_path), save_as_external_data=False)

        # 外部データファイル・INT8 ファイル不在を検証
        assert not (out_dir / "model_fp32.onnx_data").exists()
        assert not (out_dir / "model_fp32.onnx.data").exists()
        assert not (out_dir / "model.onnx_data").exists()
        assert not any("int8" in p.name.lower() for p in out_dir.iterdir() if p.is_file())

        t_fp32 = time.time() - t0
        fp32_bytes = fp32_path.stat().st_size
        fp32_sha = sha256_file(fp32_path)

        sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        in_names = [i.name for i in sess.get_inputs()]
        out_names = [o.name for o in sess.get_outputs()]
        assert in_names == ["input_ids", "attention_mask"], f"入力名不一致: {in_names}"
        assert out_names == ["logits"], f"出力名不一致: {out_names}"

        smoke_cases = [
            ("short_b1", ["とても使いやすく大満足です。毎日愛用しています。"]),
            (
                "long_b1_gt128",
                [
                    "デザインの質感は期待どおりで、梱包も非常に丁寧でした。"
                    "ただ初期設定の手順書が少し分かりにくく、接続完了まで二十分ほど迷いました。"
                    "普段使いの動作自体は静かで安定しているため、価格を考えれば概ね納得できる買い物です。"
                    * 8
                ],
            ),
            (
                "padded_b2",
                [
                    "昨日届き、サイズは幅30センチでした。",
                    "正直あまり期待していませんでしたが、実際に使ってみると動作が軽くバッテリー持ちも良いため、今は仕事でも自宅でも毎日手放せません。",
                ],
            ),
        ]

        smoke_summary = {}
        for case_name, texts in smoke_cases:
            enc = tok(
                texts,
                truncation=True,
                max_length=512,
                padding=len(texts) > 1,
                return_tensors="pt",
            )
            with torch.no_grad():
                pt_logits = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                ).logits.numpy()
            ox_logits = sess.run(
                None,
                {
                    "input_ids": enc["input_ids"].numpy().astype(np.int64),
                    "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
                },
            )[0]

            assert ox_logits.shape == (len(texts), 3), f"出力 shape 異常: {ox_logits.shape}"
            max_diff = float(np.max(np.abs(pt_logits - ox_logits)))
            pt_pred = np.argmax(pt_logits, axis=-1).tolist()
            ox_pred = np.argmax(ox_logits, axis=-1).tolist()
            assert max_diff < 1e-3, f"スモーク誤差過大 ({case_name}): {max_diff}"
            assert pt_pred == ox_pred, f"スモーク予測不一致 ({case_name}): {pt_pred} != {ox_pred}"

            smoke_summary[case_name] = {
                "input_shape": [int(x) for x in enc["input_ids"].shape],
                "max_abs_logit_diff": round(max_diff, 8),
                "pt_pred_ids": pt_pred,
                "ox_pred_ids": ox_pred,
                "argmax_match": True,
            }
        del sess

    # 元モデル不変検証
    after_hashes = record_source_hashes(src_dir)
    assert after_hashes == source_hashes, "元モデルディレクトリのハッシュが変化しました！"

    info = {
        "source_model_dir": str(args.model_dir).replace("\\", "/"),
        "transformers_export_env": transformers.__version__,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "task": "text-classification",
        "problem_type": "single_label_classification",
        "num_labels": 3,
        "opset": args.opset,
        "precision": "fp32",
        "fp32_path": str(fp32_path).replace("\\", "/"),
        "fp32_bytes": int(fp32_bytes),
        "fp32_sha256": fp32_sha,
        "fp32_export_sec": round(t_fp32, 2),
        "ort_inputs": in_names,
        "ort_outputs": out_names,
        "quantized_ops_count": int(quant_ops_count),
        "int8_created": False,
        "smoke_check": smoke_summary,
        "source_model_sha256": source_hashes,
    }
    meta_path = out_dir / "onnx_export_meta.json"
    meta_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
