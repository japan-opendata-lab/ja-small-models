#!/usr/bin/env python3
"""第4号 ja-doc-classify-v1: v2 モデル (final-hf) を単一自己完結 FP32 ONNX へ書き出すスクリプト。

実行は transformers==4.57.6 + onnx を導入した一時隔離環境 (.venv-onnx) を想定し、
runs/prod-modernbert-v2/final-hf/ は読み取り専用で一切変更しない。
INT8 量子化モデルは作成しない。
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

import numpy as np
import onnx
import onnxruntime as ort
import torch
import transformers
from safetensors.torch import load_file as load_safetensors
from torch import nn
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast


class OnnxMultiHeadWrapper(nn.Module):
    """ONNX 書き出し用の 3 ヘッド固定出力タプルラッパー。"""

    def __init__(self, cfg: ModernBertConfig, n_domain: int, n_type: int, n_format: int):
        super().__init__()
        self.encoder = ModernBertModel(cfg)
        h = cfg.hidden_size
        self.head_domain = nn.Linear(h, n_domain)
        self.head_type = nn.Linear(h, n_type)
        self.head_format = nn.Linear(h, n_format)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return (
            self.head_domain(pooled),
            self.head_type(pooled),
            self.head_format(pooled),
        )


def sha256_file(path: Path) -> str:
    """ファイルの SHA-256 ダイジェストを返す。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            blk = f.read(1024 * 1024)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def build_temp_compat_config(hf_dir: Path, tmp_dir: Path) -> ModernBertConfig:
    """元の final-hf を変更せず、transformers 4.57.x + eager ONNX 互換の config を一時生成して読み込む。"""
    raw_cfg = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    rp = raw_cfg.get("rope_parameters") or {}
    if "global_rope_theta" not in raw_cfg:
        raw_cfg["global_rope_theta"] = float(rp.get("full_attention", {}).get("rope_theta", 160000.0))
    if "local_rope_theta" not in raw_cfg:
        raw_cfg["local_rope_theta"] = float(rp.get("sliding_attention", {}).get("rope_theta", 10000.0))
    raw_cfg["attn_implementation"] = "eager"
    raw_cfg["reference_compile"] = False
    raw_cfg["use_cache"] = False
    raw_cfg["dtype"] = "float32"

    (tmp_dir / "config.json").write_text(json.dumps(raw_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg = ModernBertConfig.from_pretrained(str(tmp_dir), local_files_only=True)
    cfg.attn_implementation = "eager"
    cfg.reference_compile = False
    cfg.use_cache = False
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "model_fp32.onnx"

    routing_cfg = json.loads((hf_dir / "routing_config.json").read_text(encoding="utf-8"))
    n_domain = len(routing_cfg["domain_labels"])
    n_type = len(routing_cfg["type_labels"])
    n_format = len(routing_cfg["format_labels"])

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_cfg_dir = Path(tmp_root) / "cfg"
        tmp_exp_dir = Path(tmp_root) / "exp"
        tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
        tmp_exp_dir.mkdir(parents=True, exist_ok=True)

        cfg = build_temp_compat_config(hf_dir, tmp_cfg_dir)
        model = OnnxMultiHeadWrapper(cfg, n_domain=n_domain, n_type=n_type, n_format=n_format)
        sd = load_safetensors(str(hf_dir / "model.safetensors"), device="cpu")
        sd_f32 = {k: v.to(torch.float32) for k, v in sd.items()}
        model.load_state_dict(sd_f32, strict=True)
        model.cpu().eval()

        # batch=2（非パディング＋パディング混在）でトレース
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

        raw_onnx = tmp_exp_dir / "raw.onnx"
        torch.onnx.export(
            model,
            (dummy_ids, dummy_mask),
            str(raw_onnx),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits_domain", "logits_type", "logits_format"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits_domain": {0: "batch_size"},
                "logits_type": {0: "batch_size"},
                "logits_format": {0: "batch_size"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

        # 外部データ分割を統合し、単一自己完結ファイルとして保存
        onnx_model = onnx.load(str(raw_onnx), load_external_data=True)
        onnx.checker.check_model(onnx_model)
        onnx.save_model(onnx_model, str(fp32_path), save_as_external_data=False)

    # 外部データファイル・INT8 ファイル不在を検証
    assert not (out_dir / "model_fp32.onnx_data").exists()
    assert not (out_dir / "model_fp32.onnx.data").exists()
    assert not (out_dir / "model.onnx_data").exists()
    assert not any("int8" in p.name.lower() for p in out_dir.iterdir() if p.is_file())

    t_fp32 = time.time() - t0
    fp32_bytes = fp32_path.stat().st_size
    fp32_sha = sha256_file(fp32_path)

    # スモーク検証（短文・local_attention=128超の長文・batch=2パディング混在）
    # transformers 4.57.6 環境でも TokenizersBackend エラーを避けて直接 tokenizer.json を読み込む
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(hf_dir / "tokenizer.json"),
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        unk_token="<unk>",
        mask_token="<mask>",
        bos_token="<s>",
        eos_token="</s>",
    )
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]

    smoke_inputs = [
        ("short_b1", ["口座開設申込書の必要書類をご案内します。本人確認書類を1点ご提出ください。"]),
        (
            "long_b1_gt256",
            [
                "第1条（目的）本業務委託契約は、委託者が受託者に対してソフトウェア開発および保守運用業務を委託するにあたり、"
                "権利義務関係を明確に定めるものである。第2条（検収条件）受託者は納入物を期限までに納品し、委託者は10営業日以内に検査を行う。"
                * 14
            ],
        ),
        (
            "padded_b2",
            [
                "振込依頼書の控えをお送りいたします。金額と振込先口座をご確認ください。",
                "件名：退職手続書類のご提出依頼\n退職日までに健康保険証・社員証および貸与PCを人事労務部へご返却ください。" * 4,
            ],
        ),
    ]

    smoke_summary = {}
    for name, texts in smoke_inputs:
        enc = tok(texts, truncation=True, max_length=1024, padding=len(texts) > 1, return_tensors="pt")
        with torch.no_grad():
            pt_d, pt_t, pt_f = model(enc["input_ids"], enc["attention_mask"])
        ox_d, ox_t, ox_f = sess.run(
            None,
            {
                "input_ids": enc["input_ids"].numpy().astype(np.int64),
                "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
            },
        )
        diffs = {
            "logits_domain": round(float(np.abs(pt_d.numpy() - ox_d).max()), 6),
            "logits_type": round(float(np.abs(pt_t.numpy() - ox_t).max()), 6),
            "logits_format": round(float(np.abs(pt_f.numpy() - ox_f).max()), 6),
        }
        for k, v in diffs.items():
            assert v < 1e-3, f"Smoke diff too large on {name}.{k}: {v}"
        smoke_summary[name] = {
            "input_shape": list(enc["input_ids"].shape),
            "max_abs_diffs": diffs,
        }

    del sess
    info = {
        "transformers_export_env": transformers.__version__,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "opset": args.opset,
        "fp32_path": str(fp32_path).replace("\\", "/"),
        "fp32_bytes": int(fp32_bytes),
        "fp32_sha256": fp32_sha,
        "fp32_export_sec": round(t_fp32, 2),
        "ort_inputs": in_names,
        "ort_outputs": out_names,
        "int8_created": False,
        "smoke_check": smoke_summary,
    }
    meta_path = out_dir / "onnx_export_meta.json"
    meta_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
