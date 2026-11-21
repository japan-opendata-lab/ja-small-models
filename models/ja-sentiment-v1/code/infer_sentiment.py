#!/usr/bin/env python3
"""第12号 ja-sentiment-v1（日本語テキスト3クラス感情極性分類）自己完結型 参照推論実装。

Hugging Face 形式（PyTorch safetensors / FP32 ONNX）の双方に対応し、
入力テキストに対して 3 クラス（negative / neutral / positive）の確率分布および
予測ラベルを返す自己完結モジュール。

使い方（Python API）:
    from infer_sentiment import JaSentimentClassifier

    clf = JaSentimentClassifier("path/to/model", backend="onnx", local_files_only=True)
    result = clf.predict("とても使いやすく大満足です。毎日愛用しています。")
    print(result["label"], result["probabilities"])

使い方（CLI）:
    python infer_sentiment.py --model-dir path/to/model --backend onnx --text "確認をお願いします。"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
from transformers import AutoTokenizer

CANONICAL_LABELS = ("negative", "neutral", "positive")
DEFAULT_ID2LABEL: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}


def _softmax(logits: np.ndarray) -> np.ndarray:
    """数値安定性を保ちつつ行ごとのソフトマックス確率（総和 1.0）を算出する。"""
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - np.max(arr, axis=-1, keepdims=True)
    exp_arr = np.exp(shifted)
    return exp_arr / np.sum(exp_arr, axis=-1, keepdims=True)


def load_inference_config(model_dir: Path) -> dict:
    """モデルディレクトリから inference_config.json（不在時は config.json）を読み込む。"""
    cfg_out = {
        "labels": list(CANONICAL_LABELS),
        "id2label": dict(DEFAULT_ID2LABEL),
        "max_length": 512,
        "onnx_path": "onnx/model.onnx",
    }

    inf_path = model_dir / "inference_config.json"
    if inf_path.exists():
        raw = json.loads(inf_path.read_text(encoding="utf-8"))
        if isinstance(raw.get("id2label"), dict) and len(raw["id2label"]) == 3:
            id2label = {int(k): str(v) for k, v in raw["id2label"].items()}
            cfg_out["id2label"] = id2label
            cfg_out["labels"] = [id2label[i] for i in sorted(id2label.keys())]
        elif isinstance(raw.get("labels"), list) and len(raw["labels"]) == 3:
            labels = [str(lb) for lb in raw["labels"]]
            cfg_out["labels"] = labels
            cfg_out["id2label"] = {i: lb for i, lb in enumerate(labels)}
        if raw.get("max_length") is not None:
            cfg_out["max_length"] = int(raw["max_length"])
        if raw.get("onnx_path"):
            cfg_out["onnx_path"] = str(raw["onnx_path"])
        return cfg_out

    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(raw_cfg.get("id2label"), dict) and len(raw_cfg["id2label"]) == 3:
            id2label = {int(k): str(v) for k, v in raw_cfg["id2label"].items()}
            cfg_out["id2label"] = id2label
            cfg_out["labels"] = [id2label[i] for i in sorted(id2label.keys())]

    return cfg_out


class JaSentimentClassifier:
    """3クラス感情極性分類器の自己完結ラッパー（ONNX / PyTorch 両対応）。"""

    def __init__(
        self,
        model_dir: str | Path,
        backend: str = "onnx",
        max_length: Optional[int] = None,
        onnx_path: Optional[str | Path] = None,
        local_files_only: bool = False,
        intra_op_num_threads: int = 0,
        inter_op_num_threads: int = 1,
        device: str = "cpu",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.backend = backend.lower().strip()
        if self.backend not in ("onnx", "pytorch"):
            raise ValueError(
                f"未対応のバックエンドです: {backend} ('onnx' または 'pytorch' を指定)"
            )

        loaded_cfg = load_inference_config(self.model_dir)
        self.id2label: Dict[int, str] = dict(loaded_cfg["id2label"])
        self.labels: List[str] = [self.id2label[i] for i in sorted(self.id2label.keys())]
        self.max_length: int = int(
            max_length if max_length is not None else loaded_cfg["max_length"]
        )
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=local_files_only
        )

        self.ort_session = None
        self.ort_input_names: List[str] = []
        self.pt_model = None
        self._torch = None

        if self.backend == "onnx":
            import onnxruntime as ort

            rel_or_abs = Path(onnx_path) if onnx_path is not None else Path(loaded_cfg["onnx_path"])
            candidate = rel_or_abs if rel_or_abs.is_absolute() else (self.model_dir / rel_or_abs)
            if not candidate.exists():
                for fb_rel in ("onnx/model.onnx", "model.onnx", "model_fp32.onnx"):
                    fb = self.model_dir / fb_rel
                    if fb.exists():
                        candidate = fb
                        break
            if not candidate.exists():
                raise FileNotFoundError(f"ONNX モデルファイルが見つかりません: {candidate}")

            opts = ort.SessionOptions()
            if intra_op_num_threads > 0:
                opts.intra_op_num_threads = int(intra_op_num_threads)
            if inter_op_num_threads > 0:
                opts.inter_op_num_threads = int(inter_op_num_threads)

            self.ort_session = ort.InferenceSession(
                str(candidate),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.ort_input_names = [inp.name for inp in self.ort_session.get_inputs()]
        else:
            import torch
            from transformers import AutoModelForSequenceClassification

            self._torch = torch
            try:
                self.pt_model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.model_dir),
                    local_files_only=local_files_only,
                    dtype=torch.float32,
                    attn_implementation="eager",
                    reference_compile=False,
                )
            except TypeError:
                self.pt_model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.model_dir),
                    local_files_only=local_files_only,
                    torch_dtype=torch.float32,
                    attn_implementation="eager",
                )
            self.pt_model.to(self.device).eval()

    def predict_logits(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """複数テキストをバッチ推論し、形状 (N, 3) の FP32 ロジット配列を返す。"""
        text_list = [str(t) for t in texts]
        if not text_list:
            return np.zeros((0, len(self.labels)), dtype=np.float32)

        logits_chunks: List[np.ndarray] = []
        for start in range(0, len(text_list), batch_size):
            batch = text_list[start : start + batch_size]
            if self.backend == "onnx":
                enc = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=self.max_length,
                    padding=len(batch) > 1,
                    return_tensors="np",
                )
                feed = {
                    name: np.asarray(enc[name], dtype=np.int64)
                    for name in self.ort_input_names
                    if name in enc
                }
                if "token_type_ids" in self.ort_input_names and "token_type_ids" not in feed:
                    feed["token_type_ids"] = np.zeros_like(feed["input_ids"], dtype=np.int64)
                out = self.ort_session.run(None, feed)[0]
                logits_chunks.append(np.asarray(out, dtype=np.float32))
            else:
                torch = self._torch
                enc = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=self.max_length,
                    padding=len(batch) > 1,
                    return_tensors="pt",
                )
                enc_dev = {k: v.to(self.device) for k, v in enc.items()}
                with torch.inference_mode():
                    logits = self.pt_model(**enc_dev).logits.detach().cpu().numpy()
                logits_chunks.append(np.asarray(logits, dtype=np.float32))

        return np.concatenate(logits_chunks, axis=0)

    def predict_batch(self, texts: Sequence[str], batch_size: int = 32) -> List[dict]:
        """複数テキストの予測ラベルおよび 3 クラス確率辞書リストを返す。"""
        text_list = [str(t) for t in texts]
        if not text_list:
            return []

        logits = self.predict_logits(text_list, batch_size=batch_size)
        probs_mat = _softmax(logits)
        results: List[dict] = []
        for row_idx in range(len(text_list)):
            probs_row = probs_mat[row_idx]
            pred_id = int(np.argmax(probs_row))
            pred_label = self.id2label[pred_id]
            prob_dict = {
                self.id2label[idx]: float(probs_row[idx])
                for idx in sorted(self.id2label.keys())
            }
            results.append(
                {
                    "label": pred_label,
                    "label_id": pred_id,
                    "score": float(probs_row[pred_id]),
                    "probabilities": prob_dict,
                }
            )
        return results

    def predict(self, text: str) -> dict:
        """単一テキストの予測ラベルと 3 クラス確率を返す。"""
        return self.predict_batch([text], batch_size=1)[0]


def _default_model_dir() -> Path:
    """examples/ 配下から実行された場合にステージングルートを既定のモデルディレクトリにする。"""
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "config.json").exists() or (candidate / "inference_config.json").exists():
        return candidate
    return Path(".")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """コマンドライン推論エントリポイント。"""
    ap = argparse.ArgumentParser(description="ja-sentiment-v1 3クラス感情極性分類 推論 CLI")
    ap.add_argument("--model-dir", default=str(_default_model_dir()), help="モデル格納ディレクトリ")
    ap.add_argument("--backend", choices=["onnx", "pytorch"], default="onnx")
    ap.add_argument("--text", action="append", default=[], help="推論対象テキスト（複数指定可）")
    ap.add_argument("--file", default=None, help="1行1テキストのUTF-8入力ファイル")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--json", action="store_true", help="JSON形式で出力する")
    args = ap.parse_args(argv)

    texts: List[str] = list(args.text)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts.extend(ln.rstrip("\r\n") for ln in f if ln.strip())

    if not texts:
        ap.error("--text または --file を少なくとも1つ指定してください。")

    clf = JaSentimentClassifier(
        model_dir=args.model_dir,
        backend=args.backend,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )
    preds = clf.predict_batch(texts, batch_size=args.batch_size)

    if args.json:
        payload = [
            {"text": t, **p}
            for t, p in zip(texts, preds)
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for t, p in zip(texts, preds):
            pr = p["probabilities"]
            print(
                f"[{p['label']} ({p['score']:.4f})] "
                f"neg={pr['negative']:.4f} neu={pr['neutral']:.4f} pos={pr['positive']:.4f} | {t}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
