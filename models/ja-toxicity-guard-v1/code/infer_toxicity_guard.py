#!/usr/bin/env python3
"""第8号 ja-toxicity-guard-v1（日本語有害・虐待表現の多ラベル検知）参照推論実装。

Hugging Face 形式（PyTorch safetensors / FP32 ONNX）の双方に対応し、
dev で固定したラベル別閾値（既定: inference_config.json）を適用して
7 ラベルの予測ラベル一覧および確率スコアを返す自己完結モジュール。

注意:
  本モデルはモデレーションの自動判定ではなく、人手レビューの優先付けおよび
  警告表示の補助に用いる部品として設計されている。

使い方（Python API）:
    from infer_toxicity_guard import JaToxicityGuard

    guard = JaToxicityGuard("path/to/model", backend="onnx")
    result = guard.predict("本日の議題について資料を共有します。")

使い方（CLI）:
    python infer_toxicity_guard.py --model-dir path/to/model --backend onnx --text "確認をお願いします。"
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

import numpy as np
from transformers import AutoTokenizer

LABELS = (
    "insult",
    "indirect_attack",
    "exposure_intent",
    "threat",
    "discrimination",
    "sexual_harassment",
    "self_harm",
)

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "insult": 0.5,
    "indirect_attack": 0.85,
    "exposure_intent": 0.5,
    "threat": 0.85,
    "discrimination": 0.5,
    "sexual_harassment": 0.85,
    "self_harm": 0.5,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """オーバーフローを防ぎつつ要素ごとのシグモイド確率を算出する。"""
    clipped = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def load_inference_config(model_dir: Path) -> dict:
    """モデルディレクトリから推論設定（閾値・最大長・ONNXパス）を読み込む。"""
    cfg_out = {
        "labels": list(LABELS),
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "max_length": 256,
        "onnx_path": "onnx/model.onnx",
    }

    inf_path = model_dir / "inference_config.json"
    if inf_path.exists():
        raw = json.loads(inf_path.read_text(encoding="utf-8"))
        if isinstance(raw.get("labels"), list) and len(raw["labels"]) == len(LABELS):
            cfg_out["labels"] = [str(lb) for lb in raw["labels"]]
        if isinstance(raw.get("thresholds"), dict):
            cfg_out["thresholds"] = {
                lb: float(raw["thresholds"].get(lb, DEFAULT_THRESHOLDS[lb]))
                for lb in cfg_out["labels"]
            }
        if raw.get("max_length") is not None:
            cfg_out["max_length"] = int(raw["max_length"])
        if raw.get("onnx_path"):
            cfg_out["onnx_path"] = str(raw["onnx_path"])
        return cfg_out

    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
        embedded = cfg_json.get("toxicity_guard_inference")
        if isinstance(embedded, dict):
            if isinstance(embedded.get("thresholds"), dict):
                cfg_out["thresholds"] = {
                    lb: float(embedded["thresholds"].get(lb, DEFAULT_THRESHOLDS[lb]))
                    for lb in cfg_out["labels"]
                }
            if embedded.get("max_length") is not None:
                cfg_out["max_length"] = int(embedded["max_length"])

    return cfg_out


class JaToxicityGuard:
    """7ラベル多ラベル有害表現検知器の統一ラッパー（ONNX / PyTorch 両対応）。"""

    def __init__(
        self,
        model_dir: str | Path,
        backend: str = "onnx",
        thresholds: Optional[Dict[str, float]] = None,
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
            raise ValueError(f"未対応のバックエンドです: {backend} ('onnx' または 'pytorch' を指定)")

        loaded_cfg = load_inference_config(self.model_dir)
        self.labels: List[str] = list(loaded_cfg["labels"])
        self.thresholds: Dict[str, float] = dict(loaded_cfg["thresholds"])
        if thresholds is not None:
            for lb in self.labels:
                if lb in thresholds:
                    self.thresholds[lb] = float(thresholds[lb])

        self.max_length: int = int(max_length if max_length is not None else loaded_cfg["max_length"])
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=local_files_only
        )

        self.ort_session = None
        self.ort_input_names: List[str] = []
        self.pt_model = None

        if self.backend == "onnx":
            import onnxruntime as ort

            rel_or_abs = Path(onnx_path) if onnx_path is not None else Path(loaded_cfg["onnx_path"])
            candidate = rel_or_abs if rel_or_abs.is_absolute() else (self.model_dir / rel_or_abs)
            if not candidate.exists():
                fallback = self.model_dir / "model.onnx"
                if fallback.exists():
                    candidate = fallback
            if not candidate.exists():
                raise FileNotFoundError(f"ONNX モデルファイルが見つかりません: {candidate}")

            so = ort.SessionOptions()
            if intra_op_num_threads > 0:
                so.intra_op_num_threads = int(intra_op_num_threads)
            if inter_op_num_threads > 0:
                so.inter_op_num_threads = int(inter_op_num_threads)
            self.ort_session = ort.InferenceSession(
                str(candidate), sess_options=so, providers=["CPUExecutionProvider"]
            )
            self.ort_input_names = [inp.name for inp in self.ort_session.get_inputs()]
        else:
            import torch
            from transformers import AutoModelForSequenceClassification

            self._torch = torch
            self.pt_model = (
                AutoModelForSequenceClassification.from_pretrained(
                    str(self.model_dir),
                    local_files_only=local_files_only,
                    dtype=torch.float32,
                )
                .to(self.device)
                .eval()
            )

    def _build_ort_feed(self, enc: dict) -> Dict[str, np.ndarray]:
        """トークナイザ出力から ONNX Runtime 用の int64 入力辞書を組み立てる。"""
        ids = np.asarray(enc["input_ids"], dtype=np.int64)
        feed: Dict[str, np.ndarray] = {}
        for name in self.ort_input_names:
            if name in enc:
                feed[name] = np.asarray(enc[name], dtype=np.int64)
            elif name == "token_type_ids":
                feed[name] = np.zeros_like(ids, dtype=np.int64)
            else:
                raise KeyError(f"未対応の ONNX 入力名: {name}")
        return feed

    def predict_logits(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """テキスト列に対する生ロジット配列 shape=(N, 7) を返す。"""
        if not texts:
            return np.zeros((0, len(self.labels)), dtype=np.float32)

        chunks: List[np.ndarray] = []
        for start in range(0, len(texts), max(1, int(batch_size))):
            batch = list(texts[start : start + max(1, int(batch_size))])
            if self.backend == "onnx":
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="np",
                )
                logits = self.ort_session.run(None, self._build_ort_feed(enc))[0]
                chunks.append(np.asarray(logits, dtype=np.float32))
            else:
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                kw = {k: v.to(self.device) for k, v in enc.items()}
                with self._torch.no_grad():
                    logits = self.pt_model(**kw).logits.detach().cpu().float().numpy()
                chunks.append(logits)

        return np.vstack(chunks)

    def predict_batch(self, texts: Sequence[str], batch_size: int = 32) -> List[dict]:
        """複数テキストに対して閾値適用済みラベルと各ラベル確率スコアを返す。"""
        if not texts:
            return []
        logits = self.predict_logits(texts, batch_size=batch_size)
        probs = _sigmoid(logits)
        thr_snapshot = {lb: float(self.thresholds[lb]) for lb in self.labels}

        results: List[dict] = []
        for row_prob in probs:
            pred_labels = [
                lb for j, lb in enumerate(self.labels) if float(row_prob[j]) >= thr_snapshot[lb]
            ]
            results.append(
                {
                    "labels": pred_labels,
                    "is_toxic": bool(len(pred_labels) > 0),
                    "max_prob": round(float(row_prob.max()), 6),
                    "scores": {
                        lb: round(float(row_prob[j]), 6) for j, lb in enumerate(self.labels)
                    },
                    "thresholds": dict(thr_snapshot),
                }
            )
        return results

    def predict(self, text: str) -> dict:
        """単一テキストに対して閾値適用済みラベルと確率スコアを返す。"""
        return self.predict_batch([text], batch_size=1)[0]


def _read_texts_from_args(args: argparse.Namespace) -> List[str]:
    texts: List[str] = []
    if args.text:
        texts.extend(args.text)
    if args.stdin:
        for line in sys.stdin:
            s = line.rstrip("\r\n")
            if s:
                texts.append(s)
    return texts


def main() -> int:
    ap = argparse.ArgumentParser(description="第8号 ja-toxicity-guard-v1 参照推論 CLI")
    ap.add_argument("--model-dir", required=True, help="モデルディレクトリ（release-staging 等）")
    ap.add_argument(
        "--backend",
        choices=("onnx", "pytorch"),
        default="onnx",
        help="推論バックエンド（既定: onnx）",
    )
    ap.add_argument("--text", action="append", help="推論対象テキスト（複数回指定可）")
    ap.add_argument("--stdin", action="store_true", help="標準入力から1行1文で読み込む")
    ap.add_argument("--batch-size", type=int, default=32, help="バッチサイズ（既定: 32）")
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="ローカルファイルのみからトークナイザ・モデルを読み込む",
    )
    args = ap.parse_args()

    texts = _read_texts_from_args(args)
    if not texts:
        ap.error("--text または --stdin で入力テキストを指定してください。")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    guard = JaToxicityGuard(
        args.model_dir,
        backend=args.backend,
        local_files_only=args.local_files_only,
    )
    outputs = guard.predict_batch(texts, batch_size=args.batch_size)
    payload = outputs[0] if (len(outputs) == 1 and not args.stdin) else outputs
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
