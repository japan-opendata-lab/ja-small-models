#!/usr/bin/env python3
"""第11号 ja-nli-v1（日本語 3 クラス自然言語推論）自己完結型 参照推論実装。

Hugging Face 形式（PyTorch safetensors / FP32 ONNX）の双方に対応し、
前提（premise）と仮説（hypothesis）のペアに対して 3 クラス
（entailment / neutral / contradiction）の確率分布および予測ラベルを返す。

使い方（Python API）:
    from infer_nli import JaNLIClassifier

    clf = JaNLIClassifier("path/to/model", backend="onnx", local_files_only=True)
    result = clf.predict(
        "東京駅は東京都千代田区に位置する主要な鉄道駅である。",
        "東京駅は東京都千代田区にある。",
    )
    print(result["label"], result["probabilities"])

使い方（CLI）:
    python infer_nli.py --model-dir path/to/model --backend onnx \\
        --premise "東京駅は東京都千代田区に位置する主要な鉄道駅である。" \\
        --hypothesis "東京駅は東京都千代田区にある。"
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
from transformers import AutoTokenizer

CANONICAL_LABELS = ("entailment", "neutral", "contradiction")
DEFAULT_ID2LABEL: Dict[int, str] = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}


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
        "max_length": 1024,
        "truncation": "only_first",
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
        if raw.get("truncation") not in (None, False):
            trunc_val = raw["truncation"]
            cfg_out["truncation"] = "only_first" if trunc_val is True else str(trunc_val)
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


class JaNLIClassifier:
    """日本語 3 クラス NLI モデルの自己完結ラッパー（ONNX / PyTorch 両対応）。"""

    def __init__(
        self,
        model_dir: str | Path,
        backend: str = "onnx",
        max_length: Optional[int] = None,
        truncation: Optional[str | bool] = None,
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
        self.truncation: str | bool = (
            truncation if truncation is not None else loaded_cfg["truncation"]
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

    def _tokenize_batch(
        self,
        premises: List[str],
        hypotheses: List[str],
        padding: bool,
        return_tensors: str,
    ):
        """前提末尾優先切り詰めでトークナイズし、仮説単独超過時のみ両側切り詰めへ退避する。"""
        try:
            return self.tokenizer(
                premises,
                hypotheses,
                truncation=self.truncation,
                max_length=self.max_length,
                padding=padding,
                return_tensors=return_tensors,
            )
        except ValueError:
            return self.tokenizer(
                premises,
                hypotheses,
                truncation=True,
                max_length=self.max_length,
                padding=padding,
                return_tensors=return_tensors,
            )

    def predict_logits(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: int = 16,
    ) -> np.ndarray:
        """複数ペアをバッチ推論し、形状 (N, 3) の FP32 ロジット配列を返す。"""
        pair_list = [(str(p), str(h)) for p, h in pairs]
        if not pair_list:
            return np.zeros((0, len(self.labels)), dtype=np.float32)

        logits_chunks: List[np.ndarray] = []
        for start in range(0, len(pair_list), batch_size):
            batch = pair_list[start : start + batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]
            padding = len(batch) > 1

            if self.backend == "onnx":
                enc = self._tokenize_batch(premises, hypotheses, padding=padding, return_tensors="np")
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
                enc = self._tokenize_batch(premises, hypotheses, padding=padding, return_tensors="pt")
                enc_dev = {k: v.to(self.device) for k, v in enc.items()}
                with torch.inference_mode():
                    logits = self.pt_model(**enc_dev).logits.detach().cpu().numpy()
                logits_chunks.append(np.asarray(logits, dtype=np.float32))

        return np.concatenate(logits_chunks, axis=0)

    def predict_batch(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: int = 16,
    ) -> List[dict]:
        """複数ペアの予測ラベルおよび 3 クラス確率辞書リストを返す。"""
        pair_list = [(str(p), str(h)) for p, h in pairs]
        if not pair_list:
            return []

        logits = self.predict_logits(pair_list, batch_size=batch_size)
        probs_mat = _softmax(logits)
        results: List[dict] = []
        for row_idx in range(len(pair_list)):
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

    def predict(self, premise: str, hypothesis: str) -> dict:
        """単一の前提・仮説ペアに対して予測ラベルと 3 クラス確率を返す。"""
        return self.predict_batch([(premise, hypothesis)], batch_size=1)[0]


def _default_model_dir() -> Path:
    """examples/ 配下から実行された場合にステージングルートを既定のモデルディレクトリにする。"""
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "config.json").exists() or (candidate / "inference_config.json").exists():
        return candidate
    return Path(".")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """コマンドライン推論エントリポイント。"""
    ap = argparse.ArgumentParser(description="ja-nli-v1 3クラス日本語 NLI 推論 CLI")
    ap.add_argument("--model-dir", default=str(_default_model_dir()), help="モデル格納ディレクトリ")
    ap.add_argument("--backend", choices=["onnx", "pytorch"], default="onnx")
    ap.add_argument("--premise", default=None, help="単発推論用の前提文")
    ap.add_argument("--hypothesis", default=None, help="単発推論用の仮説文")
    ap.add_argument(
        "--jsonl",
        default=None,
        help="1行1ペア（{'premise': ..., 'hypothesis': ...}）の UTF-8 JSONL ファイル",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--json", action="store_true", help="JSON形式で出力する")
    args = ap.parse_args(argv)

    pairs: List[Tuple[str, str]] = []
    if args.premise is not None or args.hypothesis is not None:
        if args.premise is None or args.hypothesis is None:
            ap.error("--premise と --hypothesis は必ず対で指定してください。")
        pairs.append((args.premise, args.hypothesis))

    if args.jsonl:
        with open(args.jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    pairs.append((str(row["premise"]), str(row["hypothesis"])))

    if not pairs:
        ap.error("(--premise と --hypothesis) または --jsonl を指定してください。")

    clf = JaNLIClassifier(
        model_dir=args.model_dir,
        backend=args.backend,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )
    preds = clf.predict_batch(pairs, batch_size=args.batch_size)

    if args.json:
        payload = [
            {"premise": p, "hypothesis": h, **res}
            for (p, h), res in zip(pairs, preds)
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for (p, h), res in zip(pairs, preds):
            pr = res["probabilities"]
            print(
                f"[{res['label']} ({res['score']:.4f})] "
                f"E={pr['entailment']:.4f} N={pr['neutral']:.4f} C={pr['contradiction']:.4f} | "
                f"P={p[:40]}... / H={h}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
