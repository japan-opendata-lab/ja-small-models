#!/usr/bin/env python
"""日本語テキスト誤り検出モデル（ja-error-detect-v1）の参照推論実装。

Hugging Face 形式（PyTorch safetensors / FP32 ONNX）の双方に対応し、
文字オフセット整列（BertJapaneseTokenizer での NFKC 逆写像および [UNK] 保護を含む）と
確信度閾値（既定: span_mean_err >= 0.80）を適用して誤り文字スパンを検出する。

使い方（Python API）:
    from infer_error_detect import JaErrorDetector
    detector = JaErrorDetector("path/to/model", backend="onnx")
    spans = detector.detect("大きな間題が発生した。")

使い方（CLI）:
    python infer_error_detect.py --model-dir path/to/model --backend onnx --text "大きな間題が発生した。"
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from transformers import AutoConfig, AutoTokenizer

LABELS = ("O", "B-ERR", "I-ERR")
ID2LABEL = {0: "O", 1: "B-ERR", 2: "I-ERR"}


def _clusters(text: str) -> List[str]:
    """書記素クラスタの近似分割（基底文字＋結合文字・半角濁点）。"""
    clusters: List[str] = []
    cur = ""
    for ch in text:
        if cur and (
            unicodedata.combining(ch)
            or unicodedata.category(ch) in ("Mn", "Me", "Mc")
            or ch in ("\uff9e", "\uff9f")
        ):
            cur += ch
        else:
            if cur:
                clusters.append(cur)
            cur = ch
    if cur:
        clusters.append(cur)
    return clusters


def _nfkc_with_map(text: str) -> Tuple[str, List[int], List[int]]:
    """書記素クラスタ単位で NFKC 正規化を行い、元テキスト位置への逆対応表を返す。"""
    norm_parts: List[str] = []
    start_map: List[int] = []
    end_map: List[int] = []
    pos = 0
    for cl in _clusters(text):
        n = unicodedata.normalize("NFKC", cl)
        if not n:
            pos += len(cl)
            continue
        for _ in n:
            start_map.append(pos)
            end_map.append(pos + len(cl))
        norm_parts.append(n)
        pos += len(cl)
    return "".join(norm_parts), start_map, end_map


def _encode_slow_bert_japanese(
    tok, text: str, max_len: int
) -> Tuple[dict, List[Tuple[int, int]]]:
    """BertJapaneseTokenizer (is_fast=False) 向けのオフセット付きエンコード。"""
    norm, start_map, end_map = _nfkc_with_map(text)
    specials = sorted(tok.all_special_tokens, key=len, reverse=True)
    sp_re = (
        re.compile("(" + "|".join(re.escape(s) for s in specials) + ")")
        if specials
        else None
    )
    segments = sp_re.split(norm) if sp_re else [norm]
    sp_set = set(specials)
    unk = tok.unk_token

    tokens: List[str] = []
    offsets: List[Tuple[int, int]] = []
    pos = 0

    for seg in segments:
        if not seg:
            continue
        if seg in sp_set:
            idx = norm.find(seg, pos)
            if idx >= 0:
                pos = idx + len(seg)
            tokens.append(seg)
            offsets.append((0, 0))
            continue

        words = tok.word_tokenizer.tokenize(seg, never_split=tok.all_special_tokens)
        for word in words:
            idx = norm.find(word, pos)
            if idx < 0:
                for sw in tok.subword_tokenizer.tokenize(word):
                    tokens.append(sw)
                    offsets.append((0, 0))
                continue

            word_start_orig = start_map[idx]
            word_end_orig = end_map[idx + len(word) - 1]
            subwords = tok.subword_tokenizer.tokenize(word)

            if subwords == [unk]:
                tokens.append(unk)
                offsets.append((word_start_orig, word_end_orig))
            else:
                wpos = 0
                for sw in subwords:
                    surf = sw[2:] if sw.startswith("##") else sw
                    sub_pos = word.find(surf, wpos)
                    if sub_pos < 0:
                        tokens.append(sw)
                        offsets.append((word_start_orig, word_end_orig))
                    else:
                        s_ni = idx + sub_pos
                        e_ni = s_ni + len(surf) - 1
                        tokens.append(sw)
                        offsets.append((start_map[s_ni], end_map[e_ni]))
                        wpos = sub_pos + len(surf)

            pos = idx + len(word)

    max_body = max(0, max_len - tok.num_special_tokens_to_add(pair=False))
    tokens = tokens[:max_body]
    offsets = offsets[:max_body]

    ids = tok.build_inputs_with_special_tokens(tok.convert_tokens_to_ids(tokens))
    n_special = len(ids) - len(tokens)
    if n_special == 2:
        offsets = [(0, 0)] + offsets + [(0, 0)]
    elif n_special == 1:
        offsets = offsets + [(0, 0)]
    elif n_special > 2:
        offsets = [(0, 0)] + offsets + [(0, 0)] * (n_special - 1)

    enc = {"input_ids": ids, "attention_mask": [1] * len(ids)}
    if "token_type_ids" in tok.model_input_names:
        enc["token_type_ids"] = [0] * len(ids)
    return enc, offsets


def encode_with_offsets(
    tok, text: str, max_len: int
) -> Tuple[dict, List[Tuple[int, int]]]:
    """Fast / Slow 両トークナイザで (encoding, offset_mapping) を返す。"""
    if getattr(tok, "is_fast", False):
        enc = tok(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_len,
        )
        raw_offsets = enc.pop("offset_mapping")
        return dict(enc), [(int(a), int(b)) for a, b in raw_offsets]
    return _encode_slow_bert_japanese(tok, text, max_len)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """数値安定版 softmax（最終軸）。"""
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _decode_raw_bio_with_scores(
    text: str,
    offsets: List[Tuple[int, int]],
    pred_ids: Sequence[int],
    err_probs: Sequence[float],
) -> List[Tuple[int, int, float]]:
    """BIO トークン予測から文字スパンとスパン内トークン平均誤り確率を算出する。"""
    raw_spans: List[List[object]] = []
    cur: List[object] | None = None

    for (a, b), tid, err_p in zip(offsets, pred_ids, err_probs):
        if a >= b:
            continue
        if tid == 1:  # B-ERR
            if cur is not None and a == cur[0] and b == cur[1]:
                cur[1] = max(int(cur[1]), b)
                cur[2].append(float(err_p))  # type: ignore[union-attr]
            else:
                if cur is not None:
                    raw_spans.append(cur)
                cur = [a, b, [float(err_p)]]
        elif tid == 2:  # I-ERR
            if cur is not None:
                cur[1] = max(int(cur[1]), b)
                cur[2].append(float(err_p))  # type: ignore[union-attr]
            else:
                cur = [a, b, [float(err_p)]]
        else:
            if cur is not None:
                raw_spans.append(cur)
                cur = None

    if cur is not None:
        raw_spans.append(cur)

    merged: List[List[object]] = []
    for s_obj, e_obj, sc_list in raw_spans:
        s, e = int(s_obj), int(e_obj)  # type: ignore[arg-type]
        if merged and s < int(merged[-1][1]):  # type: ignore[arg-type]
            merged[-1][1] = max(int(merged[-1][1]), e)  # type: ignore[arg-type]
            merged[-1][2].extend(sc_list)  # type: ignore[union-attr]
        else:
            merged.append([s, e, list(sc_list)])  # type: ignore[arg-type]

    n_text = len(text)
    out: List[Tuple[int, int, float]] = []
    for s_obj, e_obj, sc_list in merged:
        s = max(0, min(int(s_obj), n_text))  # type: ignore[arg-type]
        e = max(0, min(int(e_obj), n_text))  # type: ignore[arg-type]
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            mean_sc = float(sum(sc_list) / len(sc_list))  # type: ignore[arg-type]
            out.append((s, e, mean_sc))
    return out


def decode_probs_to_spans(
    text: str,
    offsets: List[Tuple[int, int]],
    probs: np.ndarray,
    decoding_method: str = "span_mean_err",
    threshold: float = 0.80,
) -> Tuple[List[dict], List[int]]:
    """トークン確率行列 [seq_len, 3] に閾値を適用し、誤りスパンと文字 BIO 配列を返す。"""
    err_probs = [float(1.0 - probs[ti, 0]) for ti in range(len(offsets))]

    if decoding_method == "span_mean_err":
        pred_ids = probs.argmax(axis=-1).tolist()
        candidates = _decode_raw_bio_with_scores(text, offsets, pred_ids, err_probs)
        kept = [(s, e, sc) for s, e, sc in candidates if sc >= threshold]
    elif decoding_method == "token_err":
        pred_ids = [
            (1 if probs[ti, 1] >= probs[ti, 2] else 2) if err_probs[ti] >= threshold else 0
            for ti in range(len(offsets))
        ]
        kept = _decode_raw_bio_with_scores(text, offsets, pred_ids, err_probs)
    else:
        raise ValueError(f"未対応の decoding_method です: {decoding_method}")

    char_labels = [0] * len(text)
    spans_out: List[dict] = []
    for s, e, sc in kept:
        char_labels[s] = 1
        for k in range(s + 1, e):
            char_labels[k] = 2
        spans_out.append(
            {
                "start": int(s),
                "end": int(e),
                "text": text[s:e],
                "score": round(float(sc), 6),
            }
        )
    return spans_out, char_labels


class JaErrorDetector:
    """日本語テキスト誤り検出モデル（ONNX Runtime / PyTorch 両対応）。"""

    def __init__(
        self,
        model_dir: str | Path,
        backend: str = "onnx",
        threshold: Optional[float] = None,
        max_length: Optional[int] = None,
        providers: Optional[Sequence[str]] = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_dir = str(model_dir)
        self.backend = backend.lower()
        if self.backend not in ("onnx", "pytorch"):
            raise ValueError(f"backend は 'onnx' または 'pytorch' を指定してください: {backend}")

        inf_cfg = self._load_inference_config(model_dir, local_files_only=local_files_only)
        self.decoding_method = str(inf_cfg.get("decoding_method", "span_mean_err"))
        self.threshold = float(
            inf_cfg.get("threshold", 0.80) if threshold is None else threshold
        )
        self.max_length = int(
            inf_cfg.get("max_length", 256) if max_length is None else max_length
        )
        onnx_rel_path = str(inf_cfg.get("onnx_path", "onnx/model.onnx"))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=local_files_only
        )

        if self.backend == "onnx":
            import onnxruntime as ort

            model_path = Path(self.model_dir) / onnx_rel_path
            if not model_path.exists():
                raise FileNotFoundError(f"ONNX モデルが見つかりません: {model_path}")
            self.session = ort.InferenceSession(
                str(model_path),
                providers=list(providers) if providers else ["CPUExecutionProvider"],
            )
            self._ort_inputs = [i.name for i in self.session.get_inputs()]
            self.pt_model = None
        else:
            import torch
            from transformers import AutoModelForTokenClassification

            self._torch = torch
            self.pt_model = AutoModelForTokenClassification.from_pretrained(
                self.model_dir,
                local_files_only=local_files_only,
                dtype=torch.float32,
            ).eval()
            self.session = None
            self._ort_inputs = []

    @staticmethod
    def _load_inference_config(
        model_dir: str | Path, local_files_only: bool = False
    ) -> Dict[str, object]:
        base = Path(model_dir)
        inf_file = base / "inference_config.json"
        if inf_file.exists():
            return json.loads(inf_file.read_text(encoding="utf-8"))
        cfg_file = base / "config.json"
        if cfg_file.exists():
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            if isinstance(cfg.get("error_detect_inference"), dict):
                return dict(cfg["error_detect_inference"])
        try:
            cfg = AutoConfig.from_pretrained(
                str(model_dir), local_files_only=local_files_only
            )
            extra = getattr(cfg, "error_detect_inference", None)
            if isinstance(extra, dict):
                return dict(extra)
        except Exception:
            pass
        return {
            "decoding_method": "span_mean_err",
            "threshold": 0.80,
            "max_length": 256,
            "onnx_path": "onnx/model.onnx",
        }

    def _predict_probs(self, enc: dict) -> np.ndarray:
        if self.backend == "onnx":
            ids = np.asarray([enc["input_ids"]], dtype=np.int64)
            feed: Dict[str, np.ndarray] = {}
            for name in self._ort_inputs:
                if name in enc:
                    feed[name] = np.asarray([enc[name]], dtype=np.int64)
                elif name == "token_type_ids":
                    feed[name] = np.zeros_like(ids, dtype=np.int64)
                else:
                    raise KeyError(f"未対応の ONNX 入力名です: {name}")
            logits = self.session.run(None, feed)[0][0].astype(np.float32)  # type: ignore[union-attr]
            return _softmax(logits)

        torch = self._torch
        kw = {
            "input_ids": torch.tensor([enc["input_ids"]], dtype=torch.long),
            "attention_mask": torch.tensor([enc["attention_mask"]], dtype=torch.long),
        }
        if "token_type_ids" in enc:
            kw["token_type_ids"] = torch.tensor(
                [enc["token_type_ids"]], dtype=torch.long
            )
        with torch.no_grad():
            logits = self.pt_model(**kw).logits[0].float().numpy()  # type: ignore[misc]
        return _softmax(logits)

    def detect_with_details(self, text: str) -> dict:
        """誤りスパン一覧に加え、文字単位 BIO ラベル列や閾値情報も返す。"""
        if not text:
            return {
                "text": text,
                "spans": [],
                "char_labels": [],
                "decoding_method": self.decoding_method,
                "threshold": self.threshold,
            }
        enc, offsets = encode_with_offsets(self.tokenizer, text, max_len=self.max_length)
        probs = self._predict_probs(enc)
        spans, char_ids = decode_probs_to_spans(
            text,
            offsets,
            probs,
            decoding_method=self.decoding_method,
            threshold=self.threshold,
        )
        return {
            "text": text,
            "spans": spans,
            "char_labels": [ID2LABEL[i] for i in char_ids],
            "decoding_method": self.decoding_method,
            "threshold": self.threshold,
        }

    def detect(self, text: str) -> List[dict]:
        """設定閾値を適用した誤り文字スパン一覧のみを返す。"""
        return self.detect_with_details(text)["spans"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ja-error-detect-v1 誤りスパン検出 CLI"
    )
    ap.add_argument("--model-dir", required=True, help="Hugging Face 形式ディレクトリ")
    ap.add_argument("--backend", default="onnx", choices=["onnx", "pytorch"])
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--text", required=True, help="検査対象の日本語文")
    args = ap.parse_args()

    detector = JaErrorDetector(
        args.model_dir,
        backend=args.backend,
        threshold=args.threshold,
        local_files_only=args.local_files_only,
    )
    out = detector.detect_with_details(args.text)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
