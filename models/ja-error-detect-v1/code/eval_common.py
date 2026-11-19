#!/usr/bin/env python
"""第6号 ja-error-detect-v1 共通推論・閾値復号・評価ヘルパー。

dev 閾値調整 (tune_dev_threshold.py) と封印テスト最終評価 (eval_sealed_err.py) で
同一のエンコード検証・確率復号・スパン評価ロジックを共有する。
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import error_injector as ei  # noqa: E402
import offsets_util as ou  # noqa: E402


def load_and_encode_eval(
    path: Path, tok, max_len: int = 256
) -> Tuple[List[dict], List[dict], List[List[Tuple[int, int]]], List[List[int]], int]:
    """評価 JSONL を読み込み、全件で input_ids 一致検証を行いエンコード結果を返す。"""
    raw_rows: List[dict] = []
    enc_items: List[dict] = []
    offsets_list: List[List[Tuple[int, int]]] = []
    gold_tok_labels: List[List[int]] = []
    truncated = 0

    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            enc, offsets = ou.encode_with_offsets(tok, r["text"], max_len=max_len)
            ref = tok(r["text"], truncation=True, max_length=max_len)
            assert enc["input_ids"] == ref["input_ids"], (
                f"input_ids 不一致 (row={idx}, id={r.get('id')}): "
                f"{enc['input_ids'][:20]} != {ref['input_ids'][:20]}"
            )
            non_empty_ends = [b for a, b in offsets if b > a]
            if r["text"] and (
                not non_empty_ends or max(non_empty_ends) < len(r["text"].rstrip())
            ):
                truncated += 1

            labels = ou.align_spans_to_tokens(r["text"], offsets, r["spans"])
            raw_rows.append(r)
            enc_items.append(enc)
            offsets_list.append(offsets)
            gold_tok_labels.append(labels)

    return raw_rows, enc_items, offsets_list, gold_tok_labels, truncated


def infer_token_probs(
    model,
    enc_items: List[dict],
    pad_id: int,
    device: str,
    batch_size: int = 32,
) -> Tuple[List[np.ndarray], float]:
    """モデルで全件推論し、各文のトークン確率行列 [seq_len, 3] と推論秒数を返す。"""
    model.to(device).eval()
    probs_list: List[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for start_i in range(0, len(enc_items), batch_size):
            batch = enc_items[start_i : start_i + batch_size]
            maxlen = max(len(b["input_ids"]) for b in batch)
            input_ids, attn = [], []
            for b in batch:
                n = len(b["input_ids"])
                pad_n = maxlen - n
                input_ids.append(b["input_ids"] + [pad_id] * pad_n)
                attn.append(b["attention_mask"] + [0] * pad_n)
            kw = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
                "attention_mask": torch.tensor(attn, dtype=torch.long, device=device),
            }
            if "token_type_ids" in batch[0]:
                kw["token_type_ids"] = torch.tensor(
                    [
                        b["token_type_ids"] + [0] * (maxlen - len(b["token_type_ids"]))
                        for b in batch
                    ],
                    dtype=torch.long,
                    device=device,
                )
            logits = model(**kw).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for k, b in enumerate(batch):
                seq_len = len(b["input_ids"])
                probs_list.append(probs[k, :seq_len].copy())
    eval_sec = time.time() - t0
    return probs_list, eval_sec


def decode_span_mean_err(
    text: str,
    offsets: List[Tuple[int, int]],
    probs: np.ndarray,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[float]]:
    """argmax BIO 候補スパンのトークン平均誤り確率 1-p(O) >= threshold でフィルタする。"""
    argmax_ids = probs.argmax(axis=-1).tolist()
    raw_spans: List[List[object]] = []  # [start, end, [err_probs...]]
    cur: List[object] | None = None

    for ti, ((a, b), tid) in enumerate(zip(offsets, argmax_ids)):
        if a >= b:
            continue
        err_p = float(1.0 - probs[ti, 0])
        if tid == 1:  # B-ERR
            if cur is not None and a == cur[0] and b == cur[1]:
                cur[1] = max(int(cur[1]), b)
                cur[2].append(err_p)  # type: ignore[union-attr]
            else:
                if cur is not None:
                    raw_spans.append(cur)
                cur = [a, b, [err_p]]
        elif tid == 2:  # I-ERR
            if cur is not None:
                cur[1] = max(int(cur[1]), b)
                cur[2].append(err_p)  # type: ignore[union-attr]
            else:
                cur = [a, b, [err_p]]
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
    kept_spans: List[Tuple[int, int]] = []
    kept_scores: List[float] = []
    for s_obj, e_obj, sc_list in merged:
        s = max(0, min(int(s_obj), n_text))  # type: ignore[arg-type]
        e = max(0, min(int(e_obj), n_text))  # type: ignore[arg-type]
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            mean_sc = float(sum(sc_list) / len(sc_list))  # type: ignore[arg-type]
            if mean_sc >= threshold:
                kept_spans.append((s, e))
                kept_scores.append(mean_sc)

    char_preds = [0] * n_text
    for s, e in kept_spans:
        char_preds[s] = 1
        for k in range(s + 1, e):
            char_preds[k] = 2

    return kept_spans, char_preds, kept_scores


def decode_token_err(
    text: str,
    offsets: List[Tuple[int, int]],
    probs: np.ndarray,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[float]]:
    """トークン単位で 1-p(O) >= threshold を ERR（B/I 選択）として BIO 復号する。"""
    pred_ids: List[int] = []
    for ti in range(len(offsets)):
        err_p = float(1.0 - probs[ti, 0])
        if err_p >= threshold:
            pred_ids.append(1 if probs[ti, 1] >= probs[ti, 2] else 2)
        else:
            pred_ids.append(0)
    spans, char_preds = ou.decode_token_preds_to_spans(text, offsets, pred_ids)
    span_scores: List[float] = []
    for s, e in spans:
        tok_errs = [
            float(1.0 - probs[ti, 0])
            for ti, (a, b) in enumerate(offsets)
            if a < b and a < e and b > s
        ]
        span_scores.append(float(sum(tok_errs) / len(tok_errs)) if tok_errs else 0.0)
    return spans, char_preds, span_scores


def decode_with_method(
    text: str,
    offsets: List[Tuple[int, int]],
    probs: np.ndarray,
    method: str,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[float]]:
    """指定された方式（span_mean_err / token_err）で予測スパンと文字 BIO 配列を返す。"""
    if method == "span_mean_err":
        return decode_span_mean_err(text, offsets, probs, threshold)
    if method == "token_err":
        return decode_token_err(text, offsets, probs, threshold)
    raise ValueError(f"未知のデコード方式です: {method}")


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def _match_overlap_1to1(
    gold_spans: List[dict], pred_spans: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    pairs = []
    for gi, g in enumerate(gold_spans):
        gs, ge = int(g["start"]), int(g["end"])
        for pi, (ps, pe) in enumerate(pred_spans):
            ov = min(ge, pe) - max(gs, ps)
            if ov > 0:
                pairs.append((ov, gi, pi))
    pairs.sort(key=lambda x: x[0], reverse=True)
    used_g, used_p = set(), set()
    matched: List[Tuple[int, int]] = []
    for _, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            matched.append((gi, pi))
    return matched


def evaluate_predictions(
    raw_rows: List[dict],
    pred_spans_list: List[List[Tuple[int, int]]],
    char_preds_list: List[List[int]],
) -> dict:
    """全体・種別別・正常文誤検出の各種指標を一括算出する。"""
    ex_tp = ex_fp = ex_fn = 0
    ov_tp = ov_fp = ov_fn = 0
    ch_tp = ch_fp = ch_fn = 0

    pt_ex = {t: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for t in ei.ERROR_TYPES}
    pt_ov = {t: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for t in ei.ERROR_TYPES}

    clean_sents = 0
    clean_chars = 0
    clean_fp_spans = 0
    clean_sents_with_fp = 0
    clean_fp_chars = 0

    for row, pred_spans, char_preds in zip(raw_rows, pred_spans_list, char_preds_list):
        gold_spans = row["spans"]
        text = row["text"]

        for g_lb, p_lb in zip(row["char_labels"], char_preds):
            g_err = g_lb != 0
            p_err = p_lb != 0
            if g_err and p_err:
                ch_tp += 1
            elif (not g_err) and p_err:
                ch_fp += 1
            elif g_err and (not p_err):
                ch_fn += 1

        if not row["has_error"]:
            clean_sents += 1
            clean_chars += len(text)
            if pred_spans:
                clean_fp_spans += len(pred_spans)
                clean_sents_with_fp += 1
                clean_fp_chars += sum(1 for c in char_preds if c != 0)

        gold_map = {(int(g["start"]), int(g["end"])): g["type"] for g in gold_spans}
        pred_set = set(pred_spans)
        matched_exact = set(gold_map.keys()) & pred_set
        unmatched_pred_ex = len(pred_set - matched_exact)

        ex_tp += len(matched_exact)
        ex_fp += unmatched_pred_ex
        ex_fn += len(gold_map) - len(matched_exact)

        present_types = {g["type"] for g in gold_spans}
        for (gs, ge), gtype in gold_map.items():
            pt_ex[gtype]["support"] += 1
            if (gs, ge) in matched_exact:
                pt_ex[gtype]["tp"] += 1
            else:
                pt_ex[gtype]["fn"] += 1
        for t in present_types:
            pt_ex[t]["fp"] += unmatched_pred_ex

        matched_ov_pairs = _match_overlap_1to1(gold_spans, pred_spans)
        matched_g_idxs = {gi for gi, _ in matched_ov_pairs}
        unmatched_pred_ov = len(pred_spans) - len(matched_ov_pairs)

        ov_tp += len(matched_ov_pairs)
        ov_fp += unmatched_pred_ov
        ov_fn += len(gold_spans) - len(matched_ov_pairs)

        for gi, g in enumerate(gold_spans):
            gt = g["type"]
            pt_ov[gt]["support"] += 1
            if gi in matched_g_idxs:
                pt_ov[gt]["tp"] += 1
            else:
                pt_ov[gt]["fn"] += 1
        for t in present_types:
            pt_ov[t]["fp"] += unmatched_pred_ov

    per_type_exact = {}
    per_type_overlap = {}
    for t in ei.ERROR_TYPES:
        d_ex = _prf(pt_ex[t]["tp"], pt_ex[t]["fp"], pt_ex[t]["fn"])
        d_ex["support"] = pt_ex[t]["support"]
        per_type_exact[t] = d_ex

        d_ov = _prf(pt_ov[t]["tp"], pt_ov[t]["fp"], pt_ov[t]["fn"])
        d_ov["support"] = pt_ov[t]["support"]
        per_type_overlap[t] = d_ov

    clean_fpr = {
        "clean_sentences": clean_sents,
        "clean_chars": clean_chars,
        "fp_spans": clean_fp_spans,
        "fp_per_1000_chars": round(clean_fp_spans / (clean_chars / 1000.0), 4)
        if clean_chars
        else 0.0,
        "clean_sentences_with_fp": clean_sents_with_fp,
        "sentence_fpr": round(clean_sents_with_fp / clean_sents, 4)
        if clean_sents
        else 0.0,
        "fp_chars": clean_fp_chars,
        "fp_chars_per_1000_chars": round(clean_fp_chars / (clean_chars / 1000.0), 4)
        if clean_chars
        else 0.0,
    }

    return {
        "exact_span": _prf(ex_tp, ex_fp, ex_fn),
        "overlap_span": _prf(ov_tp, ov_fp, ov_fn),
        "char_binary": _prf(ch_tp, ch_fp, ch_fn),
        "per_type_exact": per_type_exact,
        "per_type_overlap": per_type_overlap,
        "clean_fpr": clean_fpr,
    }


def compute_tokenizer_oracle(
    raw_rows: List[dict],
    offsets_list: List[List[Tuple[int, int]]],
    gold_tok_labels: List[List[int]],
) -> dict:
    """正解トークンラベル復号によるトークナイザ理論上限指標を算出する。"""
    oracle_spans_all: List[List[Tuple[int, int]]] = []
    oracle_chars_all: List[List[int]] = []
    for row, offsets, g_tok in zip(raw_rows, offsets_list, gold_tok_labels):
        oracle_ids = [0 if lb == -100 else lb for lb in g_tok]
        sp, cp = ou.decode_token_preds_to_spans(row["text"], offsets, oracle_ids)
        oracle_spans_all.append(sp)
        oracle_chars_all.append(cp)
    res = evaluate_predictions(raw_rows, oracle_spans_all, oracle_chars_all)
    return {
        "exact_span": res["exact_span"],
        "overlap_span": res["overlap_span"],
        "per_type_exact": res["per_type_exact"],
    }
