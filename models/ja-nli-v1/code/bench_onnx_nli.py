#!/usr/bin/env python3
"""第11号 ja-nli-v1: PyTorch FP32 vs ONNX FP32 一致検証・CPU 速度ベンチマーク。

release-staging/ 単体（local_files_only=True）を読み込み、以下を計測する:
1. PyTorch CPU FP32 vs ONNX Runtime CPU FP32 の一致検証
   - data/prod/dev.jsonl 先頭 200 件
   - data/prod/sealed-test.jsonl 先頭 200 件（本文・個別IDは出力一切禁止、前後SHA-256不変確認）
2. デスクトップ CPU 処理時間計測
   - 実データ（dev 先頭 200 件: batch=1 Forward単体/E2E [default_threads vs physical_8_threads],
     batch=8 E2E、同一先頭 50 件での PyTorch CPU FP32 直接比較）
   - 固定トークン長スイープ（64, 128, 256, 512, 1024）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
LABELS = ("entailment", "neutral", "contradiction")
LABEL2ID = {lb: i for i, lb in enumerate(LABELS)}


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


def _softmax(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - np.max(arr, axis=-1, keepdims=True)
    exp_arr = np.exp(shifted)
    return exp_arr / np.sum(exp_arr, axis=-1, keepdims=True)


def tokenize_pair_only_first(tok, premises, hypotheses, max_length: int, padding: bool, return_tensors: str):
    """前提末尾優先切り詰めでトークナイズし、仮説単独超過時のみ両側切り詰めへ退避する。"""
    try:
        return tok(
            premises,
            hypotheses,
            truncation="only_first",
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors,
        )
    except ValueError:
        return tok(
            premises,
            hypotheses,
            truncation=True,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors,
        )


def load_first_n(jsonl_path: Path, n: int) -> tuple[List[Tuple[str, str]], List[int]]:
    """JSONL 先頭 n 件から ((premise, hypothesis), label_id) を読み込む（行単位で安全読込）。"""
    pairs: List[Tuple[str, str]] = []
    label_ids: List[int] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pairs.append((str(row["premise"]), str(row["hypothesis"])))
            label_ids.append(LABEL2ID[str(row["label"])])
            if len(pairs) >= n:
                break
    assert len(pairs) == n, f"件数不足 ({jsonl_path}): expected={n}, got={len(pairs)}"
    return pairs, label_ids


def run_parity_split(
    split_name: str,
    pairs: List[Tuple[str, str]],
    gold_ids: List[int],
    tok,
    pt_model,
    ort_sess: ort.InferenceSession,
    max_length: int,
) -> dict:
    """単一分割における PyTorch CPU FP32 vs ONNX Runtime CPU FP32 の一致率を計測する。"""
    n = len(pairs)
    total_chars = sum(len(p) + len(h) for p, h in pairs)
    gold_counts = dict(Counter(LABELS[i] for i in gold_ids))

    pt_logits_list: List[np.ndarray] = []
    ox_logits_list: List[np.ndarray] = []

    with torch.inference_mode():
        for prem, hyp in pairs:
            enc = tokenize_pair_only_first(
                tok, prem, hyp, max_length=max_length, padding=False, return_tensors="pt"
            )
            pt_out = pt_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            ).logits.cpu().numpy()
            ox_out = ort_sess.run(
                None,
                {
                    "input_ids": enc["input_ids"].numpy().astype(np.int64),
                    "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
                },
            )[0]
            pt_logits_list.append(pt_out[0])
            ox_logits_list.append(ox_out[0])

    pt_logits = np.stack(pt_logits_list, axis=0)
    ox_logits = np.stack(ox_logits_list, axis=0)
    pt_probs = _softmax(pt_logits)
    ox_probs = _softmax(ox_logits)

    pt_preds = np.argmax(pt_logits, axis=-1)
    ox_preds = np.argmax(ox_logits, axis=-1)
    gold_arr = np.asarray(gold_ids, dtype=np.int64)

    matches = pt_preds == ox_preds
    match_count = int(np.sum(matches))

    per_class: Dict[str, dict] = {}
    for idx, lb in enumerate(LABELS):
        mask = pt_preds == idx
        cnt = int(np.sum(mask))
        matched = int(np.sum(matches & mask))
        per_class[lb] = {"pt_pred_count": cnt, "match_count": matched}

    logit_abs = np.abs(pt_logits.astype(np.float64) - ox_logits.astype(np.float64))
    prob_abs = np.abs(pt_probs - ox_probs)

    return {
        "split": split_name,
        "n_samples": n,
        "total_chars": int(total_chars),
        "gold_label_counts": gold_counts,
        "label_match_count": match_count,
        "label_match_rate": round(match_count / n, 6),
        "per_class_match": per_class,
        "logit_max_abs_diff": float(np.max(logit_abs)),
        "logit_mean_abs_diff": float(np.mean(logit_abs)),
        "prob_max_abs_diff": float(np.max(prob_abs)),
        "prob_mean_abs_diff": float(np.mean(prob_abs)),
        "pt_accuracy": round(float(np.mean(pt_preds == gold_arr)), 6),
        "onnx_accuracy": round(float(np.mean(ox_preds == gold_arr)), 6),
        "_raw": (pt_logits, ox_logits, pt_probs, ox_probs, pt_preds, ox_preds, gold_arr),
    }


def summarize_ms(times_ms: List[float], total_chars: int) -> dict:
    """レイテンシ (ms) リストから平均・分位点・スループットを算出する。"""
    arr = np.asarray(times_ms, dtype=np.float64)
    total_sec = float(np.sum(arr)) / 1000.0
    n = len(times_ms)
    return {
        "n": n,
        "mean_ms": round(float(np.mean(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "total_ms": round(float(np.sum(arr)), 2),
        "samples_per_sec": round(n / total_sec, 3),
        "chars_per_sec": round(total_chars / total_sec, 1),
    }


def make_ort_session(onnx_path: Path, intra_threads: int, inter_threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    if intra_threads > 0:
        opts.intra_op_num_threads = int(intra_threads)
    if inter_threads > 0:
        opts.inter_op_num_threads = int(inter_threads)
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def bench_ort_dataset(
    sess: ort.InferenceSession,
    tok,
    pairs: List[Tuple[str, str]],
    max_length: int,
    warmup: int = 5,
) -> dict:
    """ONNX Runtime CPU で Forward 単体と E2E の batch=1 レイテンシを計測する。"""
    total_chars = sum(len(p) + len(h) for p, h in pairs)
    pre_enc = []
    for p, h in pairs:
        enc = tokenize_pair_only_first(tok, p, h, max_length=max_length, padding=False, return_tensors="np")
        pre_enc.append(
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            }
        )

    for feed in pre_enc[:warmup]:
        _ = sess.run(None, feed)

    fwd_times: List[float] = []
    for feed in pre_enc:
        t0 = time.perf_counter()
        _ = sess.run(None, feed)
        fwd_times.append((time.perf_counter() - t0) * 1000.0)

    for p, h in pairs[:warmup]:
        enc = tokenize_pair_only_first(tok, p, h, max_length=max_length, padding=False, return_tensors="np")
        out = sess.run(
            None,
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )[0]
        _ = _softmax(out)

    e2e_times: List[float] = []
    for p, h in pairs:
        t0 = time.perf_counter()
        enc = tokenize_pair_only_first(tok, p, h, max_length=max_length, padding=False, return_tensors="np")
        out = sess.run(
            None,
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )[0]
        probs = _softmax(out)
        _ = int(np.argmax(probs[0]))
        e2e_times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "forward_only": summarize_ms(fwd_times, total_chars),
        "e2e": summarize_ms(e2e_times, total_chars),
    }


def bench_ort_batched(
    sess: ort.InferenceSession,
    tok,
    pairs: List[Tuple[str, str]],
    max_length: int,
    batch_size: int = 8,
) -> dict:
    """バッチ推論（例: batch_size=8）での E2E スループットを計測する。"""
    total_chars = sum(len(p) + len(h) for p, h in pairs)
    warm = pairs[: min(batch_size, len(pairs))]
    warm_enc = tokenize_pair_only_first(
        tok,
        [p for p, _ in warm],
        [h for _, h in warm],
        max_length=max_length,
        padding=True,
        return_tensors="np",
    )
    _ = sess.run(
        None,
        {
            "input_ids": warm_enc["input_ids"].astype(np.int64),
            "attention_mask": warm_enc["attention_mask"].astype(np.int64),
        },
    )

    t0 = time.perf_counter()
    for st in range(0, len(pairs), batch_size):
        batch = pairs[st : st + batch_size]
        enc = tokenize_pair_only_first(
            tok,
            [p for p, _ in batch],
            [h for _, h in batch],
            max_length=max_length,
            padding=True,
            return_tensors="np",
        )
        out = sess.run(
            None,
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )[0]
        probs = _softmax(out)
        _ = np.argmax(probs, axis=-1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    n = len(pairs)
    return {
        "n": n,
        "batch_size": batch_size,
        "total_ms": round(elapsed_ms, 2),
        "ms_per_sample": round(elapsed_ms / n, 3),
        "samples_per_sec": round(n / (elapsed_ms / 1000.0), 3),
        "chars_per_sec": round(total_chars / (elapsed_ms / 1000.0), 1),
    }


def bench_pt_dataset(
    pt_model,
    tok,
    pairs: List[Tuple[str, str]],
    max_length: int,
    warmup: int = 3,
) -> dict:
    """参考: PyTorch CPU FP32 で Forward 単体と E2E の batch=1 レイテンシを計測する。"""
    total_chars = sum(len(p) + len(h) for p, h in pairs)
    pre_enc = [
        tokenize_pair_only_first(tok, p, h, max_length=max_length, padding=False, return_tensors="pt")
        for p, h in pairs
    ]

    with torch.inference_mode():
        for enc in pre_enc[:warmup]:
            _ = pt_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits

        fwd_times: List[float] = []
        for enc in pre_enc:
            t0 = time.perf_counter()
            _ = pt_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            fwd_times.append((time.perf_counter() - t0) * 1000.0)

        for p, h in pairs[:warmup]:
            enc = tokenize_pair_only_first(tok, p, h, max_length=max_length, padding=False, return_tensors="pt")
            logits = pt_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.numpy()
            _ = _softmax(logits)

        e2e_times: List[float] = []
        for p, h in pairs:
            t0 = time.perf_counter()
            enc = tokenize_pair_only_first(tok, p, h, max_length=max_length, padding=False, return_tensors="pt")
            logits = pt_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.numpy()
            probs = _softmax(logits)
            _ = int(np.argmax(probs[0]))
            e2e_times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "forward_only": summarize_ms(fwd_times, total_chars),
        "e2e": summarize_ms(e2e_times, total_chars),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="第11号 ja-nli-v1 ONNX 一致検証 & CPU ベンチ")
    ap.add_argument("--staging-dir", default=str(ROOT / "release-staging"))
    ap.add_argument("--dev-file", default=str(ROOT / "data" / "prod" / "dev.jsonl"))
    ap.add_argument("--sealed-file", default=str(ROOT / "data" / "prod" / "sealed-test.jsonl"))
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--out-parity", default=str(ROOT / "onnx" / "onnx_parity.json"))
    ap.add_argument("--out-bench", default=str(ROOT / "onnx" / "onnx_bench_cpu.json"))
    args = ap.parse_args()

    staging_dir = Path(args.staging_dir)
    onnx_path = staging_dir / "onnx" / "model.onnx"
    sealed_path = Path(args.sealed_file)
    sealed_sha_before = sha256_file(sealed_path)

    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(str(staging_dir), local_files_only=True)
    try:
        pt_model = AutoModelForSequenceClassification.from_pretrained(
            str(staging_dir),
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
            reference_compile=False,
        ).cpu().eval()
    except TypeError:
        pt_model = AutoModelForSequenceClassification.from_pretrained(
            str(staging_dir),
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).cpu().eval()

    # 1. 一致検証（dev 200 件 + sealed-test 200 件）
    dev_pairs, dev_labels = load_first_n(Path(args.dev_file), args.n_samples)
    sealed_pairs, sealed_labels = load_first_n(sealed_path, args.n_samples)

    sess_parity = make_ort_session(onnx_path, intra_threads=8, inter_threads=1)
    res_dev = run_parity_split("dev_200", dev_pairs, dev_labels, tok, pt_model, sess_parity, args.max_length)
    res_sealed = run_parity_split(
        "sealed_test_200", sealed_pairs, sealed_labels, tok, pt_model, sess_parity, args.max_length
    )

    # 封印本文を即座にメモリから解放し、SHA-256 不変を確認
    del sealed_pairs
    sealed_sha_after = sha256_file(sealed_path)
    assert sealed_sha_before == sealed_sha_after, "sealed-test.jsonl の SHA-256 が変化しました！"

    pt_l1, ox_l1, pt_p1, ox_p1, pt_y1, ox_y1, g1 = res_dev.pop("_raw")
    pt_l2, ox_l2, pt_p2, ox_p2, pt_y2, ox_y2, g2 = res_sealed.pop("_raw")

    comb_pt_l = np.concatenate([pt_l1, pt_l2], axis=0)
    comb_ox_l = np.concatenate([ox_l1, ox_l2], axis=0)
    comb_pt_p = np.concatenate([pt_p1, pt_p2], axis=0)
    comb_ox_p = np.concatenate([ox_p1, ox_p2], axis=0)
    comb_pt_y = np.concatenate([pt_y1, pt_y2], axis=0)
    comb_ox_y = np.concatenate([ox_y1, ox_y2], axis=0)
    comb_g = np.concatenate([g1, g2], axis=0)
    comb_n = len(comb_g)
    comb_match = int(np.sum(comb_pt_y == comb_ox_y))

    parity_payload = {
        "staging_dir": str(staging_dir).replace("\\", "/"),
        "max_length": args.max_length,
        "truncation": "only_first",
        "n_per_split": args.n_samples,
        "sealed_text_disclosed": False,
        "sealed_test_sha256": sealed_sha_after,
        "sealed_test_unchanged": True,
        "dev_200": res_dev,
        "sealed_test_200": res_sealed,
        "combined_400": {
            "n_samples": comb_n,
            "label_match_count": comb_match,
            "label_match_rate": round(comb_match / comb_n, 6),
            "logit_max_abs_diff": float(np.max(np.abs(comb_pt_l.astype(np.float64) - comb_ox_l.astype(np.float64)))),
            "logit_mean_abs_diff": float(np.mean(np.abs(comb_pt_l.astype(np.float64) - comb_ox_l.astype(np.float64)))),
            "prob_max_abs_diff": float(np.max(np.abs(comb_pt_p - comb_ox_p))),
            "prob_mean_abs_diff": float(np.mean(np.abs(comb_pt_p - comb_ox_p))),
            "pt_accuracy": round(float(np.mean(comb_pt_y == comb_g)), 6),
            "onnx_accuracy": round(float(np.mean(comb_ox_y == comb_g)), 6),
        },
    }
    Path(args.out_parity).write_text(
        json.dumps(parity_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    del sess_parity

    # 2. CPU 推論時間ベンチマーク
    char_lens = [len(p) + len(h) for p, h in dev_pairs]
    tok_lens = [
        len(
            tokenize_pair_only_first(
                tok, p, h, max_length=args.max_length, padding=False, return_tensors=None
            )["input_ids"]
        )
        for p, h in dev_pairs
    ]
    dist_info = {
        "n": len(dev_pairs),
        "char_min": int(min(char_lens)),
        "char_p50": round(float(np.percentile(char_lens, 50)), 1),
        "char_mean": round(float(np.mean(char_lens)), 1),
        "char_p95": round(float(np.percentile(char_lens, 95)), 1),
        "char_max": int(max(char_lens)),
        "char_total": int(sum(char_lens)),
        "tok_min": int(min(tok_lens)),
        "tok_p50": round(float(np.percentile(tok_lens, 50)), 1),
        "tok_mean": round(float(np.mean(tok_lens)), 1),
        "tok_p95": round(float(np.percentile(tok_lens, 95)), 1),
        "tok_max": int(max(tok_lens)),
        "tok_total": int(sum(tok_lens)),
    }

    sess_def = make_ort_session(onnx_path, intra_threads=0, inter_threads=0)
    sess_p8 = make_ort_session(onnx_path, intra_threads=8, inter_threads=1)

    ort_def_200 = bench_ort_dataset(sess_def, tok, dev_pairs, args.max_length, warmup=5)
    ort_p8_200 = bench_ort_dataset(sess_p8, tok, dev_pairs, args.max_length, warmup=5)
    ort_p8_b8 = bench_ort_batched(sess_p8, tok, dev_pairs, args.max_length, batch_size=8)

    # 先頭 50 件の PyTorch CPU との直接同一集合比較
    dev_50 = dev_pairs[:50]
    ort_p8_50 = bench_ort_dataset(sess_p8, tok, dev_50, args.max_length, warmup=3)
    pt_8_50 = bench_pt_dataset(pt_model, tok, dev_50, args.max_length, warmup=3)

    # 固定トークン長スイープ（64, 128, 256, 512, 1024）
    sweep_specs = [
        (64, 20, 10, 3),
        (128, 20, 10, 3),
        (256, 20, 10, 3),
        (512, 15, 8, 3),
        (1024, 10, 5, 2),
    ]
    sweep_results = {}
    for seq_len, ort_rep, pt_rep, warm in sweep_specs:
        # ペア入力構造: [<cls>=6, ..., <sep>=4, ..., <sep>=4]
        ids_np = np.full((1, seq_len), 100, dtype=np.int64)
        ids_np[0, 0] = 6
        ids_np[0, max(2, (seq_len * 3) // 4)] = 4
        ids_np[0, -1] = 4
        mask_np = np.ones((1, seq_len), dtype=np.int64)
        feed_np = {"input_ids": ids_np, "attention_mask": mask_np}
        feed_pt_ids = torch.from_numpy(ids_np)
        feed_pt_mask = torch.from_numpy(mask_np)

        def measure_ort(s: ort.InferenceSession, repeats: int, wm: int) -> dict:
            for _ in range(wm):
                _ = s.run(None, feed_np)
            ts = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                _ = s.run(None, feed_np)
                ts.append((time.perf_counter() - t0) * 1000.0)
            arr = np.asarray(ts, dtype=np.float64)
            return {
                "repeats": repeats,
                "mean_ms": round(float(np.mean(arr)), 3),
                "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
            }

        def measure_pt(repeats: int, wm: int) -> dict:
            with torch.inference_mode():
                for _ in range(wm):
                    _ = pt_model(input_ids=feed_pt_ids, attention_mask=feed_pt_mask).logits
                ts = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    _ = pt_model(input_ids=feed_pt_ids, attention_mask=feed_pt_mask).logits
                    ts.append((time.perf_counter() - t0) * 1000.0)
            arr = np.asarray(ts, dtype=np.float64)
            return {
                "repeats": repeats,
                "mean_ms": round(float(np.mean(arr)), 3),
                "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
            }

        sweep_results[str(seq_len)] = {
            "onnx_default_threads": measure_ort(sess_def, ort_rep, warm),
            "onnx_physical_8_threads": measure_ort(sess_p8, ort_rep, warm),
            "pytorch_cpu_8_threads": measure_pt(pt_rep, warm),
        }

    del sess_def, sess_p8

    bench_payload = {
        "cpu_processor": platform.processor(),
        "logical_cores": os.cpu_count(),
        "onnxruntime_version": ort.__version__,
        "torch_version": torch.__version__,
        "ort_providers": ["CPUExecutionProvider"],
        "max_length": args.max_length,
        "truncation": "only_first",
        "dev_200_length_stats": dist_info,
        "real_data_dev_200_batch1": {
            "onnx_default_threads": ort_def_200,
            "onnx_physical_8_threads": ort_p8_200,
        },
        "real_data_dev_200_batch8": {
            "onnx_physical_8_threads_e2e": ort_p8_b8,
        },
        "real_data_dev_first50_batch1_compare": {
            "onnx_physical_8_threads": ort_p8_50,
            "pytorch_cpu_8_threads": pt_8_50,
        },
        "fixed_seqlen_sweep_batch1": sweep_results,
    }
    Path(args.out_bench).write_text(
        json.dumps(bench_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "parity_dev_200": f"{res_dev['label_match_count']}/{res_dev['n_samples']}",
                "parity_sealed_200": f"{res_sealed['label_match_count']}/{res_sealed['n_samples']}",
                "logit_max_diff_400": round(parity_payload["combined_400"]["logit_max_abs_diff"], 8),
                "prob_max_diff_400": round(parity_payload["combined_400"]["prob_max_abs_diff"], 8),
                "ort_p8_200_e2e_mean_ms": ort_p8_200["e2e"]["mean_ms"],
                "ort_def_200_e2e_mean_ms": ort_def_200["e2e"]["mean_ms"],
                "ort_p8_50_e2e_mean_ms": ort_p8_50["e2e"]["mean_ms"],
                "pt_8_50_e2e_mean_ms": pt_8_50["e2e"]["mean_ms"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
