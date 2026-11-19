#!/usr/bin/env python3
"""第3号 ja-punct-restore-v1: PyTorch FP32 vs ONNX FP32 一致検証＆CPU(i7-11700F)推論速度ベンチマーク。

1. Parity Verification:
   - dev.jsonl 先頭 200 ウィンドウ: PyTorch CPU FP32 (final-baked) vs ONNX Runtime CPU FP32 (onnx/model_fp32.onnx)
   - fleurs-ja-test.jsonl 先頭 50 発話: 同上
2. CPU Speed Benchmark (Core i7-11700F, CPUExecutionProvider, batch=1):
   - default_threads (intra=0, inter=0) および physical_8_threads (intra=8, inter=1)
   - FLEURS ASR 200 発話 (ORT単体 & トークナイズ込 E2E)
   - dev.jsonl 100 ウィンドウ (640文字重なり分割, ORT単体 & E2E)
   - 固定トークン長スイープ (seq_len = 64, 128, 256, 512) ＋ PyTorch CPU FP32 比較
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import onnxruntime as ort
import torch
import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_final import (  # noqa: E402
    decode_labels_from_logits,
    eval_sealed_split,
    extract_char_logits,
    load_jsonl,
)


def make_ort_session(onnx_path: str, intra_threads: int, inter_threads: int) -> ort.InferenceSession:
    """指定スレッド設定で CPUExecutionProvider セッションを作成する。"""
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_threads
    so.inter_op_num_threads = inter_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])


def extract_ort_char_logits(
    sess: ort.InferenceSession,
    tok,
    texts: list[str],
    max_len: int = 640,
    overlap: int = 128,
    batch_size: int = 16,
) -> list[np.ndarray]:
    """PyTorch 側と完全同一のチャンク分割・先頭優先規則で ORT から文字位置ロジットを抽出する。"""
    step = max(1, max_len - overlap)
    chunk_meta: list[tuple[int, int, str]] = []
    win_logits: list[np.ndarray] = []

    for wi, text in enumerate(texts):
        n = len(text)
        win_logits.append(np.full((n, 5), np.nan, dtype=np.float32))
        start = 0
        while start < n:
            chunk = text[start : start + max_len]
            chunk_meta.append((wi, start, chunk))
            if start + max_len >= n:
                break
            start += step

    chunk_meta.sort(key=lambda x: (x[1], len(x[2]), x[0]))
    in_names = [i.name for i in sess.get_inputs()]

    for b0 in range(0, len(chunk_meta), batch_size):
        batch = chunk_meta[b0 : b0 + batch_size]
        enc = tok(
            [c[2] for c in batch],
            return_offsets_mapping=True,
            padding=True,
            return_tensors="np",
        )
        feed = {k: np.asarray(enc[k], dtype=np.int64) for k in in_names}
        logits_np = sess.run(None, feed)[0].astype(np.float32)

        for bi, (wi, start, _) in enumerate(batch):
            arr = win_logits[wi]
            offsets = enc["offset_mapping"][bi]
            lg = logits_np[bi]
            for ti, (s, e) in enumerate(offsets):
                if e > s:
                    pos = start + int(e) - 1
                    if np.isnan(arr[pos, 0]):
                        arr[pos] = lg[ti]

    return win_logits


def compare_logits(
    logits_pt: list[np.ndarray],
    logits_ox: list[np.ndarray],
) -> dict:
    """PyTorch と ONNX の文字位置ロジット・予測ラベルの一致統計を算出する。"""
    zero_bias = [0.0, 0.0, 0.0, 0.0, 0.0]
    preds_pt = decode_labels_from_logits(logits_pt, zero_bias)
    preds_ox = decode_labels_from_logits(logits_ox, zero_bias)

    item_exact = 0
    char_exact = 0
    total_chars = 0
    cov_exact = 0
    total_cov = 0
    diffs = []

    for lp, lx, pp, px in zip(logits_pt, logits_ox, preds_pt, preds_ox):
        assert len(lp) == len(lx) == len(pp) == len(px)
        total_chars += len(pp)
        eq_arr = np.asarray(pp, dtype=np.int64) == np.asarray(px, dtype=np.int64)
        char_exact += int(eq_arr.sum())
        if pp == px:
            item_exact += 1
        mask_p = ~np.isnan(lp[:, 0])
        mask_x = ~np.isnan(lx[:, 0])
        assert np.array_equal(mask_p, mask_x)
        if mask_p.any():
            total_cov += int(mask_p.sum())
            cov_exact += int(eq_arr[mask_p].sum())
            diffs.append(np.abs(lp[mask_p] - lx[mask_p]))

    flat_diff = np.concatenate([d.reshape(-1) for d in diffs])
    return {
        "items": len(logits_pt),
        "total_chars": total_chars,
        "covered_chars": total_cov,
        "item_exact_match": item_exact,
        "item_exact_rate": round(item_exact / len(logits_pt), 6),
        "char_exact_match": char_exact,
        "char_exact_rate": round(char_exact / total_chars, 8),
        "covered_char_exact_match": cov_exact,
        "covered_char_exact_rate": round(cov_exact / total_cov, 8),
        "max_abs_logit_diff": round(float(flat_diff.max()), 6),
        "mean_abs_logit_diff": round(float(flat_diff.mean()), 8),
    }


def summarize_latencies(lat_ms: list[float], total_sec: float, total_chars: int = 0) -> dict:
    """レイテンシ一覧 (ms) から要約統計を計算する。"""
    arr = np.asarray(lat_ms, dtype=np.float64)
    out = {
        "n": int(len(arr)),
        "total_sec": round(float(total_sec), 4),
        "samples_per_sec": round(float(len(arr) / total_sec), 3) if total_sec > 0 else 0.0,
        "mean_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
    }
    if total_chars > 0:
        out["total_chars"] = int(total_chars)
        out["chars_per_sec"] = round(float(total_chars / total_sec), 1) if total_sec > 0 else 0.0
    return out


def decode_single_ort_window(
    sess: ort.InferenceSession,
    tok,
    text: str,
    in_names: list[str],
    max_len: int = 640,
    overlap: int = 128,
) -> list[int]:
    """1 ウィンドウ分の E2E 推論（チャンク分割＋トークナイズ＋ORT＋文字ラベル復号）を実行する。"""
    n = len(text)
    step = max(1, max_len - overlap)
    preds = np.full(n, -1, dtype=np.int64)
    start = 0
    while start < n:
        chunk = text[start : start + max_len]
        enc = tok(chunk, return_offsets_mapping=True, return_tensors="np")
        feed = {k: np.asarray(enc[k], dtype=np.int64) for k in in_names}
        logits = sess.run(None, feed)[0][0]
        pred_ids = logits.argmax(axis=-1)
        for ti, (s, e) in enumerate(enc["offset_mapping"][0]):
            if e > s:
                pos = start + int(e) - 1
                if preds[pos] == -1:
                    preds[pos] = int(pred_ids[ti])
        if start + max_len >= n:
            break
        start += step
    preds[preds == -1] = 0
    return preds.tolist()


def bench_ort_config(
    sess: ort.InferenceSession,
    tok,
    asr_texts: list[str],
    dev_texts: list[str],
) -> dict:
    """単一 ORT スレッド設定で ASR・dev ウィンドウ・固定長スイープを計測する。"""
    in_names = [i.name for i in sess.get_inputs()]

    # 1. FLEURS ASR 実データ (先頭 200 発話, batch=1)
    asr_feeds = []
    for t in asr_texts:
        enc = tok(t, return_tensors="np")
        asr_feeds.append({k: np.asarray(enc[k], dtype=np.int64) for k in in_names})
    asr_chars = sum(len(t) for t in asr_texts)

    for feed in asr_feeds[:10]:
        sess.run(None, feed)

    lats_asr_ort = []
    t_start = time.perf_counter()
    for feed in asr_feeds:
        t0 = time.perf_counter()
        _ = sess.run(None, feed)[0]
        lats_asr_ort.append((time.perf_counter() - t0) * 1000.0)
    tot_asr_ort = time.perf_counter() - t_start

    for t in asr_texts[:10]:
        _ = decode_single_ort_window(sess, tok, t, in_names)

    lats_asr_e2e = []
    t_start = time.perf_counter()
    for t in asr_texts:
        t0 = time.perf_counter()
        _ = decode_single_ort_window(sess, tok, t, in_names)
        lats_asr_e2e.append((time.perf_counter() - t0) * 1000.0)
    tot_asr_e2e = time.perf_counter() - t_start

    # 2. dev.jsonl 実ウィンドウ (先頭 100 件, 640文字重なり分割)
    dev_win_feeds = []
    dev_chars = sum(len(t) for t in dev_texts)
    total_dev_chunks = 0
    for t in dev_texts:
        wfeeds = []
        s = 0
        while s < len(t):
            enc = tok(t[s : s + 640], return_tensors="np")
            wfeeds.append({k: np.asarray(enc[k], dtype=np.int64) for k in in_names})
            if s + 640 >= len(t):
                break
            s += 512
        dev_win_feeds.append(wfeeds)
        total_dev_chunks += len(wfeeds)

    for wf in dev_win_feeds[:5]:
        for feed in wf:
            sess.run(None, feed)

    lats_dev_win_ort = []
    lats_dev_chunk_ort = []
    t_start = time.perf_counter()
    for wf in dev_win_feeds:
        tw0 = time.perf_counter()
        for feed in wf:
            tc0 = time.perf_counter()
            _ = sess.run(None, feed)[0]
            lats_dev_chunk_ort.append((time.perf_counter() - tc0) * 1000.0)
        lats_dev_win_ort.append((time.perf_counter() - tw0) * 1000.0)
    tot_dev_ort = time.perf_counter() - t_start

    for t in dev_texts[:5]:
        _ = decode_single_ort_window(sess, tok, t, in_names)

    lats_dev_e2e = []
    t_start = time.perf_counter()
    for t in dev_texts:
        t0 = time.perf_counter()
        _ = decode_single_ort_window(sess, tok, t, in_names)
        lats_dev_e2e.append((time.perf_counter() - t0) * 1000.0)
    tot_dev_e2e = time.perf_counter() - t_start

    # 3. 固定トークン長スイープ (64, 128, 256, 512, batch=1)
    rng = np.random.default_rng(42)
    sweep = {}
    for seq_len in (64, 128, 256, 512):
        ids_np = rng.integers(100, 30000, size=(1, seq_len), dtype=np.int64)
        mask_np = np.ones((1, seq_len), dtype=np.int64)
        feed = {"input_ids": ids_np, "attention_mask": mask_np}
        if "token_type_ids" in in_names:
            feed["token_type_ids"] = np.zeros((1, seq_len), dtype=np.int64)
        for _ in range(5):
            sess.run(None, feed)
        lats = []
        ts = time.perf_counter()
        for _ in range(20):
            t0 = time.perf_counter()
            _ = sess.run(None, feed)[0]
            lats.append((time.perf_counter() - t0) * 1000.0)
        sweep[str(seq_len)] = summarize_latencies(lats, time.perf_counter() - ts)

    return {
        "asr_200": {
            "ort_only": summarize_latencies(lats_asr_ort, tot_asr_ort, asr_chars),
            "e2e": summarize_latencies(lats_asr_e2e, tot_asr_e2e, asr_chars),
        },
        "dev_100": {
            "total_chunks": total_dev_chunks,
            "ort_only_per_window": summarize_latencies(lats_dev_win_ort, tot_dev_ort, dev_chars),
            "ort_only_per_chunk": summarize_latencies(lats_dev_chunk_ort, tot_dev_ort),
            "e2e_per_window": summarize_latencies(lats_dev_e2e, tot_dev_e2e, dev_chars),
        },
        "fixed_seq_len_sweep": sweep,
    }


def bench_pytorch_cpu(pt_model, tok, asr_texts_50: list[str]) -> dict:
    """PyTorch CPU FP32 (batch=1) の比較ベンチマークを計測する。"""
    asr_feeds = []
    asr_chars = sum(len(t) for t in asr_texts_50)
    for t in asr_texts_50:
        enc = tok(t, return_tensors="pt")
        asr_feeds.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

    with torch.no_grad():
        for feed in asr_feeds[:5]:
            _ = pt_model(**feed).logits
        lats_asr = []
        t_start = time.perf_counter()
        for feed in asr_feeds:
            t0 = time.perf_counter()
            _ = pt_model(**feed).logits
            lats_asr.append((time.perf_counter() - t0) * 1000.0)
        tot_asr = time.perf_counter() - t_start

        rng = np.random.default_rng(42)
        sweep = {}
        for seq_len in (64, 128, 256, 512):
            ids_t = torch.from_numpy(rng.integers(100, 30000, size=(1, seq_len), dtype=np.int64))
            mask_t = torch.ones((1, seq_len), dtype=torch.int64)
            for _ in range(3):
                _ = pt_model(input_ids=ids_t, attention_mask=mask_t).logits
            lats = []
            ts = time.perf_counter()
            for _ in range(10):
                t0 = time.perf_counter()
                _ = pt_model(input_ids=ids_t, attention_mask=mask_t).logits
                lats.append((time.perf_counter() - t0) * 1000.0)
            sweep[str(seq_len)] = summarize_latencies(lats, time.perf_counter() - ts)

    return {
        "torch_num_threads": torch.get_num_threads(),
        "asr_50_forward_only": summarize_latencies(lats_asr, tot_asr, asr_chars),
        "fixed_seq_len_sweep": sweep,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baked-model", default="runs/prod-modernbert/final-baked")
    ap.add_argument("--onnx-model", default="onnx/model_fp32.onnx")
    ap.add_argument("--dev", default="data/prod/dev.jsonl")
    ap.add_argument("--asr", default="data/asr-bench/fleurs-ja-test.jsonl")
    ap.add_argument("--out-parity", default="onnx/onnx_parity_dev200.json")
    ap.add_argument("--out-bench", default="onnx/onnx_bench_cpu.json")
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.baked_model)
    pt_model = AutoModelForTokenClassification.from_pretrained(args.baked_model, dtype=torch.float32).eval()
    sess_def = make_ort_session(args.onnx_model, intra_threads=0, inter_threads=0)

    dev_rows = load_jsonl(Path(args.dev))[:200]
    dev_texts = [r["text"] for r in dev_rows]
    dev_golds = [r["labels"] for r in dev_rows]

    asr_rows = load_jsonl(Path(args.asr))[:200]
    asr_texts_200 = [r["asr_text_no_punct"] for r in asr_rows]
    asr_texts_50 = asr_texts_200[:50]

    # Part A: 一致検証 (dev200 + asr50)
    print("Running PyTorch CPU FP32 vs ONNX FP32 parity on dev200 & asr50...", flush=True)
    dev_pt = extract_char_logits(pt_model, tok, dev_texts, device="cpu", batch_size=16, label="pt-dev200")
    dev_ox = extract_ort_char_logits(sess_def, tok, dev_texts, batch_size=16)
    dev_cmp = compare_logits(dev_pt, dev_ox)

    zero_bias = [0.0, 0.0, 0.0, 0.0, 0.0]
    dev200_pt_metrics = eval_sealed_split(dev_pt, dev_golds, zero_bias)
    dev200_ox_metrics = eval_sealed_split(dev_ox, dev_golds, zero_bias)
    del dev200_pt_metrics["confusion_matrix_pred_by_gold"]
    del dev200_ox_metrics["confusion_matrix_pred_by_gold"]

    asr_pt = extract_char_logits(pt_model, tok, asr_texts_50, device="cpu", batch_size=16, label="pt-asr50")
    asr_ox = extract_ort_char_logits(sess_def, tok, asr_texts_50, batch_size=16)
    asr_cmp = compare_logits(asr_pt, asr_ox)

    parity = {
        "baked_model": args.baked_model,
        "onnx_model": args.onnx_model,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "onnxruntime_version": ort.__version__,
        "dev200": dev_cmp,
        "dev200_metrics_pt": dev200_pt_metrics,
        "dev200_metrics_onnx": dev200_ox_metrics,
        "asr50": asr_cmp,
    }
    Path(args.out_parity).write_text(json.dumps(parity, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PARITY dev200: {dev_cmp}", flush=True)
    print(f"PARITY asr50:  {asr_cmp}", flush=True)

    # Part B: CPU 速度ベンチマーク
    print("Running CPU speed benchmarks (default_threads)...", flush=True)
    res_def = bench_ort_config(sess_def, tok, asr_texts_200, dev_texts[:100])
    del sess_def

    print("Running CPU speed benchmarks (physical_8_threads)...", flush=True)
    sess_8 = make_ort_session(args.onnx_model, intra_threads=8, inter_threads=1)
    res_8 = bench_ort_config(sess_8, tok, asr_texts_200, dev_texts[:100])
    del sess_8

    print("Running PyTorch CPU comparison benchmark...", flush=True)
    res_pt = bench_pytorch_cpu(pt_model, tok, asr_texts_50)

    asr_lens = [len(t) for t in asr_texts_200]
    dev_lens = [len(t) for t in dev_texts[:100]]
    bench = {
        "cpu": platform.processor(),
        "logical_cores": os.cpu_count(),
        "onnxruntime_version": ort.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "asr_200_char_len": {
            "min": min(asr_lens),
            "median": float(np.median(asr_lens)),
            "max": max(asr_lens),
            "total": sum(asr_lens),
        },
        "dev_100_char_len": {
            "min": min(dev_lens),
            "median": float(np.median(dev_lens)),
            "max": max(dev_lens),
            "total": sum(dev_lens),
        },
        "ort_default_threads": res_def,
        "ort_physical_8_threads": res_8,
        "pytorch_cpu_fp32": res_pt,
        "total_bench_elapsed_sec": round(time.time() - t0, 2),
    }
    Path(args.out_bench).write_text(json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Bench complete in {bench['total_bench_elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
