#!/usr/bin/env python3
"""ONNX FP32 一致検証、デスクトップ CPU 速度ベンチマーク、および動的 INT8 の dev 全件評価スクリプト。

構成:
1. Parity (dev 先頭200件): PyTorch (FP32, CPU) vs onnx/model_fp32.onnx (CPUExecutionProvider)
2. CPU Benchmark (Core i7-11700F): onnx/model_fp32.onnx と onnx/model_int8.onnx の実データ batch=1 ＆ 固定長スイープ
3. INT8 Accuracy (dev 全5,000件): onnx/model_int8.onnx を評価し、PyTorch FP32 (eval_select_cls.json / dev_probs.npy) との F1 落差等を算出
"""
import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_select_cls import (  # noqa: E402
    binary_metrics,
    find_threshold_fpr_1pct,
    load_jsonl,
    source_group,
)


def softmax_pos_prob(logits: np.ndarray) -> np.ndarray:
    """ロジット配列 (N, 2) からクラス1 (injection) の確率を float64 で計算する。"""
    x = logits.astype(np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return (e[:, 1] / np.sum(e, axis=-1))


def make_ort_session(onnx_path: str, intra_threads: int, inter_threads: int) -> ort.InferenceSession:
    """指定スレッド設定で CPUExecutionProvider セッションを作成する。"""
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_threads
    so.inter_op_num_threads = inter_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])


def run_ort_batches(sess: ort.InferenceSession, tok, texts: list[str], max_len: int, batch_size: int) -> np.ndarray:
    """テキスト配列を元の順序で動的パディングしながら ORT 推論し、ロジット (N, 2) を返す。"""
    input_names = {i.name for i in sess.get_inputs()}
    all_logits = []
    for start in range(0, len(texts), batch_size):
        enc = tok(
            texts[start : start + batch_size],
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="np",
        )
        feed = {k: enc[k].astype(np.int64) for k in input_names}
        logits = sess.run(None, feed)[0]
        all_logits.append(logits.astype(np.float64))
    return np.concatenate(all_logits, axis=0)


def summarize_latencies(lat_ms: list[float], total_sec: float) -> dict:
    """レイテンシ一覧 (ms) から要約統計を計算する。"""
    arr = np.asarray(lat_ms, dtype=np.float64)
    return {
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


def bench_model_cpu(sess: ort.InferenceSession, tok, texts: list[str], max_len: int) -> dict:
    """単一モデル・単一スレッド設定で実データ batch=1 計測と固定トークン長スイープを行う。"""
    input_names = [i.name for i in sess.get_inputs()]

    # 事前トークナイズ (ORT 単体計測用)
    pre_feeds = []
    for t in texts:
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="np")
        pre_feeds.append({k: enc[k].astype(np.int64) for k in input_names})

    # ウォームアップ 10回
    for feed in pre_feeds[:10]:
        sess.run(None, feed)

    # ベンチ A-1: ORT 単体 (dev 先頭200件, batch=1)
    ort_lats = []
    t_all0 = time.perf_counter()
    for feed in pre_feeds:
        t0 = time.perf_counter()
        sess.run(None, feed)
        ort_lats.append((time.perf_counter() - t0) * 1000.0)
    ort_total = time.perf_counter() - t_all0

    # ベンチ A-2: トークナイズ込み E2E (dev 先頭200件, batch=1)
    for t in texts[:5]:
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="np")
        sess.run(None, {k: enc[k].astype(np.int64) for k in input_names})

    e2e_lats = []
    t_e0 = time.perf_counter()
    for t in texts:
        t0 = time.perf_counter()
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="np")
        sess.run(None, {k: enc[k].astype(np.int64) for k in input_names})
        e2e_lats.append((time.perf_counter() - t0) * 1000.0)
    e2e_total = time.perf_counter() - t_e0

    # ベンチ B: 固定トークン長スイープ (batch=1)
    sweep = {}
    for seq_len in (64, 128, 256, 512, 1024):
        warmup = 5 if seq_len <= 256 else 3
        repeats = 30 if seq_len <= 256 else 15
        ids = np.full((1, seq_len), 100, dtype=np.int64)
        ids[0, 0] = tok.cls_token_id or 1
        ids[0, -1] = tok.sep_token_id or 2
        mask = np.ones((1, seq_len), dtype=np.int64)
        feed = {"input_ids": ids}
        if "attention_mask" in input_names:
            feed["attention_mask"] = mask

        for _ in range(warmup):
            sess.run(None, feed)
        lats = []
        ts0 = time.perf_counter()
        for _ in range(repeats):
            t0 = time.perf_counter()
            sess.run(None, feed)
            lats.append((time.perf_counter() - t0) * 1000.0)
        tot = time.perf_counter() - ts0
        sweep[str(seq_len)] = summarize_latencies(lats, tot)

    return {
        "real_dev_batch1_ort_only": summarize_latencies(ort_lats, ort_total),
        "real_dev_batch1_e2e_tok": summarize_latencies(e2e_lats, e2e_total),
        "seq_len_sweep_batch1": sweep,
    }


def eval_full_slices(probs: np.ndarray, gold: np.ndarray, rows: list, threshold: float) -> dict:
    """全体・サブカテゴリ・出典・512文字超スライスを算出する。"""
    overall = binary_metrics(probs, gold, threshold)

    subcat_indices = defaultdict(list)
    src_grp_indices = defaultdict(list)
    for idx, r in enumerate(rows):
        subcat_indices[r.get("subcategory", "?")].append(idx)
        src_grp_indices[source_group(r["source"])].append(idx)

    per_sub = {}
    for cat, idxs in sorted(subcat_indices.items()):
        arr = np.array(idxs, dtype=int)
        bm = binary_metrics(probs[arr], gold[arr], threshold)
        bm["correct"] = int(((probs[arr] >= threshold).astype(int) == gold[arr]).sum())
        per_sub[cat] = bm

    per_src = {}
    for grp, idxs in sorted(src_grp_indices.items()):
        arr = np.array(idxs, dtype=int)
        per_src[grp] = binary_metrics(probs[arr], gold[arr], threshold)

    mask_long = np.array([len(r["text"]) > 512 for r in rows], dtype=bool)
    long_slice = binary_metrics(probs[mask_long], gold[mask_long], threshold)

    return {
        "threshold": float(threshold),
        "overall": overall,
        "per_subcategory": per_sub,
        "per_source_group": per_src,
        "long_slice_gt512chars": long_slice,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="runs/select-modernbert/final")
    ap.add_argument("--fp32-onnx", default="onnx/model_fp32.onnx")
    ap.add_argument("--int8-onnx", default="onnx/model_int8.onnx")
    ap.add_argument("--dev", default="data/prod/dev.jsonl")
    ap.add_argument("--dev-fp32-probs", default="runs/select-modernbert/dev_probs.npy")
    ap.add_argument("--dev-fp32-eval", default="runs/select-modernbert/eval_select_cls.json")
    ap.add_argument("--out-dir", default="onnx")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--parity-n", type=int, default=200)
    ap.add_argument("--bench-n", type=int, default=200)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--dev-thresh", type=float, default=0.3395)
    ap.add_argument("--intra-threads", type=int, default=8)
    ap.add_argument("--skip-int8", action="store_true", help="FP32 一致検証・CPU ベンチのみ実施し INT8 を参照・評価しない")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    dev_rows = load_jsonl(args.dev)
    dev_texts = [r["text"] for r in dev_rows]
    dev_gold = np.array([int(r["label"]) for r in dev_rows], dtype=int)

    # ------------------------------------------------------------
    # 1. dev 先頭 200件で PyTorch FP32 (CPU) vs ONNX FP32 一致検証
    # ------------------------------------------------------------
    parity_rows = dev_rows[: args.parity_n]
    parity_texts = [r["text"] for r in parity_rows]

    pt_model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).float()
    pt_model.eval()
    pt_logits_list = []
    with torch.no_grad():
        for start in range(0, len(parity_texts), args.eval_batch):
            enc = tok(
                parity_texts[start : start + args.eval_batch],
                truncation=True,
                max_length=args.max_len,
                padding=True,
                return_tensors="pt",
            )
            out = pt_model(**enc)
            pt_logits_list.append(out.logits.detach().cpu().to(torch.float64).numpy())
    pt_logits = np.concatenate(pt_logits_list, axis=0)
    pt_probs = softmax_pos_prob(pt_logits)
    del pt_model

    sess_fp32_def = make_ort_session(args.fp32_onnx, intra_threads=0, inter_threads=0)
    ort_logits = run_ort_batches(sess_fp32_def, tok, parity_texts, args.max_len, args.eval_batch)
    ort_probs = softmax_pos_prob(ort_logits)

    logit_diff = np.abs(pt_logits - ort_logits)
    prob_diff = np.abs(pt_probs - ort_probs)
    agree_05 = int(((pt_probs >= 0.5) == (ort_probs >= 0.5)).sum())
    agree_thr = int(((pt_probs >= args.dev_thresh) == (ort_probs >= args.dev_thresh)).sum())

    parity = {
        "n": len(parity_rows),
        "pytorch_device": "cpu_fp32",
        "onnx_provider": "CPUExecutionProvider",
        "agree_at_0_5": agree_05,
        "agree_rate_0_5": round(agree_05 / len(parity_rows), 6),
        "agree_at_dev_thresh": agree_thr,
        "agree_rate_dev_thresh": round(agree_thr / len(parity_rows), 6),
        "dev_thresh": args.dev_thresh,
        "max_abs_logit_diff": float(logit_diff.max()),
        "mean_abs_logit_diff": float(logit_diff.mean()),
        "max_abs_prob_diff": float(prob_diff.max()),
        "mean_abs_prob_diff": float(prob_diff.mean()),
    }
    with open(os.path.join(args.out_dir, "onnx_parity_dev200.json"), "w", encoding="utf-8") as f:
        json.dump(parity, f, ensure_ascii=False, indent=2)
    print("PARITY_DEV200:", json.dumps(parity, ensure_ascii=False), flush=True)

    # ------------------------------------------------------------
    # 2. デスクトップ CPU 速度ベンチマーク (FP32 / [任意 INT8] × default / 8threads)
    # ------------------------------------------------------------
    bench_texts = dev_texts[: args.bench_n]
    sess_fp32_t8 = make_ort_session(args.fp32_onnx, intra_threads=args.intra_threads, inter_threads=1)
    sess_int8_def = None if args.skip_int8 else make_ort_session(args.int8_onnx, intra_threads=0, inter_threads=0)
    sess_int8_t8 = None if args.skip_int8 else make_ort_session(args.int8_onnx, intra_threads=args.intra_threads, inter_threads=1)

    thread_configs = {
        "default_threads": (sess_fp32_def, sess_int8_def, {"intra_op_num_threads": 0, "inter_op_num_threads": 0}),
        f"physical_{args.intra_threads}_threads": (sess_fp32_t8, sess_int8_t8, {"intra_op_num_threads": args.intra_threads, "inter_op_num_threads": 1}),
    }

    bench_results = {
        "env": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "onnxruntime_version": ort.__version__,
            "provider": "CPUExecutionProvider",
            "fp32_bytes": int(os.path.getsize(args.fp32_onnx)),
            "int8_bytes": None if args.skip_int8 else int(os.path.getsize(args.int8_onnx)),
            "skip_int8": bool(args.skip_int8),
            "bench_real_n": len(bench_texts),
            "max_len": args.max_len,
        },
        "configs": {},
    }

    for cfg_name, (s_fp32, s_int8, meta_t) in thread_configs.items():
        b_fp32 = bench_model_cpu(s_fp32, tok, bench_texts, args.max_len)
        entry = {"thread_settings": meta_t, "fp32": b_fp32}
        if s_int8 is not None:
            b_int8 = bench_model_cpu(s_int8, tok, bench_texts, args.max_len)
            speedup = {
                "real_ort_mean_speedup": round(
                    b_fp32["real_dev_batch1_ort_only"]["mean_ms"] / b_int8["real_dev_batch1_ort_only"]["mean_ms"], 3
                ),
                "real_ort_p50_speedup": round(
                    b_fp32["real_dev_batch1_ort_only"]["p50_ms"] / b_int8["real_dev_batch1_ort_only"]["p50_ms"], 3
                ),
                "real_e2e_mean_speedup": round(
                    b_fp32["real_dev_batch1_e2e_tok"]["mean_ms"] / b_int8["real_dev_batch1_e2e_tok"]["mean_ms"], 3
                ),
                "seq_len_mean_speedup": {
                    k: round(b_fp32["seq_len_sweep_batch1"][k]["mean_ms"] / b_int8["seq_len_sweep_batch1"][k]["mean_ms"], 3)
                    for k in b_fp32["seq_len_sweep_batch1"]
                },
            }
            entry["int8"] = b_int8
            entry["speedup_int8_over_fp32"] = speedup
        bench_results["configs"][cfg_name] = entry

    with open(os.path.join(args.out_dir, "onnx_bench_cpu.json"), "w", encoding="utf-8") as f:
        json.dump(bench_results, f, ensure_ascii=False, indent=2)
    print("BENCH_CPU_DONE", flush=True)
    del sess_fp32_def, sess_fp32_t8, sess_int8_def

    if args.skip_int8:
        print(
            json.dumps(
                {
                    "skip_int8": True,
                    "parity": parity,
                    "fp32_bytes": bench_results["env"]["fp32_bytes"],
                    "configs": {
                        k: {
                            "real_dev_batch1_ort_only": v["fp32"]["real_dev_batch1_ort_only"],
                            "real_dev_batch1_e2e_tok": v["fp32"]["real_dev_batch1_e2e_tok"],
                        }
                        for k, v in bench_results["configs"].items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    # ------------------------------------------------------------
    # 3. 動的 INT8 による dev.jsonl 全 5,000 件評価と PyTorch FP32 との比較
    # ------------------------------------------------------------
    # 最速側の設定 (t8 vs default) の内、物理8コアセッションで全件推論する
    t_inf0 = time.perf_counter()
    int8_logits = run_ort_batches(sess_int8_t8, tok, dev_texts, args.max_len, args.eval_batch)
    int8_eval_sec = time.perf_counter() - t_inf0
    int8_probs = softmax_pos_prob(int8_logits)
    np.save(os.path.join(args.out_dir, "dev_int8_probs.npy"), int8_probs)
    del sess_int8_t8

    int8_at_05 = eval_full_slices(int8_probs, dev_gold, dev_rows, 0.5)
    int8_at_dev_thr = eval_full_slices(int8_probs, dev_gold, dev_rows, args.dev_thresh)
    int8_fpr_1pct_op = find_threshold_fpr_1pct(int8_probs, dev_gold)

    # PyTorch FP32 (eval_select_cls.json ＆ dev_probs.npy) との F1 落差・予測一致率を算出
    fp32_eval = json.load(open(args.dev_fp32_eval, encoding="utf-8"))
    fp32_probs = np.load(args.dev_fp32_probs)
    fp32_at_05 = fp32_eval["threshold_0_5"]
    fp32_at_dev_03395 = binary_metrics(fp32_probs, dev_gold, args.dev_thresh)
    fp32_at_fpr1_exact = fp32_eval["fpr_1pct_operating_point"]

    def _diff_summary(m_int8: dict, m_fp32: dict, thr: float) -> dict:
        return {
            "delta_attack_recall": round(m_int8["attack_recall"] - m_fp32["attack_recall"], 6),
            "delta_benign_fpr": round(m_int8["benign_fpr"] - m_fp32["benign_fpr"], 6),
            "delta_precision": round(m_int8["precision"] - m_fp32["precision"], 6),
            "delta_f1_binary": round(m_int8["f1_binary"] - m_fp32["f1_binary"], 6),
            "delta_accuracy": round(m_int8["accuracy"] - m_fp32["accuracy"], 6),
            "pred_agree_vs_fp32": int(((int8_probs >= thr) == (fp32_probs >= thr)).sum()),
            "pred_agree_rate_vs_fp32": round(float(((int8_probs >= thr) == (fp32_probs >= thr)).mean()), 6),
        }

    eval_int8_out = {
        "int8_onnx_path": args.int8_onnx,
        "int8_bytes": int(os.path.getsize(args.int8_onnx)),
        "fp32_bytes": int(os.path.getsize(args.fp32_onnx)),
        "dev_n": len(dev_rows),
        "eval_batch": args.eval_batch,
        "intra_threads": args.intra_threads,
        "inference_sec": round(int8_eval_sec, 2),
        "samples_per_sec": round(len(dev_rows) / int8_eval_sec, 2),
        "max_abs_prob_diff_vs_fp32": float(np.abs(int8_probs - fp32_probs).max()),
        "mean_abs_prob_diff_vs_fp32": float(np.abs(int8_probs - fp32_probs).mean()),
        "threshold_0_5": int8_at_05,
        "threshold_dev_0_3395": int8_at_dev_thr,
        "int8_own_fpr_1pct_operating_point": int8_fpr_1pct_op,
        "fp32_reference": {
            "threshold_0_5": fp32_at_05,
            "threshold_dev_0_3395": fp32_at_dev_03395,
            "threshold_fpr_1pct_exact": fp32_at_fpr1_exact,
        },
        "delta_int8_minus_fp32": {
            "at_0_5": _diff_summary(int8_at_05["overall"], fp32_at_05, 0.5),
            "at_dev_0_3395": _diff_summary(int8_at_dev_thr["overall"], fp32_at_dev_03395, args.dev_thresh),
            "at_dev_0_3395_vs_fp32_exact": {
                "delta_attack_recall": round(int8_at_dev_thr["overall"]["attack_recall"] - fp32_at_fpr1_exact["attack_recall"], 6),
                "delta_benign_fpr": round(int8_at_dev_thr["overall"]["benign_fpr"] - fp32_at_fpr1_exact["benign_fpr"], 6),
                "delta_precision": round(int8_at_dev_thr["overall"]["precision"] - fp32_at_fpr1_exact["precision"], 6),
                "delta_f1_binary": round(int8_at_dev_thr["overall"]["f1_binary"] - fp32_at_fpr1_exact["f1_binary"], 6),
            },
        },
    }

    with open(os.path.join(args.out_dir, "eval_dev_int8.json"), "w", encoding="utf-8") as f:
        json.dump(eval_int8_out, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "int8_eval_sec": eval_int8_out["inference_sec"],
                "int8_at_0_5": int8_at_05["overall"],
                "int8_at_dev_0_3395": int8_at_dev_thr["overall"],
                "delta_int8_minus_fp32": eval_int8_out["delta_int8_minus_fp32"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
