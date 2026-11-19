#!/usr/bin/env python
"""第6号 ja-error-detect-v1: PyTorch FP32 vs ONNX FP32 一致検証および CPU 推論速度計測。

INT8 は対象外とし、FP32 のみで以下を計測・保存する:
- onnx/onnx_parity.json: dev 先頭 200 件および sealed-test 先頭 200 件での PT vs ONNX 予測・ロジット一致
- onnx/onnx_bench_cpu.json: ONNX Runtime CPU (default_threads / physical_8_threads) および
  PyTorch CPU (threads=8) の推論速度（実データ 200 文＋固定トークン長スイープ）
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as ec  # noqa: E402
import infer_error_detect as ied  # noqa: E402


def load_rows(path: Path, n: int) -> List[dict]:
    """JSONL から先頭 n 件を読み込む。"""
    out: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def make_ort_session(onnx_path: Path, intra: int, inter: int) -> ort.InferenceSession:
    """指定スレッド設定で CPUExecutionProvider セッションを作成する。"""
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra
    so.inter_op_num_threads = inter
    return ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )


def build_ort_feed(enc: dict, in_names: List[str]) -> Dict[str, np.ndarray]:
    """トークナイザ出力から ONNX Runtime 用の int64 入力辞書を組み立てる。"""
    ids = np.asarray([enc["input_ids"]], dtype=np.int64)
    feed: Dict[str, np.ndarray] = {}
    for name in in_names:
        if name in enc:
            feed[name] = np.asarray([enc[name]], dtype=np.int64)
        elif name == "token_type_ids":
            feed[name] = np.zeros_like(ids, dtype=np.int64)
        else:
            raise KeyError(f"未対応の ONNX 入力名: {name}")
    return feed


def eval_parity(
    rows: List[dict],
    tok,
    pt_model,
    ort_sess: ort.InferenceSession,
    threshold: float = 0.80,
    max_len: int = 256,
) -> dict:
    """PyTorch CPU FP32 と ONNX CPU FP32 のロジット・確率・スパン・F1 一致を計測する。"""
    in_names = [i.name for i in ort_sess.get_inputs()]

    tok_total = 0
    tok_match = 0
    sent_span_match = 0
    char_total = 0
    char_match = 0

    max_lg_diff = 0.0
    sum_lg_diff = 0.0
    n_lg_elems = 0
    max_pr_diff = 0.0
    sum_pr_diff = 0.0
    n_pr_elems = 0

    pt_spans_all, pt_chars_all = [], []
    ox_spans_all, ox_chars_all = [], []

    for r in rows:
        txt = r["text"]
        enc, offs = ied.encode_with_offsets(tok, txt, max_len=max_len)

        kw = {
            "input_ids": torch.tensor([enc["input_ids"]], dtype=torch.long),
            "attention_mask": torch.tensor([enc["attention_mask"]], dtype=torch.long),
        }
        if "token_type_ids" in enc:
            kw["token_type_ids"] = torch.tensor([enc["token_type_ids"]], dtype=torch.long)

        with torch.no_grad():
            pt_lg = pt_model(**kw).logits[0].float().numpy()
        ox_lg = ort_sess.run(None, build_ort_feed(enc, in_names))[0][0].astype(np.float32)

        pt_prob = ied._softmax(pt_lg.astype(np.float64))
        ox_prob = ied._softmax(ox_lg.astype(np.float64))

        pt_arg = pt_lg.argmax(axis=-1)
        ox_arg = ox_lg.argmax(axis=-1)
        tok_total += len(pt_arg)
        tok_match += int((pt_arg == ox_arg).sum())

        lg_diff = np.abs(pt_lg.astype(np.float64) - ox_lg.astype(np.float64))
        pr_diff = np.abs(pt_prob - ox_prob)
        max_lg_diff = max(max_lg_diff, float(lg_diff.max()))
        sum_lg_diff += float(lg_diff.sum())
        n_lg_elems += int(lg_diff.size)
        max_pr_diff = max(max_pr_diff, float(pr_diff.max()))
        sum_pr_diff += float(pr_diff.sum())
        n_pr_elems += int(pr_diff.size)

        pt_sp_dicts, pt_cp = ied.decode_probs_to_spans(
            txt, offs, pt_prob.astype(np.float32), "span_mean_err", threshold
        )
        ox_sp_dicts, ox_cp = ied.decode_probs_to_spans(
            txt, offs, ox_prob.astype(np.float32), "span_mean_err", threshold
        )
        pt_sp = [(s["start"], s["end"]) for s in pt_sp_dicts]
        ox_sp = [(s["start"], s["end"]) for s in ox_sp_dicts]

        if pt_sp == ox_sp:
            sent_span_match += 1
        char_total += len(txt)
        char_match += sum(1 for a, b in zip(pt_cp, ox_cp) if a == b)

        pt_spans_all.append(pt_sp)
        pt_chars_all.append(pt_cp)
        ox_spans_all.append(ox_sp)
        ox_chars_all.append(ox_cp)

    pt_ev = ec.evaluate_predictions(rows, pt_spans_all, pt_chars_all)
    ox_ev = ec.evaluate_predictions(rows, ox_spans_all, ox_chars_all)

    n = len(rows)
    return {
        "n_sentences": n,
        "n_chars": char_total,
        "token_argmax_matches": tok_match,
        "token_argmax_total": tok_total,
        "token_argmax_match_rate": round(tok_match / tok_total, 6) if tok_total else 0.0,
        "span_sentence_matches": sent_span_match,
        "span_sentence_match_rate": round(sent_span_match / n, 6) if n else 0.0,
        "char_bio_matches": char_match,
        "char_bio_match_rate": round(char_match / char_total, 6) if char_total else 0.0,
        "max_abs_logit_diff": float(f"{max_lg_diff:.6e}"),
        "mean_abs_logit_diff": float(f"{(sum_lg_diff / max(1, n_lg_elems)):.6e}"),
        "max_abs_prob_diff": float(f"{max_pr_diff:.6e}"),
        "mean_abs_prob_diff": float(f"{(sum_pr_diff / max(1, n_pr_elems)):.6e}"),
        "pytorch_exact_span": pt_ev["exact_span"],
        "onnx_exact_span": ox_ev["exact_span"],
    }


def summarize_latencies(lat_sec: List[float], total_sec: float, total_chars: int = 0) -> dict:
    """レイテンシ一覧（秒）から ms 統計とスループットを算出する。"""
    ms = np.asarray(lat_sec, dtype=np.float64) * 1000.0
    n = len(lat_sec)
    out = {
        "n": n,
        "total_sec": round(total_sec, 4),
        "samples_per_sec": round(n / max(1e-9, total_sec), 3),
        "mean_ms": round(float(ms.mean()), 3),
        "p50_ms": round(float(np.percentile(ms, 50)), 3),
        "p95_ms": round(float(np.percentile(ms, 95)), 3),
        "p99_ms": round(float(np.percentile(ms, 99)), 3),
        "min_ms": round(float(ms.min()), 3),
        "max_ms": round(float(ms.max()), 3),
    }
    if total_chars > 0:
        out["chars_per_sec"] = round(total_chars / max(1e-9, total_sec), 1)
    return out


def bench_ort_real(
    rows: List[dict],
    tok,
    sess: ort.InferenceSession,
    threshold: float = 0.80,
    max_len: int = 256,
    warmup: int = 10,
) -> dict:
    """ONNX Runtime 実データ推論の Forward 単体・E2E 速度を計測する。"""
    in_names = [i.name for i in sess.get_inputs()]
    feeds = []
    for r in rows:
        enc, _ = ied.encode_with_offsets(tok, r["text"], max_len=max_len)
        feeds.append(build_ort_feed(enc, in_names))

    for i in range(warmup):
        sess.run(None, feeds[i % len(feeds)])

    lat_fwd = []
    t0 = time.perf_counter()
    for fd in feeds:
        s0 = time.perf_counter()
        sess.run(None, fd)
        lat_fwd.append(time.perf_counter() - s0)
    tot_fwd = time.perf_counter() - t0

    for i in range(warmup):
        txt = rows[i % len(rows)]["text"]
        enc, offs = ied.encode_with_offsets(tok, txt, max_len=max_len)
        lg = sess.run(None, build_ort_feed(enc, in_names))[0][0].astype(np.float32)
        ied.decode_probs_to_spans(txt, offs, ied._softmax(lg), "span_mean_err", threshold)

    lat_e2e = []
    t1 = time.perf_counter()
    for r in rows:
        txt = r["text"]
        s0 = time.perf_counter()
        enc, offs = ied.encode_with_offsets(tok, txt, max_len=max_len)
        lg = sess.run(None, build_ort_feed(enc, in_names))[0][0].astype(np.float32)
        ied.decode_probs_to_spans(txt, offs, ied._softmax(lg), "span_mean_err", threshold)
        lat_e2e.append(time.perf_counter() - s0)
    tot_e2e = time.perf_counter() - t1

    total_chars = sum(len(r["text"]) for r in rows)
    return {
        "ort_forward_only": summarize_latencies(lat_fwd, tot_fwd, total_chars),
        "e2e_tok_decode": summarize_latencies(lat_e2e, tot_e2e, total_chars),
    }


def bench_pt_real(
    rows: List[dict],
    tok,
    pt_model,
    threshold: float = 0.80,
    max_len: int = 256,
    warmup: int = 5,
) -> dict:
    """PyTorch CPU FP32 実データ推論速度を計測する。"""
    tensors = []
    for r in rows:
        enc, _ = ied.encode_with_offsets(tok, r["text"], max_len=max_len)
        tensors.append({k: torch.tensor([v], dtype=torch.long) for k, v in enc.items()})

    with torch.no_grad():
        for i in range(warmup):
            _ = pt_model(**tensors[i % len(tensors)]).logits

        lat_fwd = []
        t0 = time.perf_counter()
        for kw in tensors:
            s0 = time.perf_counter()
            _ = pt_model(**kw).logits
            lat_fwd.append(time.perf_counter() - s0)
        tot_fwd = time.perf_counter() - t0

        lat_e2e = []
        t1 = time.perf_counter()
        for r in rows:
            txt = r["text"]
            s0 = time.perf_counter()
            enc, offs = ied.encode_with_offsets(tok, txt, max_len=max_len)
            kw = {k: torch.tensor([v], dtype=torch.long) for k, v in enc.items()}
            lg = pt_model(**kw).logits[0].float().numpy()
            ied.decode_probs_to_spans(txt, offs, ied._softmax(lg), "span_mean_err", threshold)
            lat_e2e.append(time.perf_counter() - s0)
        tot_e2e = time.perf_counter() - t1

    total_chars = sum(len(r["text"]) for r in rows)
    return {
        "pt_forward_only": summarize_latencies(lat_fwd, tot_fwd, total_chars),
        "e2e_tok_decode": summarize_latencies(lat_e2e, tot_e2e, total_chars),
    }


def bench_fixed_seq_sweep(
    sess_def: ort.InferenceSession,
    sess_8: ort.InferenceSession,
    pt_model,
    seq_lens: Tuple[int, ...] = (32, 64, 128, 256),
) -> dict:
    """固定トークン長（batch=1）スイープ計測。"""
    in_names = [i.name for i in sess_def.get_inputs()]
    out = {}
    for sl in seq_lens:
        ids_np = np.full((1, sl), 100, dtype=np.int64)
        mask_np = np.ones((1, sl), dtype=np.int64)
        tt_np = np.zeros((1, sl), dtype=np.int64)
        feed = {"input_ids": ids_np, "attention_mask": mask_np}
        if "token_type_ids" in in_names:
            feed["token_type_ids"] = tt_np

        def run_ort(sess, iters=30):
            for _ in range(5):
                sess.run(None, feed)
            lats = []
            t0 = time.perf_counter()
            for _ in range(iters):
                s0 = time.perf_counter()
                sess.run(None, feed)
                lats.append(time.perf_counter() - s0)
            return summarize_latencies(lats, time.perf_counter() - t0)

        kw = {
            "input_ids": torch.tensor(ids_np, dtype=torch.long),
            "attention_mask": torch.tensor(mask_np, dtype=torch.long),
            "token_type_ids": torch.tensor(tt_np, dtype=torch.long),
        }
        with torch.no_grad():
            for _ in range(3):
                _ = pt_model(**kw).logits
            pt_lats = []
            t0 = time.perf_counter()
            for _ in range(10):
                s0 = time.perf_counter()
                _ = pt_model(**kw).logits
                pt_lats.append(time.perf_counter() - s0)
            pt_stat = summarize_latencies(pt_lats, time.perf_counter() - t0)

        out[str(sl)] = {
            "onnx_default_threads": run_ort(sess_def, iters=30),
            "onnx_physical_8_threads": run_ort(sess_8, iters=30),
            "pytorch_cpu_threads_8": pt_stat,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="runs/prod-bert-v3/final")
    ap.add_argument("--onnx-fp32", default="onnx/model_fp32.onnx")
    ap.add_argument("--dev", default="data/prod/dev.jsonl")
    ap.add_argument("--sealed", default="data/prod/sealed-test.jsonl")
    ap.add_argument("--out-dir", default="onnx")
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(args.onnx_fp32)

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    pt_model = AutoModelForTokenClassification.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.float32
    ).eval()
    torch.set_num_threads(8)

    sess_def = make_ort_session(onnx_path, intra=0, inter=0)
    sess_8 = make_ort_session(onnx_path, intra=8, inter=1)

    dev200 = load_rows(Path(args.dev), 200)
    sealed200 = load_rows(Path(args.sealed), 200)

    print("[1/2] PyTorch FP32 vs ONNX FP32 一致検証を実行中 (dev 200 / sealed 200)...")
    parity = {
        "model_dir": str(args.model_dir).replace("\\", "/"),
        "onnx_fp32": str(onnx_path).replace("\\", "/"),
        "threshold": args.threshold,
        "decoding_method": "span_mean_err",
        "dev_200": eval_parity(dev200, tok, pt_model, sess_def, args.threshold, args.max_len),
        "sealed_200": eval_parity(
            sealed200, tok, pt_model, sess_def, args.threshold, args.max_len
        ),
    }
    (out_dir / "onnx_parity.json").write_text(
        json.dumps(parity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"  dev200: span_match={parity['dev_200']['span_sentence_matches']}/200, "
        f"tok_match={parity['dev_200']['token_argmax_match_rate']}, "
        f"max_logit_diff={parity['dev_200']['max_abs_logit_diff']}"
    )
    print(
        f"  sealed200: span_match={parity['sealed_200']['span_sentence_matches']}/200, "
        f"tok_match={parity['sealed_200']['token_argmax_match_rate']}, "
        f"max_logit_diff={parity['sealed_200']['max_abs_logit_diff']}"
    )

    print("[2/2] デスクトップ CPU 推論速度計測を実行中 (batch=1)...")
    bench = {
        "cpu_platform": platform.processor(),
        "logical_cores": os.cpu_count(),
        "torch_version": torch.__version__,
        "onnxruntime_version": ort.__version__,
        "real_dev_200": {
            "onnx_default_threads": bench_ort_real(
                dev200, tok, sess_def, args.threshold, args.max_len
            ),
            "onnx_physical_8_threads": bench_ort_real(
                dev200, tok, sess_8, args.threshold, args.max_len
            ),
            "pytorch_cpu_threads_8_first50": bench_pt_real(
                dev200[:50], tok, pt_model, args.threshold, args.max_len
            ),
        },
        "fixed_seq_sweep_batch1": bench_fixed_seq_sweep(sess_def, sess_8, pt_model),
    }
    (out_dir / "onnx_bench_cpu.json").write_text(
        json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"  ONNX(default) E2E mean={bench['real_dev_200']['onnx_default_threads']['e2e_tok_decode']['mean_ms']}ms | "
        f"ONNX(8thr) E2E mean={bench['real_dev_200']['onnx_physical_8_threads']['e2e_tok_decode']['mean_ms']}ms | "
        f"PT(8thr,50) E2E mean={bench['real_dev_200']['pytorch_cpu_threads_8_first50']['e2e_tok_decode']['mean_ms']}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
