#!/usr/bin/env python3
"""第5号 ja-phish-guard-v1: PyTorch FP32 vs ONNX FP32 一致検証および CPU (i7-11700F) 速度計測。

INT8 は対象外とし、FP32 のみで以下を計測・保存する:
- onnx/onnx_parity_dev200.json: dev 先頭200件および sealed-test 先頭200件での PT vs ONNX 予測・ロジット一致
- onnx/onnx_bench_cpu.json: ONNX Runtime CPU (default_threads / physical_8_threads) および PyTorch CPU 推論速度
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["benign", "benign_hard", "fraud", "phishing", "spam"]
ATTACK_IDX = [2, 3, 4]  # fraud, phishing, spam
THR_DEV_FPR1 = 8.077847599249799e-05


def load_texts(path: Path, n: int) -> list[str]:
    """JSONL から先頭 n 件のテキストを読み込む。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["text"])
            if len(out) >= n:
                break
    return out


def make_ort_session(
    onnx_path: Path, intra: int, inter: int
) -> ort.InferenceSession:
    """指定スレッド設定で CPUExecutionProvider セッションを作成する。"""
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra
    so.inter_op_num_threads = inter
    return ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )


def build_ort_feed(enc: dict, in_names: list[str]) -> dict[str, np.ndarray]:
    """トークナイザ出力から ONNX Runtime 用の int64 入力辞書を組み立てる。"""
    ids = np.asarray(enc["input_ids"], dtype=np.int64)
    feed = {}
    for name in in_names:
        if name in enc:
            feed[name] = np.asarray(enc[name], dtype=np.int64)
        elif name == "token_type_ids":
            feed[name] = np.zeros_like(ids, dtype=np.int64)
        else:
            raise KeyError(f"未対応の ONNX 入力名: {name}")
    return feed


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """数値安定版 softmax を最終軸へ適用する。"""
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def eval_parity(
    texts: list[str],
    tok,
    pt_model,
    ort_sess: ort.InferenceSession,
    max_len: int = 512,
) -> dict:
    """PyTorch CPU FP32 と ONNX CPU FP32 のロジット・確率・予測一致を計測する。"""
    in_names = [i.name for i in ort_sess.get_inputs()]
    pt_logits_all, ox_logits_all = [], []
    for txt in texts:
        enc_pt = tok(
            txt, truncation=True, max_length=max_len, return_tensors="pt"
        )
        with torch.no_grad():
            pt_lg = pt_model(**enc_pt).logits[0].float().numpy()

        enc_np = {k: v.numpy() for k, v in enc_pt.items()}
        ox_lg = ort_sess.run(None, build_ort_feed(enc_np, in_names))[0][0].astype(
            np.float32
        )
        pt_logits_all.append(pt_lg)
        ox_logits_all.append(ox_lg)

    pt_lg_arr = np.asarray(pt_logits_all, dtype=np.float64)
    ox_lg_arr = np.asarray(ox_logits_all, dtype=np.float64)
    pt_prob = softmax_np(pt_lg_arr)
    ox_prob = softmax_np(ox_lg_arr)
    pt_risk = pt_prob[:, ATTACK_IDX].sum(axis=1)
    ox_risk = ox_prob[:, ATTACK_IDX].sum(axis=1)

    pt_cls = pt_lg_arr.argmax(axis=1)
    ox_cls = ox_lg_arr.argmax(axis=1)
    n = len(texts)
    cls_match = int((pt_cls == ox_cls).sum())
    t05_match = int(((pt_risk >= 0.5) == (ox_risk >= 0.5)).sum())
    tfpr_match = int(
        ((pt_risk >= THR_DEV_FPR1) == (ox_risk >= THR_DEV_FPR1)).sum()
    )

    lg_diff = np.abs(pt_lg_arr - ox_lg_arr)
    pr_diff = np.abs(pt_prob - ox_prob)
    rk_diff = np.abs(pt_risk - ox_risk)
    return {
        "n": n,
        "class_argmax_matches": cls_match,
        "class_argmax_match_rate": round(cls_match / n, 6),
        "risk_t05_matches": t05_match,
        "risk_t05_match_rate": round(t05_match / n, 6),
        "risk_thr_fpr1_matches": tfpr_match,
        "risk_thr_fpr1_match_rate": round(tfpr_match / n, 6),
        "max_abs_logit_diff": float(f"{lg_diff.max():.6e}"),
        "mean_abs_logit_diff": float(f"{lg_diff.mean():.6e}"),
        "max_abs_prob_diff": float(f"{pr_diff.max():.6e}"),
        "mean_abs_prob_diff": float(f"{pr_diff.mean():.6e}"),
        "max_abs_risk_diff": float(f"{rk_diff.max():.6e}"),
        "mean_abs_risk_diff": float(f"{rk_diff.mean():.6e}"),
    }


def summarize_latencies(lat_sec: list[float], total_sec: float) -> dict:
    """レイテンシ一覧（秒）から ms 統計とスループットを算出する。"""
    ms = np.asarray(lat_sec, dtype=np.float64) * 1000.0
    n = len(lat_sec)
    return {
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


def bench_ort_config(
    ort_sess: ort.InferenceSession,
    tok,
    dev_texts: list[str],
    max_len: int = 512,
) -> dict:
    """1 つの ORT スレッド設定に対し、実データ ORT単体 / E2E / 固定長スイープを計測する。"""
    in_names = [i.name for i in ort_sess.get_inputs()]
    pre_feeds = [
        build_ort_feed(
            tok(t, truncation=True, max_length=max_len, return_tensors="np"),
            in_names,
        )
        for t in dev_texts
    ]
    for fd in pre_feeds[:10]:
        ort_sess.run(None, fd)

    lat_ort = []
    t0 = time.perf_counter()
    for fd in pre_feeds:
        ts = time.perf_counter()
        ort_sess.run(None, fd)
        lat_ort.append(time.perf_counter() - ts)
    tot_ort = time.perf_counter() - t0

    for t in dev_texts[:10]:
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="np")
        ort_sess.run(None, build_ort_feed(enc, in_names))

    lat_e2e = []
    t0 = time.perf_counter()
    for t in dev_texts:
        ts = time.perf_counter()
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="np")
        ort_sess.run(None, build_ort_feed(enc, in_names))
        lat_e2e.append(time.perf_counter() - ts)
    tot_e2e = time.perf_counter() - t0

    sweep_spec = [(64, 30), (128, 30), (256, 30), (512, 15)]
    sweep = {}
    for seq_len, n_rep in sweep_spec:
        dummy = {
            "input_ids": np.full((1, seq_len), 100, dtype=np.int64),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        }
        fd = build_ort_feed(dummy, in_names)
        for _ in range(3):
            ort_sess.run(None, fd)
        lats = []
        t0 = time.perf_counter()
        for _ in range(n_rep):
            ts = time.perf_counter()
            ort_sess.run(None, fd)
            lats.append(time.perf_counter() - ts)
        sweep[str(seq_len)] = summarize_latencies(lats, time.perf_counter() - t0)

    return {
        "real_dev_batch1_ort_only": summarize_latencies(lat_ort, tot_ort),
        "real_dev_batch1_e2e_tok": summarize_latencies(lat_e2e, tot_e2e),
        "seq_len_sweep_batch1": sweep,
    }


def bench_pytorch_cpu(
    pt_model,
    tok,
    dev_texts_50: list[str],
    max_len: int = 512,
    num_threads: int = 8,
) -> dict:
    """参考比較として PyTorch CPU FP32 の推論速度を計測する。"""
    torch.set_num_threads(num_threads)
    pre_enc = [
        tok(t, truncation=True, max_length=max_len, return_tensors="pt")
        for t in dev_texts_50
    ]
    with torch.no_grad():
        for enc in pre_enc[:5]:
            _ = pt_model(**enc).logits

        lat_fwd = []
        t0 = time.perf_counter()
        for enc in pre_enc:
            ts = time.perf_counter()
            _ = pt_model(**enc).logits
            lat_fwd.append(time.perf_counter() - ts)
        tot_fwd = time.perf_counter() - t0

        for t in dev_texts_50[:5]:
            enc = tok(t, truncation=True, max_length=max_len, return_tensors="pt")
            _ = pt_model(**enc).logits

        lat_e2e = []
        t0 = time.perf_counter()
        for t in dev_texts_50:
            ts = time.perf_counter()
            enc = tok(t, truncation=True, max_length=max_len, return_tensors="pt")
            _ = pt_model(**enc).logits
            lat_e2e.append(time.perf_counter() - ts)
        tot_e2e = time.perf_counter() - t0

        sweep = {}
        for seq_len in (64, 128, 256, 512):
            ids = torch.full((1, seq_len), 100, dtype=torch.long)
            mask = torch.ones((1, seq_len), dtype=torch.long)
            for _ in range(3):
                _ = pt_model(input_ids=ids, attention_mask=mask).logits
            lats = []
            t0 = time.perf_counter()
            for _ in range(10):
                ts = time.perf_counter()
                _ = pt_model(input_ids=ids, attention_mask=mask).logits
                lats.append(time.perf_counter() - ts)
            sweep[str(seq_len)] = summarize_latencies(lats, time.perf_counter() - t0)

    return {
        "torch_num_threads": num_threads,
        "real_dev50_batch1_forward_only": summarize_latencies(lat_fwd, tot_fwd),
        "real_dev50_batch1_e2e_tok": summarize_latencies(lat_e2e, tot_e2e),
        "seq_len_sweep_batch1": sweep,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="runs/prod-bert-v3/final")
    ap.add_argument("--onnx-fp32", default="onnx/model_fp32.onnx")
    ap.add_argument("--dev", default="data/prod/dev.jsonl")
    ap.add_argument("--sealed", default="data/prod/sealed-test.jsonl")
    ap.add_argument("--out-parity", default="onnx/onnx_parity_dev200.json")
    ap.add_argument("--out-bench", default="onnx/onnx_bench_cpu.json")
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    onnx_path = Path(args.onnx_fp32)

    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    pt_model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float32,
    ).eval()

    dev_200 = load_texts(Path(args.dev), 200)
    sealed_200 = load_texts(Path(args.sealed), 200)

    # 1. PyTorch vs ONNX FP32 一致検証
    sess_def = make_ort_session(onnx_path, 0, 0)
    parity = {
        "model_dir": str(model_dir).replace("\\", "/"),
        "onnx_fp32": str(onnx_path).replace("\\", "/"),
        "onnx_fp32_bytes": int(onnx_path.stat().st_size),
        "thr_dev_fpr1": THR_DEV_FPR1,
        "dev_200": eval_parity(dev_200, tok, pt_model, sess_def, args.max_len),
        "sealed_200": eval_parity(
            sealed_200, tok, pt_model, sess_def, args.max_len
        ),
    }
    Path(args.out_parity).write_text(
        json.dumps(parity, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2. CPU 推論速度計測 (default_threads / physical_8_threads / PyTorch参考)
    bench_default = bench_ort_config(sess_def, tok, dev_200, args.max_len)
    del sess_def
    sess_8 = make_ort_session(onnx_path, 8, 1)
    bench_8 = bench_ort_config(sess_8, tok, dev_200, args.max_len)
    del sess_8
    bench_pt = bench_pytorch_cpu(pt_model, tok, dev_200[:50], args.max_len, 8)

    bench = {
        "env": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "onnxruntime_version": ort.__version__,
            "torch_version": torch.__version__,
            "provider": "CPUExecutionProvider",
            "fp32_bytes": int(onnx_path.stat().st_size),
            "int8_created": False,
            "bench_real_n": len(dev_200),
            "max_len": args.max_len,
        },
        "configs": {
            "default_threads": {
                "thread_settings": {
                    "intra_op_num_threads": 0,
                    "inter_op_num_threads": 0,
                },
                "fp32": bench_default,
            },
            "physical_8_threads": {
                "thread_settings": {
                    "intra_op_num_threads": 8,
                    "inter_op_num_threads": 1,
                },
                "fp32": bench_8,
            },
        },
        "pytorch_cpu_fp32_ref": bench_pt,
    }
    Path(args.out_bench).write_text(
        json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("PARITY:", json.dumps(parity, ensure_ascii=False, indent=2))
    print("BENCH_ENV:", json.dumps(bench["env"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
