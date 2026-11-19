#!/usr/bin/env python3
"""第7号 ja-address-ner-v1 PyTorch FP32 vs ONNX FP32 予測一致検証および CPU 処理時間計測。

1. dev.jsonl 先頭 200 件 + sealed-test.jsonl 先頭 200 件（計 400 件）で
   トークン argmax 一致、文単位の住所スパン完全一致、5要素スパン完全一致、
   リンカー出力完全一致、ロジット差、Exact F1 を検証し onnx/onnx_parity.json に保存する。
2. dev.jsonl 先頭 200 件（batch=1）で ONNX Runtime（default_threads / physical_8_threads）
   および参考 PyTorch CPU（threads=8, 50件）の推論速度（Forward / 第1段階 E2E /
   第1+2段階リンカー E2E）を計測し onnx/onnx_bench_cpu.json に保存する。
"""
import argparse
import json
import os
import platform
import subprocess
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
import infer_address as ia  # noqa: E402
from release_linker.linker import AddressLinker  # noqa: E402


def _cpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            text=True,
            timeout=5,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor()


def _stats_ms(times_sec: List[float], total_chars: int) -> dict:
    ms = sorted(t * 1000.0 for t in times_sec)
    n = len(ms)
    mean_ms = sum(ms) / n
    p50 = ms[n // 2]
    p95 = ms[min(n - 1, int(n * 0.95))]
    p99 = ms[min(n - 1, int(n * 0.99))]
    total_s = sum(times_sec)
    return {
        "n": n,
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "sentences_per_sec": round(n / total_s, 3) if total_s > 0 else 0.0,
        "chars_per_sec": round(total_chars / total_s, 1) if total_s > 0 else 0.0,
    }


def _addr_exact_f1(rows: List[dict], preds_list: List[List[dict]]) -> dict:
    tp = fp = fn = 0
    for r, ps in zip(rows, preds_list):
        gs = {(int(a["start"]), int(a["end"])) for a in r["addresses"]}
        pss = {(int(a["start"]), int(a["end"])) for a in ps}
        tp += len(gs & pss)
        fp += len(pss - gs)
        fn += len(gs - pss)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * rc / (p + rc) if (p + rc) else 0.0
    return {"precision": round(p, 4), "recall": round(rc, 4), "f1": round(f1, 4)}


def verify_parity_split(
    split_name: str,
    rows: List[dict],
    tok,
    pt_model,
    ort_sess,
    linker: AddressLinker,
    max_len: int = 384,
) -> dict:
    ort_in_names = {i.name for i in ort_sess.get_inputs()}

    total_tokens = 0
    argmax_match_tokens = 0
    sent_addr_match = 0
    sent_comp_match = 0
    sent_link_match = 0
    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    total_logits_elems = 0
    total_chars = 0

    pt_addrs_all: List[List[dict]] = []
    ox_addrs_all: List[List[dict]] = []

    for r in rows:
        txt = r["text"]
        total_chars += len(txt)
        enc, offsets = ia.encode_with_offsets(tok, txt, max_len)

        kw_pt = {
            "input_ids": torch.tensor([enc["input_ids"]], dtype=torch.long),
            "attention_mask": torch.tensor([enc["attention_mask"]], dtype=torch.long),
        }
        if "token_type_ids" in enc:
            kw_pt["token_type_ids"] = torch.tensor([enc["token_type_ids"]], dtype=torch.long)

        with torch.no_grad():
            pt_logits = pt_model(**kw_pt).logits[0].cpu().numpy()

        feed_ox = {}
        for k in ("input_ids", "attention_mask", "token_type_ids"):
            if k in ort_in_names:
                vals = enc.get(k, [0] * len(enc["input_ids"]))
                feed_ox[k] = np.asarray([vals], dtype=np.int64)
        ox_logits = ort_sess.run(None, feed_ox)[0][0]

        diff = np.abs(pt_logits - ox_logits)
        max_abs_diff = max(max_abs_diff, float(diff.max()))
        sum_abs_diff += float(diff.sum())
        total_logits_elems += int(diff.size)

        pt_ids = [int(x) for x in pt_logits.argmax(axis=-1)]
        ox_ids = [int(x) for x in ox_logits.argmax(axis=-1)]
        total_tokens += len(pt_ids)
        argmax_match_tokens += sum(1 for a, b in zip(pt_ids, ox_ids) if a == b)

        pt_comps = ia.decode_token_preds_to_components(txt, offsets, pt_ids)
        ox_comps = ia.decode_token_preds_to_components(txt, offsets, ox_ids)
        if pt_comps == ox_comps:
            sent_comp_match += 1

        pt_addrs = ia.group_components_to_addresses(txt, pt_comps)
        ox_addrs = ia.group_components_to_addresses(txt, ox_comps)
        pt_addrs_all.append(pt_addrs)
        ox_addrs_all.append(ox_addrs)
        if [(a["start"], a["end"]) for a in pt_addrs] == [
            (a["start"], a["end"]) for a in ox_addrs
        ]:
            sent_addr_match += 1

        pt_links = [
            (a["start"], a["end"], linker.link_components(a["components"])["municipality_code"])
            for a in pt_addrs
        ]
        ox_links = [
            (a["start"], a["end"], linker.link_components(a["components"])["municipality_code"])
            for a in ox_addrs
        ]
        if pt_links == ox_links:
            sent_link_match += 1

    n = len(rows)
    return {
        "split": split_name,
        "sentences": n,
        "chars": total_chars,
        "tokens": total_tokens,
        "token_argmax_match": argmax_match_tokens,
        "token_argmax_match_rate": round(argmax_match_tokens / total_tokens, 6),
        "sentence_address_span_exact_match": sent_addr_match,
        "sentence_address_span_exact_match_rate": round(sent_addr_match / n, 6),
        "sentence_component_span_exact_match": sent_comp_match,
        "sentence_component_span_exact_match_rate": round(sent_comp_match / n, 6),
        "sentence_linker_exact_match": sent_link_match,
        "sentence_linker_exact_match_rate": round(sent_link_match / n, 6),
        "max_logit_abs_diff": float(f"{max_abs_diff:.6e}"),
        "mean_logit_abs_diff": float(f"{(sum_abs_diff / total_logits_elems):.6e}"),
        "pt_address_exact": _addr_exact_f1(rows, pt_addrs_all),
        "onnx_address_exact": _addr_exact_f1(rows, ox_addrs_all),
    }


def benchmark_ort(
    label: str,
    rows: List[dict],
    tok,
    onnx_path: Path,
    linker: AddressLinker,
    intra_threads: int,
    inter_threads: int,
    max_len: int = 384,
) -> dict:
    opts = ort.SessionOptions()
    if intra_threads > 0:
        opts.intra_op_num_threads = intra_threads
        opts.inter_op_num_threads = inter_threads
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    in_names = {i.name for i in sess.get_inputs()}

    # 事前エンコード（Forward 単体計測用）
    feeds = []
    for r in rows:
        enc, _ = ia.encode_with_offsets(tok, r["text"], max_len)
        fd = {}
        for k in ("input_ids", "attention_mask", "token_type_ids"):
            if k in in_names:
                vals = enc.get(k, [0] * len(enc["input_ids"]))
                fd[k] = np.asarray([vals], dtype=np.int64)
        feeds.append(fd)

    # ウォームアップ 10 件
    for fd in feeds[:10]:
        sess.run(None, fd)

    total_chars = sum(len(r["text"]) for r in rows)

    # 1) ORT Forward 単体
    t_fwd: List[float] = []
    for fd in feeds:
        t0 = time.perf_counter()
        _ = sess.run(None, fd)[0]
        t_fwd.append(time.perf_counter() - t0)

    # 2) 第1段階 E2E（トークナイズ + ORT Forward + 復号・グルーピング）
    t_s1: List[float] = []
    for r in rows:
        t0 = time.perf_counter()
        enc, offs = ia.encode_with_offsets(tok, r["text"], max_len)
        fd = {}
        for k in ("input_ids", "attention_mask", "token_type_ids"):
            if k in in_names:
                vals = enc.get(k, [0] * len(enc["input_ids"]))
                fd[k] = np.asarray([vals], dtype=np.int64)
        pred_ids = [int(x) for x in sess.run(None, fd)[0][0].argmax(axis=-1)]
        comps = ia.decode_token_preds_to_components(r["text"], offs, pred_ids)
        _ = ia.group_components_to_addresses(r["text"], comps)
        t_s1.append(time.perf_counter() - t0)

    # 3) 第1段階 + 第2段階リンカー E2E
    t_s2: List[float] = []
    for r in rows:
        t0 = time.perf_counter()
        enc, offs = ia.encode_with_offsets(tok, r["text"], max_len)
        fd = {}
        for k in ("input_ids", "attention_mask", "token_type_ids"):
            if k in in_names:
                vals = enc.get(k, [0] * len(enc["input_ids"]))
                fd[k] = np.asarray([vals], dtype=np.int64)
        pred_ids = [int(x) for x in sess.run(None, fd)[0][0].argmax(axis=-1)]
        comps = ia.decode_token_preds_to_components(r["text"], offs, pred_ids)
        addrs = ia.group_components_to_addresses(r["text"], comps)
        for a in addrs:
            _ = linker.link_components(a["components"])
        t_s2.append(time.perf_counter() - t0)

    # 固定長スイープ（batch=1, 30回）
    sweep = {}
    for seq_len in (32, 64, 128, 256):
        fd_s = {
            k: np.ones((1, seq_len), dtype=np.int64)
            for k in ("input_ids", "attention_mask")
            if k in in_names
        }
        if "token_type_ids" in in_names:
            fd_s["token_type_ids"] = np.zeros((1, seq_len), dtype=np.int64)
        for _ in range(5):
            sess.run(None, fd_s)
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            sess.run(None, fd_s)
            ts.append(time.perf_counter() - t0)
        sweep[str(seq_len)] = _stats_ms(ts, seq_len * 30)

    del sess
    return {
        "config": label,
        "intra_threads": intra_threads,
        "inter_threads": inter_threads,
        "forward_only": _stats_ms(t_fwd, total_chars),
        "stage1_e2e": _stats_ms(t_s1, total_chars),
        "stage1_plus_linker_e2e": _stats_ms(t_s2, total_chars),
        "seq_len_sweep": sweep,
    }


def benchmark_pytorch_cpu(
    rows: List[dict],
    tok,
    pt_model,
    linker: AddressLinker,
    threads: int = 8,
    max_len: int = 384,
) -> dict:
    torch.set_num_threads(threads)
    feeds = []
    for r in rows:
        enc, _ = ia.encode_with_offsets(tok, r["text"], max_len)
        kw = {
            "input_ids": torch.tensor([enc["input_ids"]], dtype=torch.long),
            "attention_mask": torch.tensor([enc["attention_mask"]], dtype=torch.long),
        }
        if "token_type_ids" in enc:
            kw["token_type_ids"] = torch.tensor([enc["token_type_ids"]], dtype=torch.long)
        feeds.append(kw)

    with torch.no_grad():
        for kw in feeds[:5]:
            _ = pt_model(**kw).logits

        total_chars = sum(len(r["text"]) for r in rows)
        t_fwd: List[float] = []
        for kw in feeds:
            t0 = time.perf_counter()
            _ = pt_model(**kw).logits
            t_fwd.append(time.perf_counter() - t0)

        t_s2: List[float] = []
        for r in rows:
            t0 = time.perf_counter()
            enc, offs = ia.encode_with_offsets(tok, r["text"], max_len)
            kw = {
                "input_ids": torch.tensor([enc["input_ids"]], dtype=torch.long),
                "attention_mask": torch.tensor([enc["attention_mask"]], dtype=torch.long),
            }
            if "token_type_ids" in enc:
                kw["token_type_ids"] = torch.tensor([enc["token_type_ids"]], dtype=torch.long)
            pred_ids = [int(x) for x in pt_model(**kw).logits[0].argmax(dim=-1).tolist()]
            comps = ia.decode_token_preds_to_components(r["text"], offs, pred_ids)
            addrs = ia.group_components_to_addresses(r["text"], comps)
            for a in addrs:
                _ = linker.link_components(a["components"])
            t_s2.append(time.perf_counter() - t0)

    return {
        "threads": threads,
        "forward_only": _stats_ms(t_fwd, total_chars),
        "stage1_plus_linker_e2e": _stats_ms(t_s2, total_chars),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=r"runs/prod-bert-v3/final")
    ap.add_argument("--onnx-path", default=r"onnx/model_fp32.onnx")
    ap.add_argument("--master-dir", default=r"data/master")
    ap.add_argument("--dev", default=r"data/prod/dev.jsonl")
    ap.add_argument("--sealed", default=r"data/prod/sealed-test.jsonl")
    ap.add_argument("--n-parity", type=int, default=200)
    ap.add_argument("--n-bench", type=int, default=200)
    ap.add_argument("--out-dir", default=r"onnx")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(args.onnx_path)

    dev_rows = [json.loads(l) for i, l in enumerate(open(args.dev, encoding="utf-8")) if i < args.n_parity]
    sealed_rows = [json.loads(l) for i, l in enumerate(open(args.sealed, encoding="utf-8")) if i < args.n_parity]

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    pt_model = AutoModelForTokenClassification.from_pretrained(
        args.model_dir, local_files_only=True
    ).to("cpu")
    pt_model.eval()
    torch.set_num_threads(8)
    linker = AddressLinker(args.master_dir)

    ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    parity_dev = verify_parity_split("dev", dev_rows, tok, pt_model, ort_sess, linker)
    parity_sealed = verify_parity_split("sealed-test", sealed_rows, tok, pt_model, ort_sess, linker)
    del ort_sess

    parity_report = {
        "model_dir": str(args.model_dir).replace("\\", "/"),
        "onnx_path": str(onnx_path).replace("\\", "/"),
        "dev_200": parity_dev,
        "sealed_200": parity_sealed,
    }
    (out_dir / "onnx_parity.json").write_text(
        json.dumps(parity_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    bench_rows = dev_rows[: args.n_bench]
    bench_default = benchmark_ort("default_threads", bench_rows, tok, onnx_path, linker, 0, 0)
    bench_phys8 = benchmark_ort("physical_8_threads", bench_rows, tok, onnx_path, linker, 8, 1)
    bench_pt = benchmark_pytorch_cpu(bench_rows[:50], tok, pt_model, linker, threads=8)

    bench_report = {
        "cpu_name": _cpu_name(),
        "logical_cores": os.cpu_count(),
        "onnxruntime_version": ort.__version__,
        "torch_version": torch.__version__,
        "bench_rows": len(bench_rows),
        "onnx_default_threads": bench_default,
        "onnx_physical_8_threads": bench_phys8,
        "pytorch_cpu_8_threads_50": bench_pt,
    }
    (out_dir / "onnx_bench_cpu.json").write_text(
        json.dumps(bench_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== PARITY ===")
    print(json.dumps(parity_report, ensure_ascii=False, indent=2))
    print("=== BENCH ===")
    print(
        json.dumps(
            {
                "cpu_name": bench_report["cpu_name"],
                "default_s1_e2e_ms": bench_default["stage1_e2e"]["mean_ms"],
                "default_s2_e2e_ms": bench_default["stage1_plus_linker_e2e"]["mean_ms"],
                "phys8_s1_e2e_ms": bench_phys8["stage1_e2e"]["mean_ms"],
                "phys8_s2_e2e_ms": bench_phys8["stage1_plus_linker_e2e"]["mean_ms"],
                "pt_s2_e2e_ms": bench_pt["stage1_plus_linker_e2e"]["mean_ms"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
