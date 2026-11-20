#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 PyTorch FP32 vs ONNX FP32 一致検証＆CPU速度ベンチマークスクリプト。

検証項目:
1. PyTorch FP32 vs ONNX FP32 一致検証 (onnx/onnx_parity.json):
   - release-staging/ 単体読み込み (local_files_only=True) を用い、
     data/prod/dev.jsonl 先頭 200 件および data/prod/sealed-test.jsonl 先頭 200 件で
     文単位/スパン単位の読み・発音一致率、多読スパン選択一致率、ロジット最大/平均絶対差を計測。
   - 封印テスト本文は出力 JSON や標準出力ログに一切出力しない。
   - dev 200 件について旧参照実装 code/reading_tagger.py とステージング単体版の 200/200 完全一致も確認する。
2. デスクトップ CPU (batch=1) 推論速度計測 (onnx/onnx_bench_cpu.json):
   - ONNX default_threads / physical_8_threads (Forward 単体・E2E、dev 先頭 200 件)
   - 参考 PyTorch CPU FP32 (threads=8、dev 先頭 50 件)
   - 固定トークン長スイープ [32, 64, 128, 256]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import onnxruntime as ort
import torch
import transformers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import eval_reading as er  # noqa: E402
from reading_tagger import JaReadingTagger as LegacyTagger  # noqa: E402


def load_jsonl_head(path: Path, n: int) -> List[dict]:
    """JSONL ファイルから先頭 n 件を読み込む。"""
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def get_cpu_name() -> str:
    """Windows レジストリまたは platform から CPU 名を取得する。"""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if val:
                return str(val).strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def load_staged_module(staging_dir: Path):
    """__pycache__ を汚さずに release-staging/examples/reading_tagger.py を読み込む。"""
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        mod_path = staging_dir / "examples" / "reading_tagger.py"
        spec = importlib.util.spec_from_file_location("staged_reading_bench", str(mod_path))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.dont_write_bytecode = prev


def eval_parity_split(
    tagger_pt,
    tagger_ox,
    rows: List[dict],
    catalog_keys: set,
    split_label: str,
) -> Tuple[dict, List[List[dict]]]:
    """PT vs ONNX を 1 文ずつ (batch=1) 比較し、全パリティ指標を返す。"""
    texts = [r["text"] for r in rows]
    total_chars = sum(len(t) for t in texts)

    pt_preds, pt_lmeta = tagger_pt.predict_batch_with_logits(texts, batch_size=1)
    ox_preds, ox_lmeta = tagger_ox.predict_batch_with_logits(texts, batch_size=1)

    sent_read_match = 0
    sent_pron_match = 0
    span_total = 0
    span_read_match = 0
    span_pron_match = 0
    amb_total = 0
    amb_match = 0

    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    total_logit_elems = 0

    for txt, sp_pt, sp_ox, (lg_pt, m_pt), (lg_ox, m_ox) in zip(
        texts, pt_preds, ox_preds, pt_lmeta, ox_lmeta
    ):
        assert m_pt == m_ox
        assert len(sp_pt) == len(sp_ox)

        # スパン構造・読み・発音比較
        r_ok = True
        p_ok = True
        ext_sps = tagger_pt.extractor.extract_spans(txt)
        amb_positions = {
            (sp["start"], sp["end"])
            for sp in ext_sps
            if sp["key"] in catalog_keys and len(sp["candidates"]) >= 2
        }

        for s1, s2 in zip(sp_pt, sp_ox):
            assert (s1["start"], s1["end"], s1["surface"]) == (
                s2["start"],
                s2["end"],
                s2["surface"],
            )
            span_total += 1
            if s1["reading"] == s2["reading"]:
                span_read_match += 1
            else:
                r_ok = False
            if s1["pron"] == s2["pron"]:
                span_pron_match += 1
            else:
                p_ok = False
            if (s1["start"], s1["end"]) in amb_positions:
                amb_total += 1
                if s1["reading"] == s2["reading"] and s1["pron"] == s2["pron"]:
                    amb_match += 1

        if r_ok:
            sent_read_match += 1
        if p_ok:
            sent_pron_match += 1

        mask = np.asarray(m_pt, dtype=bool)
        diff = np.abs(lg_pt[mask] - lg_ox[mask])
        if diff.size > 0:
            cur_max = float(np.max(diff))
            if cur_max > max_abs_diff:
                max_abs_diff = cur_max
            sum_abs_diff += float(np.sum(diff))
            total_logit_elems += int(diff.size)

    # 参考ゴールド評価（eval_reading.evaluate_predictions 互換形式へ整形）
    def _to_eval_fmt(plist: List[List[dict]]) -> List[List[dict]]:
        return [[{"reading_kata": "", **s} for s in sent] for sent in plist]

    m_pt_gold = er.evaluate_predictions(rows, _to_eval_fmt(pt_preds))
    m_ox_gold = er.evaluate_predictions(rows, _to_eval_fmt(ox_preds))

    n = len(rows)
    res = {
        "split": split_label,
        "n_sentences": n,
        "total_chars": total_chars,
        "sentence_reading_match": sent_read_match,
        "sentence_reading_match_rate": round(sent_read_match / max(1, n), 6),
        "sentence_pron_match": sent_pron_match,
        "sentence_pron_match_rate": round(sent_pron_match / max(1, n), 6),
        "span_total": span_total,
        "span_reading_match": span_read_match,
        "span_reading_match_rate": round(span_read_match / max(1, span_total), 6),
        "span_pron_match": span_pron_match,
        "span_pron_match_rate": round(span_pron_match / max(1, span_total), 6),
        "ambiguous_catalog_span_total": amb_total,
        "ambiguous_catalog_span_match": amb_match,
        "ambiguous_catalog_span_match_rate": round(amb_match / max(1, amb_total), 6),
        "max_abs_logit_diff": max_abs_diff,
        "mean_abs_logit_diff": sum_abs_diff / max(1, total_logit_elems),
        "ref_gold_pytorch_fp32": {
            "sentence_reading_acc": m_pt_gold["sentence_reading_acc"],
            "sentence_pron_acc": m_pt_gold["sentence_pron_acc"],
            "span_reading_acc": m_pt_gold["span_reading_acc"],
        },
        "ref_gold_onnx_fp32": {
            "sentence_reading_acc": m_ox_gold["sentence_reading_acc"],
            "sentence_pron_acc": m_ox_gold["sentence_pron_acc"],
            "span_reading_acc": m_ox_gold["span_reading_acc"],
        },
    }
    return res, pt_preds


def summarize_latencies(lat_ms: List[float], total_chars: int) -> dict:
    """レイテンシ統計を算出する。"""
    arr = np.asarray(lat_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    total_sec = float(np.sum(arr)) / 1000.0
    n = len(lat_ms)
    return {
        "n": n,
        "total_chars": total_chars,
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "sent_per_sec": round(n / max(1e-9, total_sec), 3),
        "chars_per_sec": round(total_chars / max(1e-9, total_sec), 1),
    }


def measure_real_latency(tagger, rows: List[dict], warmup: int) -> Tuple[dict, dict]:
    """Forward 単体と E2E 全体の CPU 推論速度（batch=1）を計測する。"""
    texts = [r["text"] for r in rows]
    total_chars = sum(len(t) for t in texts)
    encoded = [tagger.tok(t, truncation=True, max_length=tagger.max_len) for t in texts]

    # 1. Forward 単体計測
    for enc in encoded[:warmup]:
        tagger._forward_logits([enc["input_ids"]], [enc["attention_mask"]])
    fwd_ms: List[float] = []
    for enc in encoded:
        t0 = time.perf_counter()
        tagger._forward_logits([enc["input_ids"]], [enc["attention_mask"]])
        fwd_ms.append((time.perf_counter() - t0) * 1000.0)

    # 2. E2E 全体計測 (Sudachi スパン抽出 + トークナイズ + Forward + デコード + 発音変換)
    for txt in texts[:warmup]:
        tagger.predict(txt)
    e2e_ms: List[float] = []
    for txt in texts:
        t0 = time.perf_counter()
        tagger.predict(txt)
        e2e_ms.append((time.perf_counter() - t0) * 1000.0)

    return summarize_latencies(fwd_ms, total_chars), summarize_latencies(e2e_ms, total_chars)


def measure_seq_sweep(tagger, seq_lens: List[int], repeats: int, warmup: int = 5) -> Dict[str, dict]:
    """固定トークン長での Forward レイテンシスイープ（batch=1）を行う。"""
    out: Dict[str, dict] = {}
    for sl in seq_lens:
        ids = [[100] * sl]
        attn = [[1] * sl]
        for _ in range(warmup):
            tagger._forward_logits(ids, attn)
        times: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            tagger._forward_logits(ids, attn)
            times.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(times, dtype=np.float64)
        out[str(sl)] = {
            "repeats": repeats,
            "mean_ms": round(float(np.mean(arr)), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="第9号 ONNX FP32 一致検証＆CPUベンチマーク")
    ap.add_argument("--staging-dir", default=str(ROOT / "release-staging"))
    ap.add_argument("--best-dir", default=str(ROOT / "runs" / "prod-modernbert" / "best"))
    ap.add_argument("--master-dir", default=str(ROOT / "data" / "master"))
    ap.add_argument("--dev-path", default=str(ROOT / "data" / "prod" / "dev.jsonl"))
    ap.add_argument("--sealed-path", default=str(ROOT / "data" / "prod" / "sealed-test.jsonl"))
    ap.add_argument("--out-parity", default=str(ROOT / "onnx" / "onnx_parity.json"))
    ap.add_argument("--out-bench", default=str(ROOT / "onnx" / "onnx_bench_cpu.json"))
    ap.add_argument("--n-parity", type=int, default=200)
    args = ap.parse_args()

    staging_dir = Path(args.staging_dir)
    staged_mod = load_staged_module(staging_dir)

    dev_200 = load_jsonl_head(Path(args.dev_path), args.n_parity)
    sealed_200 = load_jsonl_head(Path(args.sealed_path), args.n_parity)
    assert len(dev_200) == args.n_parity and len(sealed_200) == args.n_parity

    torch.set_num_threads(8)
    tagger_pt = staged_mod.JaReadingTagger(
        staging_dir, backend="pytorch", device="cpu", local_files_only=True
    )
    tagger_ox_def = staged_mod.JaReadingTagger(
        staging_dir, backend="onnx", local_files_only=True, ort_intra_threads=0, ort_inter_threads=0
    )

    with open(staging_dir / "inference_config.json", encoding="utf-8") as f:
        cat_keys = set(json.load(f)["catalog_candidates"].keys())

    # 1. PyTorch FP32 vs ONNX FP32 一致検証 (dev 200 & sealed-test 200)
    dev_parity, dev_pt_preds = eval_parity_split(
        tagger_pt, tagger_ox_def, dev_200, cat_keys, "prod_dev_head200"
    )
    sealed_parity, _ = eval_parity_split(
        tagger_pt, tagger_ox_def, sealed_200, cat_keys, "prod_sealed_test_head200"
    )

    # 旧 code/reading_tagger.py とステージング単体版の dev 200 件完全一致確認
    legacy_tagger = LegacyTagger(Path(args.best_dir), Path(args.master_dir), device="cpu")
    legacy_preds = legacy_tagger.predict_batch([r["text"] for r in dev_200], batch_size=32)
    legacy_match = sum(1 for a, b in zip(legacy_preds, dev_pt_preds) if a == b)
    del legacy_tagger

    parity_report = {
        "n_per_split": args.n_parity,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "onnxruntime_version": ort.__version__,
        "legacy_vs_standalone_dev200_match": f"{legacy_match}/{args.n_parity}",
        "legacy_vs_standalone_dev200_exact": (legacy_match == args.n_parity),
        "dev_200": dev_parity,
        "sealed_test_200": sealed_parity,
    }
    Path(args.out_parity).write_text(
        json.dumps(parity_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"PARITY dev200: sent_read={dev_parity['sentence_reading_match']}/{args.n_parity} "
        f"({dev_parity['sentence_reading_match_rate']:.4%}), max_diff={dev_parity['max_abs_logit_diff']:.3e} | "
        f"sealed200: sent_read={sealed_parity['sentence_reading_match']}/{args.n_parity} "
        f"({sealed_parity['sentence_reading_match_rate']:.4%}), max_diff={sealed_parity['max_abs_logit_diff']:.3e} | "
        f"legacy_match={legacy_match}/{args.n_parity}",
        flush=True,
    )

    # 2. デスクトップ CPU 推論速度計測 (batch=1)
    tagger_ox_8 = staged_mod.JaReadingTagger(
        staging_dir, backend="onnx", local_files_only=True, ort_intra_threads=8, ort_inter_threads=1
    )

    ox_def_fwd, ox_def_e2e = measure_real_latency(tagger_ox_def, dev_200, warmup=10)
    ox_8_fwd, ox_8_e2e = measure_real_latency(tagger_ox_8, dev_200, warmup=10)
    pt_50_fwd, pt_50_e2e = measure_real_latency(tagger_pt, dev_200[:50], warmup=5)

    seq_lens = [32, 64, 128, 256]
    sweep_ox_def = measure_seq_sweep(tagger_ox_def, seq_lens, repeats=30, warmup=5)
    sweep_ox_8 = measure_seq_sweep(tagger_ox_8, seq_lens, repeats=30, warmup=5)
    sweep_pt_8 = measure_seq_sweep(tagger_pt, seq_lens, repeats=10, warmup=3)

    bench_report = {
        "cpu_name": get_cpu_name(),
        "logical_cores": os.cpu_count(),
        "torch_version": torch.__version__,
        "onnxruntime_version": ort.__version__,
        "providers": ["CPUExecutionProvider"],
        "batch_size": 1,
        "max_length": tagger_ox_def.max_len,
        "real_data_dev": {
            "onnx_default_threads": {
                "intra": 0,
                "inter": 0,
                "forward": ox_def_fwd,
                "e2e": ox_def_e2e,
            },
            "onnx_physical_8_threads": {
                "intra": 8,
                "inter": 1,
                "forward": ox_8_fwd,
                "e2e": ox_8_e2e,
            },
            "pytorch_cpu_fp32_ref_8_threads": {
                "threads": 8,
                "forward": pt_50_fwd,
                "e2e": pt_50_e2e,
            },
        },
        "seq_len_sweep": {
            "onnx_default_threads": sweep_ox_def,
            "onnx_physical_8_threads": sweep_ox_8,
            "pytorch_cpu_fp32_ref_8_threads": sweep_pt_8,
        },
    }
    Path(args.out_bench).write_text(
        json.dumps(bench_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"BENCHMARK CPU ({bench_report['cpu_name']}): "
        f"ORT def E2E={ox_def_e2e['mean_ms']}ms ({ox_def_e2e['sent_per_sec']} sent/s), "
        f"ORT 8th E2E={ox_8_e2e['mean_ms']}ms ({ox_8_e2e['sent_per_sec']} sent/s), "
        f"PT 8th E2E={pt_50_e2e['mean_ms']}ms ({pt_50_e2e['sent_per_sec']} sent/s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
