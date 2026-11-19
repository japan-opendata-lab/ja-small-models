#!/usr/bin/env python3
"""第4号 ja-doc-classify-v1 (v2): PyTorch FP32 vs ONNX FP32 一致検証＆CPU推論速度ベンチマーク。

1. 一致検証 (onnx/onnx_parity.json):
   - OOD v1 (200件) / OOD v2 (200件) / dev 先頭200件（計600件、sealed-test 非参照）
   - PyTorch CPU FP32 (final-hf) vs ONNX Runtime CPU FP32 (onnx/model_fp32.onnx)
   - Raw argmax 一致率、確定既定閾値 (0.95 / 0.80 / 0.12) 適用後の全フィールド一致率、
     3ヘッドロジット差・p_high_risk_sum 差、OOD v1/v2 実運用指標を算出。
2. CPU 推論速度ベンチマーク (onnx/onnx_bench_cpu.json):
   - ONNX default_threads (intra=0, inter=0) および physical_8_threads (intra=8, inter=1)
     - OOD v2 200件 (ORT単体 & トークナイズ・後処理込み E2E)
     - 固定トークン長スイープ (64, 128, 256, 512 各20回 / 1024 10回)
   - 参考: PyTorch CPU FP32 (torch.set_num_threads(8))
     - OOD v2 先頭50件 (Forward単体 & E2E)
     - 固定トークン長スイープ (64, 128, 256, 512 各10回)
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
import transformers
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def softmax_f64(logits_1d: np.ndarray) -> np.ndarray:
    z = np.asarray(logits_1d, dtype=np.float64)
    m = np.max(z)
    e = np.exp(z - m)
    return e / np.sum(e)


def route_single(
    ld: np.ndarray,
    lt: np.ndarray,
    lf: np.ndarray,
    routing_cfg: dict,
) -> dict:
    """単一文書の3ヘッドロジットから float64 softmax + 確定既定閾値でルーティング辞書を算出する。"""
    th = routing_cfg["routing_thresholds"]
    d_th = float(th["domain_reject_threshold"])
    t_th = float(th["type_reject_threshold"])
    hr_th = float(th["high_risk_prob_sum_threshold"])

    pd = softmax_f64(ld)
    pt = softmax_f64(lt)
    pf = softmax_f64(lf)

    d_idx = int(np.argmax(pd))
    t_idx = int(np.argmax(pt))
    f_idx = int(np.argmax(pf))

    d_conf = float(pd[d_idx])
    t_conf = float(pt[t_idx])
    f_conf = float(pf[f_idx])
    hr_idx = routing_cfg["high_risk_type_indices"]
    hr_sum = float(pt[hr_idx].sum())

    d_top1 = routing_cfg["domain_labels"][d_idx]
    t_top1 = routing_cfg["type_labels"][t_idx]
    f_top1 = routing_cfg["format_labels"][f_idx]

    d_rej = bool(d_conf < d_th)
    t_rej = bool(t_conf < t_th)
    hr_flag = bool(hr_sum >= hr_th)

    return {
        "domain": None if d_rej else d_top1,
        "domain_rejected": d_rej,
        "domain_top1": d_top1,
        "domain_confidence": d_conf,
        "document_type": None if t_rej else t_top1,
        "type_rejected": t_rej,
        "document_type_top1": t_top1,
        "type_confidence": t_conf,
        "high_risk": hr_flag,
        "high_risk_prob_sum": hr_sum,
        "document_format": f_top1,
        "format_confidence": f_conf,
    }


def make_ort_session(onnx_path: str, intra_threads: int, inter_threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_threads
    so.inter_op_num_threads = inter_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])


def summarize_routing_metrics(rows: list[dict], preds: list[dict], routing_cfg: dict) -> dict:
    """確定閾値適用時の領域棄却後・種別棄却後・高機密フラグの実測指標を計算する。"""
    hr_set = set(routing_cfg["high_risk_types"])
    n = len(rows)

    # 領域（棄却閾値 0.95）
    kept_dom_idx = [i for i, p in enumerate(preds) if not p["domain_rejected"]]
    dom_rej_cnt = n - len(kept_dom_idx)
    if kept_dom_idx:
        yd = [rows[i]["domain"] for i in kept_dom_idx]
        pd = [preds[i]["domain"] for i in kept_dom_idx]
        dom_labs = sorted(set(yd))
        dom_acc = round(float(accuracy_score(yd, pd)), 4)
        dom_f1 = round(float(f1_score(yd, pd, labels=dom_labs, average="macro", zero_division=0)), 4)
    else:
        dom_acc = dom_f1 = None

    # 種別（棄却閾値 0.80）
    all_cov_types = sorted({r["document_type"] for r in rows})
    kept_type_idx = [i for i, p in enumerate(preds) if not p["type_rejected"]]
    type_rej_cnt = n - len(kept_type_idx)
    if kept_type_idx:
        yt = [rows[i]["document_type"] for i in kept_type_idx]
        pt = [preds[i]["document_type"] for i in kept_type_idx]
        kept_true_types = sorted(set(yt))
        type_acc = round(float(accuracy_score(yt, pt)), 4)
        type_f1_kept = round(float(f1_score(yt, pt, labels=kept_true_types, average="macro", zero_division=0)), 4)
        type_f1_all = round(float(f1_score(yt, pt, labels=all_cov_types, average="macro", zero_division=0)), 4)
        kept_types_cnt = len(kept_true_types)
    else:
        type_acc = type_f1_kept = type_f1_all = None
        kept_types_cnt = 0

    # 高機密フラグ（p_high_risk_sum >= 0.12）
    hr_gold = np.array([1 if r["document_type"] in hr_set else 0 for r in rows])
    hr_pred = np.array([1 if p["high_risk"] else 0 for p in preds])
    tp = int(((hr_gold == 1) & (hr_pred == 1)).sum())
    fn = int(((hr_gold == 1) & (hr_pred == 0)).sum())
    fp = int(((hr_gold == 0) & (hr_pred == 1)).sum())
    tn = int(((hr_gold == 0) & (hr_pred == 0)).sum())
    hr_total = tp + fn
    std_total = fp + tn

    return {
        "rows": n,
        "domain_routing": {
            "reject_threshold": float(routing_cfg["routing_thresholds"]["domain_reject_threshold"]),
            "kept_rows": len(kept_dom_idx),
            "rejected_rows": dom_rej_cnt,
            "reject_rate": round(dom_rej_cnt / n, 4),
            "domain_accuracy": dom_acc,
            "domain_macro_f1": dom_f1,
        },
        "type_suggest": {
            "reject_threshold": float(routing_cfg["routing_thresholds"]["type_reject_threshold"]),
            "kept_rows": len(kept_type_idx),
            "rejected_rows": type_rej_cnt,
            "reject_rate": round(type_rej_cnt / n, 4),
            "type_accuracy": type_acc,
            "type_macro_f1_kept_true": type_f1_kept,
            "kept_true_types": kept_types_cnt,
            "total_covered_types": len(all_cov_types),
            "type_macro_f1_all_covered": type_f1_all,
        },
        "high_risk_flag": {
            "threshold": float(routing_cfg["routing_thresholds"]["high_risk_prob_sum_threshold"]),
            "hr_total": hr_total,
            "std_total": std_total,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "recall": round(tp / hr_total, 4) if hr_total else None,
            "std_fpr": round(fp / std_total, 4) if std_total else None,
            "precision": round(float(precision_score(hr_gold, hr_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(hr_gold, hr_pred, zero_division=0)), 4),
            "flag_rate": round((tp + fp) / n, 4),
        },
    }


def eval_parity_dataset(
    name: str,
    rows: list[dict],
    pt_model,
    ort_sess: ort.InferenceSession,
    tok,
    routing_cfg: dict,
) -> dict:
    """PyTorch CPU FP32 vs ONNX Runtime CPU FP32 の完全一致検証を 1 データセットで実行する。"""
    max_len = int(routing_cfg["max_length"])
    in_names = [i.name for i in ort_sess.get_inputs()]

    diffs_d, diffs_t, diffs_f, diffs_hr = [], [], [], []
    pt_routes, ox_routes = [], []

    raw_domain_match = 0
    raw_type_match = 0
    raw_format_match = 0
    final_domain_match = 0
    final_type_match = 0
    final_hr_match = 0
    final_format_match = 0
    all_fields_match = 0
    mismatches = []

    for r in rows:
        enc_pt = tok(r["text"], truncation=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            out_pt = pt_model(input_ids=enc_pt["input_ids"], attention_mask=enc_pt["attention_mask"])
        pt_ld = out_pt.logits_domain[0].numpy().astype(np.float32)
        pt_lt = out_pt.logits_type[0].numpy().astype(np.float32)
        pt_lf = out_pt.logits_format[0].numpy().astype(np.float32)

        feed = {
            k: enc_pt[k].numpy().astype(np.int64)
            for k in in_names
        }
        ox_ld, ox_lt, ox_lf = ort_sess.run(None, feed)
        ox_ld = ox_ld[0].astype(np.float32)
        ox_lt = ox_lt[0].astype(np.float32)
        ox_lf = ox_lf[0].astype(np.float32)

        diffs_d.append(np.abs(pt_ld - ox_ld))
        diffs_t.append(np.abs(pt_lt - ox_lt))
        diffs_f.append(np.abs(pt_lf - ox_lf))

        rp = route_single(pt_ld, pt_lt, pt_lf, routing_cfg)
        rx = route_single(ox_ld, ox_lt, ox_lf, routing_cfg)
        pt_routes.append(rp)
        ox_routes.append(rx)

        diffs_hr.append(abs(rp["high_risk_prob_sum"] - rx["high_risk_prob_sum"]))

        if rp["domain_top1"] == rx["domain_top1"]:
            raw_domain_match += 1
        if rp["document_type_top1"] == rx["document_type_top1"]:
            raw_type_match += 1
        if rp["document_format"] == rx["document_format"]:
            raw_format_match += 1

        d_ok = (rp["domain"] == rx["domain"]) and (rp["domain_rejected"] == rx["domain_rejected"])
        t_ok = (rp["document_type"] == rx["document_type"]) and (rp["type_rejected"] == rx["type_rejected"])
        h_ok = rp["high_risk"] == rx["high_risk"]
        f_ok = rp["document_format"] == rx["document_format"]

        if d_ok:
            final_domain_match += 1
        if t_ok:
            final_type_match += 1
        if h_ok:
            final_hr_match += 1
        if f_ok:
            final_format_match += 1
        if d_ok and t_ok and h_ok and f_ok:
            all_fields_match += 1
        else:
            mismatches.append(
                {
                    "id": r.get("id"),
                    "pt": {k: rp[k] for k in ("domain", "document_type", "high_risk", "document_format")},
                    "ox": {k: rx[k] for k in ("domain", "document_type", "high_risk", "document_format")},
                }
            )

    n = len(rows)
    ad = np.concatenate([x.ravel() for x in diffs_d])
    at = np.concatenate([x.ravel() for x in diffs_t])
    af = np.concatenate([x.ravel() for x in diffs_f])
    ahr = np.array(diffs_hr, dtype=np.float64)

    return {
        "dataset": name,
        "rows": n,
        "raw_argmax_match": {
            "domain": f"{raw_domain_match}/{n}",
            "domain_rate": round(raw_domain_match / n, 4),
            "type": f"{raw_type_match}/{n}",
            "type_rate": round(raw_type_match / n, 4),
            "format": f"{raw_format_match}/{n}",
            "format_rate": round(raw_format_match / n, 4),
        },
        "routed_output_match": {
            "all_fields_exact": all_fields_match,
            "all_fields_exact_rate": round(all_fields_match / n, 4),
            "domain": final_domain_match,
            "type": final_type_match,
            "high_risk": final_hr_match,
            "format": final_format_match,
            "mismatches": mismatches,
        },
        "logit_abs_diff": {
            "domain": {"max": float(ad.max()), "mean": float(ad.mean())},
            "type": {"max": float(at.max()), "mean": float(at.mean())},
            "format": {"max": float(af.max()), "mean": float(af.mean())},
        },
        "high_risk_prob_sum_abs_diff": {
            "max": float(ahr.max()),
            "mean": float(ahr.mean()),
        },
        "routed_metrics_pytorch": summarize_routing_metrics(rows, pt_routes, routing_cfg),
        "routed_metrics_onnx": summarize_routing_metrics(rows, ox_routes, routing_cfg),
    }


def stats_from_ms(lat_ms: list[float], total_sec: float, total_chars: int = 0) -> dict:
    arr = np.array(lat_ms, dtype=np.float64)
    n = len(arr)
    res = {
        "n": n,
        "total_sec": round(total_sec, 4),
        "docs_per_sec": round(n / total_sec, 3) if total_sec > 0 else None,
        "mean_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
    }
    if total_chars > 0:
        res["total_chars"] = int(total_chars)
        res["chars_per_sec"] = round(total_chars / total_sec, 1) if total_sec > 0 else None
    return res


def bench_ort_suite(
    onnx_path: str,
    intra_threads: int,
    inter_threads: int,
    ood_v2_texts: list[str],
    tok,
    routing_cfg: dict,
) -> dict:
    """指定スレッド設定で ORT CPU 速度（OOD v2 200件 単体/E2E ＆ 固定長スイープ）を計測する。"""
    sess = make_ort_session(onnx_path, intra_threads=intra_threads, inter_threads=inter_threads)
    in_names = [i.name for i in sess.get_inputs()]
    max_len = int(routing_cfg["max_length"])

    # 事前トークナイズ
    feeds = []
    for txt in ood_v2_texts:
        enc = tok(txt, truncation=True, max_length=max_len, return_tensors="np")
        feeds.append({k: np.asarray(enc[k], dtype=np.int64) for k in in_names})

    # ウォームアップ 5 回
    for f in feeds[:5]:
        sess.run(None, f)

    # (1) OOD v2 200件 ORT 単体
    lat_ort = []
    t0 = time.perf_counter()
    for f in feeds:
        ts = time.perf_counter()
        sess.run(None, f)
        lat_ort.append((time.perf_counter() - ts) * 1000.0)
    tot_ort = time.perf_counter() - t0
    total_chars = sum(len(t) for t in ood_v2_texts)

    # (2) OOD v2 200件 E2E（トークナイズ＋ORT推論＋閾値後処理）
    lat_e2e = []
    t0 = time.perf_counter()
    for txt in ood_v2_texts:
        ts = time.perf_counter()
        enc = tok(txt, truncation=True, max_length=max_len, return_tensors="np")
        feed = {k: np.asarray(enc[k], dtype=np.int64) for k in in_names}
        ld, lt, lf = sess.run(None, feed)
        _ = route_single(ld[0], lt[0], lf[0], routing_cfg)
        lat_e2e.append((time.perf_counter() - ts) * 1000.0)
    tot_e2e = time.perf_counter() - t0

    # (3) 固定トークン長スイープ
    sweep = {}
    for seq_len, n_iter, warmup in [(64, 20, 5), (128, 20, 5), (256, 20, 5), (512, 20, 5), (1024, 10, 3)]:
        ids = np.full((1, seq_len), 100, dtype=np.int64)
        ids[0, 0] = 6
        ids[0, -1] = 4
        mask = np.ones((1, seq_len), dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        for _ in range(warmup):
            sess.run(None, feed)
        lat = []
        ts0 = time.perf_counter()
        for _ in range(n_iter):
            ts = time.perf_counter()
            sess.run(None, feed)
            lat.append((time.perf_counter() - ts) * 1000.0)
        tot = time.perf_counter() - ts0
        sweep[str(seq_len)] = stats_from_ms(lat, tot)

    del sess
    return {
        "intra_op_num_threads": intra_threads,
        "inter_op_num_threads": inter_threads,
        "ood_v2_200": {
            "ort_only": stats_from_ms(lat_ort, tot_ort, total_chars=total_chars),
            "e2e": stats_from_ms(lat_e2e, tot_e2e, total_chars=total_chars),
        },
        "fixed_seq_len_sweep": sweep,
    }


def bench_pytorch_cpu(
    pt_model,
    ood_v2_texts_50: list[str],
    tok,
    routing_cfg: dict,
    num_threads: int = 8,
) -> dict:
    """参考計測: PyTorch CPU FP32 (OOD v2 先頭50件 ＆ 固定トークン長スイープ)。"""
    torch.set_num_threads(num_threads)
    max_len = int(routing_cfg["max_length"])

    pt_inputs = [
        tok(txt, truncation=True, max_length=max_len, return_tensors="pt")
        for txt in ood_v2_texts_50
    ]
    for enc in pt_inputs[:3]:
        with torch.no_grad():
            pt_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    lat_fwd = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for enc in pt_inputs:
            ts = time.perf_counter()
            pt_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            lat_fwd.append((time.perf_counter() - ts) * 1000.0)
    tot_fwd = time.perf_counter() - t0
    chars_50 = sum(len(t) for t in ood_v2_texts_50)

    lat_e2e = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for txt in ood_v2_texts_50:
            ts = time.perf_counter()
            enc = tok(txt, truncation=True, max_length=max_len, return_tensors="pt")
            out = pt_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            _ = route_single(
                out.logits_domain[0].numpy(),
                out.logits_type[0].numpy(),
                out.logits_format[0].numpy(),
                routing_cfg,
            )
            lat_e2e.append((time.perf_counter() - ts) * 1000.0)
    tot_e2e = time.perf_counter() - t0

    sweep = {}
    for seq_len in (64, 128, 256, 512):
        ids = torch.full((1, seq_len), 100, dtype=torch.long)
        ids[0, 0] = 6
        ids[0, -1] = 4
        mask = torch.ones((1, seq_len), dtype=torch.long)
        with torch.no_grad():
            for _ in range(3):
                pt_model(input_ids=ids, attention_mask=mask)
            lat = []
            ts0 = time.perf_counter()
            for _ in range(10):
                ts = time.perf_counter()
                pt_model(input_ids=ids, attention_mask=mask)
                lat.append((time.perf_counter() - ts) * 1000.0)
            tot = time.perf_counter() - ts0
        sweep[str(seq_len)] = stats_from_ms(lat, tot)

    return {
        "torch_num_threads": num_threads,
        "ood_v2_50": {
            "forward_only": stats_from_ms(lat_fwd, tot_fwd, total_chars=chars_50),
            "e2e": stats_from_ms(lat_e2e, tot_e2e, total_chars=chars_50),
        },
        "fixed_seq_len_sweep": sweep,
    }


def main() -> int:
    root = Path(".")  # project root (run from the model folder)
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default=str(root / "runs" / "prod-modernbert-v2" / "final-hf"))
    ap.add_argument("--onnx-fp32", default=str(root / "onnx" / "model_fp32.onnx"))
    ap.add_argument("--ood-v1", default=str(root / "data" / "ood" / "ood-200.jsonl"))
    ap.add_argument("--ood-v2", default=str(root / "data" / "ood_v2" / "ood-v2-200.jsonl"))
    ap.add_argument("--dev-jsonl", default=str(root / "data" / "pilot" / "dev.jsonl"))
    ap.add_argument("--out-parity", default=str(root / "onnx" / "onnx_parity.json"))
    ap.add_argument("--out-bench", default=str(root / "onnx" / "onnx_bench_cpu.json"))
    args = ap.parse_args()

    t_start = time.perf_counter()
    hf_dir = Path(args.hf_dir)
    routing_cfg = json.loads((hf_dir / "routing_config.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(hf_dir), local_files_only=True)
    pt_model = AutoModel.from_pretrained(
        str(hf_dir), trust_remote_code=True, local_files_only=True, dtype=torch.float32
    ).cpu().eval()

    # 1. 一致検証 (OOD v1 200, OOD v2 200, dev 先頭200)
    rows_v1 = load_jsonl(Path(args.ood_v1))
    rows_v2 = load_jsonl(Path(args.ood_v2))
    rows_dev200 = load_jsonl(Path(args.dev_jsonl), limit=200)

    torch.set_num_threads(8)
    sess_parity = make_ort_session(args.onnx_fp32, intra_threads=8, inter_threads=1)
    parity_v1 = eval_parity_dataset("ood_v1_200", rows_v1, pt_model, sess_parity, tok, routing_cfg)
    parity_v2 = eval_parity_dataset("ood_v2_200", rows_v2, pt_model, sess_parity, tok, routing_cfg)
    parity_dev = eval_parity_dataset("dev_first200", rows_dev200, pt_model, sess_parity, tok, routing_cfg)
    del sess_parity

    parity_report = {
        "hf_dir": str(hf_dir),
        "onnx_path": str(args.onnx_fp32),
        "routing_thresholds": routing_cfg["routing_thresholds"],
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "onnxruntime_version": ort.__version__,
        "sealed_test_opened": False,
        "datasets": {
            "ood_v1": parity_v1,
            "ood_v2": parity_v2,
            "dev_first200": parity_dev,
        },
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
    }
    Path(args.out_parity).write_text(
        json.dumps(parity_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Saved parity verification to {args.out_parity}")

    # 2. CPU 推論速度ベンチマーク
    t_b0 = time.perf_counter()
    ood_v2_texts = [r["text"] for r in rows_v2]
    char_lens = [len(t) for t in ood_v2_texts]
    tok_lens = [len(tok(t, truncation=True, max_length=routing_cfg["max_length"])["input_ids"]) for t in ood_v2_texts]

    ort_def = bench_ort_suite(args.onnx_fp32, 0, 0, ood_v2_texts, tok, routing_cfg)
    ort_p8 = bench_ort_suite(args.onnx_fp32, 8, 1, ood_v2_texts, tok, routing_cfg)
    pt_cpu = bench_pytorch_cpu(pt_model, ood_v2_texts[:50], tok, routing_cfg, num_threads=8)

    bench_report = {
        "cpu": platform.processor(),
        "logical_cores": os.cpu_count(),
        "onnxruntime_version": ort.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "ood_v2_200_len_stats": {
            "char_len": {
                "min": int(min(char_lens)),
                "median": float(np.median(char_lens)),
                "mean": round(float(np.mean(char_lens)), 1),
                "p95": float(np.percentile(char_lens, 95)),
                "max": int(max(char_lens)),
                "total": int(sum(char_lens)),
            },
            "token_len": {
                "min": int(min(tok_lens)),
                "median": float(np.median(tok_lens)),
                "mean": round(float(np.mean(tok_lens)), 1),
                "p95": float(np.percentile(tok_lens, 95)),
                "max": int(max(tok_lens)),
                "total": int(sum(tok_lens)),
            },
        },
        "ort_default_threads": ort_def,
        "ort_physical_8_threads": ort_p8,
        "pytorch_cpu_fp32": pt_cpu,
        "bench_elapsed_sec": round(time.perf_counter() - t_b0, 2),
        "total_elapsed_sec": round(time.perf_counter() - t_start, 2),
    }
    Path(args.out_bench).write_text(
        json.dumps(bench_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Saved CPU benchmark to {args.out_bench}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
