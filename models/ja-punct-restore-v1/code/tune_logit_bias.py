#!/usr/bin/env python3
"""第3号 ja-punct-restore-v1: dev.jsonl でのクラス別ロジットバイアス調整。

runs/prod-modernbert/final を用い、train_prod_tok.py の predict_chars と
完全同一の重なり分割仕様（max_len=640, overlap=128, 最初の予測を優先）で
dev.jsonl 各文字位置の 5 クラス生ロジットを抽出し、b_0=0 固定で
micro-F1（O 除く）が最大となるバイアス [0.0, b_1, b_2, b_3, b_4] を探索する。
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from transformers import AutoModelForTokenClassification, AutoTokenizer

LABELS = ["O", "COMMA", "PERIOD", "QUESTION", "EXCLAIM"]


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def extract_dev_logits(
    model, tok, dev_rows: list[dict], device: str, max_len: int = 640, overlap: int = 128, batch_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """predict_chars と同一優先規則で dev 全文字の生ロジット・正解ラベルを抽出する。

    戻り値:
      covered_logits: (N_cov, 5) float32 tensor (device 上)
      covered_gold:   (N_cov,) int64 tensor (device 上)
      uncovered_gold_counts: (5,) int64 ndarray（モデル予測が常に O=0 に固定される未カバー文字の gold 内訳）
    """
    step = max(1, max_len - overlap)
    chunk_meta = []  # (win_idx, start_char, chunk_text)
    win_preds = []
    win_golds = []

    for wi, r in enumerate(dev_rows):
        text = r["text"]
        n = len(text)
        win_preds.append(np.full((n, 5), np.nan, dtype=np.float32))
        win_golds.append(np.array(r["labels"], dtype=np.int64))
        start = 0
        while start < n:
            chunk = text[start : start + max_len]
            chunk_meta.append((wi, start, chunk))
            if start + max_len >= n:
                break
            start += step

    total_chunks = len(chunk_meta)
    for b0 in range(0, total_chunks, batch_size):
        batch = chunk_meta[b0 : b0 + batch_size]
        enc = tok(
            [c[2] for c in batch],
            return_offsets_mapping=True,
            padding=True,
        )
        ids = torch.tensor(enc["input_ids"], device=device)
        attn = torch.tensor(enc["attention_mask"], device=device)
        kw = {"input_ids": ids, "attention_mask": attn}
        if "token_type_ids" in enc:
            kw["token_type_ids"] = torch.tensor(enc["token_type_ids"], device=device)
        logits_np = model(**kw).logits.float().cpu().numpy()

        for bi, (wi, start, _) in enumerate(batch):
            arr = win_preds[wi]
            offsets = enc["offset_mapping"][bi]
            lg = logits_np[bi]
            for ti, (s, e) in enumerate(offsets):
                if e > s:
                    pos = start + e - 1
                    if np.isnan(arr[pos, 0]):
                        arr[pos] = lg[ti]

    cov_logits_list = []
    cov_gold_list = []
    uncov_counts = np.zeros(5, dtype=np.int64)
    for arr, g in zip(win_preds, win_golds):
        cov_mask = ~np.isnan(arr[:, 0])
        if cov_mask.any():
            cov_logits_list.append(arr[cov_mask])
            cov_gold_list.append(g[cov_mask])
        if (~cov_mask).any():
            uncov_counts += np.bincount(g[~cov_mask], minlength=5)

    covered_logits = torch.from_numpy(np.vstack(cov_logits_list)).to(device)
    covered_gold = torch.from_numpy(np.concatenate(cov_gold_list)).to(device)
    return covered_logits, covered_gold, uncov_counts


def cm_from_bias(
    covered_logits: torch.Tensor,
    covered_gold: torch.Tensor,
    uncov_counts: np.ndarray,
    bias_vec: list[float] | np.ndarray,
) -> np.ndarray:
    """指定バイアスに対する 5x5 混同行列 cm[pred, gold] を返す。"""
    b_t = torch.tensor(bias_vec, dtype=covered_logits.dtype, device=covered_logits.device)
    pred = (covered_logits + b_t).argmax(dim=-1)
    idx = pred * 5 + covered_gold
    cm = torch.bincount(idx, minlength=25).reshape(5, 5).cpu().numpy()
    cm[0, :] += uncov_counts
    return cm


def metrics_from_cm(cm: np.ndarray) -> dict:
    """5x5 混同行列 cm[pred, gold] から micro-F1(O除く), MCC, 記号別 P/R/F1 を算出する。"""
    tp_all = np.diag(cm)
    pred_sum = cm.sum(axis=1).astype(np.float64)
    gold_sum = cm.sum(axis=0).astype(np.float64)
    total = float(cm.sum())

    per = {}
    tp_no_o = fp_no_o = fn_no_o = 0
    for c in range(1, 5):
        tp = int(tp_all[c])
        fp = int(pred_sum[c] - tp)
        fn = int(gold_sum[c] - tp)
        tp_no_o += tp
        fp_no_o += fp
        fn_no_o += fn
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per[LABELS[c]] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": int(tp + fn),
        }

    mp = tp_no_o / (tp_no_o + fp_no_o) if (tp_no_o + fp_no_o) else 0.0
    mr = tp_no_o / (tp_no_o + fn_no_o) if (tp_no_o + fn_no_o) else 0.0
    mf = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0

    # 多クラス MCC (Gorodkin)
    c_correct = float(tp_all.sum())
    cov_xy = c_correct * total - float(np.dot(pred_sum, gold_sum))
    cov_xx = total * total - float(np.dot(pred_sum, pred_sum))
    cov_yy = total * total - float(np.dot(gold_sum, gold_sum))
    denom = math.sqrt(cov_xx * cov_yy)
    mcc = cov_xy / denom if denom > 0 else 0.0

    return {
        "micro_no_O": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "f1": round(mf, 4),
        },
        "micro_f1_exact": mf,
        "mcc_exact": mcc,
        "mcc_all": round(mcc, 4),
        "per_symbol": per,
    }


def optimize_coordinate_ascent(
    covered_logits: torch.Tensor,
    covered_gold: torch.Tensor,
    uncov_counts: np.ndarray,
    init_b: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """1 初期点から粗・中・精の3段階で座標上昇法を実行する。"""
    b = init_b.copy()
    best_m = metrics_from_cm(cm_from_bias(covered_logits, covered_gold, uncov_counts, b))
    best_f1 = best_m["micro_f1_exact"]
    best_mcc = best_m["mcc_exact"]

    for step, span in [(0.10, 15), (0.02, 10), (0.005, 8)]:
        improved = True
        while improved:
            improved = False
            for c in (1, 2, 3, 4):
                orig = b[c]
                cand_vals = [orig + k * step for k in range(-span, span + 1) if k != 0]
                best_val = orig
                for val in cand_vals:
                    b[c] = val
                    m = metrics_from_cm(
                        cm_from_bias(covered_logits, covered_gold, uncov_counts, b)
                    )
                    f1 = m["micro_f1_exact"]
                    mcc = m["mcc_exact"]
                    if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and mcc > best_mcc + 1e-12):
                        best_f1 = f1
                        best_mcc = mcc
                        best_val = val
                        improved = True
                b[c] = best_val
    return b, best_f1, best_mcc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/prod-modernbert/final")
    ap.add_argument("--dev", default="data/prod/dev.jsonl")
    ap.add_argument("--eval-ref", default="runs/prod-modernbert/eval_prod.json")
    ap.add_argument("--out", default="runs/prod-modernbert")
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()
    dev_rows = load_rows(Path(args.dev))
    eval_ref = json.loads(Path(args.eval_ref).read_text(encoding="utf-8"))

    print(f"Extracting dev logits for {len(dev_rows)} windows on {device}...", flush=True)
    cov_logits, cov_gold, uncov_counts = extract_dev_logits(
        model, tok, dev_rows, device, args.max_len, args.overlap, args.batch_size
    )
    total_chars = int(cov_gold.numel() + uncov_counts.sum())
    print(
        f"Logits extracted in {time.time()-t0:.1f}s: covered={cov_gold.numel()}, "
        f"uncovered={int(uncov_counts.sum())}, total={total_chars}",
        flush=True,
    )

    # 1. bias=0 で eval_prod.json と整合するか検証（fp32ロードと学習直後fp16 autocastの微小丸め差を許容）
    cm_zero = cm_from_bias(cov_logits, cov_gold, uncov_counts, [0.0, 0.0, 0.0, 0.0, 0.0])
    m_zero = metrics_from_cm(cm_zero)
    assert total_chars == eval_ref["dev_chars"], (total_chars, eval_ref["dev_chars"])
    for k in ("precision", "recall", "f1"):
        assert abs(m_zero["micro_no_O"][k] - eval_ref["micro_no_O"][k]) <= 0.001, (
            k, m_zero["micro_no_O"][k], eval_ref["micro_no_O"][k]
        )
    assert abs(m_zero["mcc_all"] - eval_ref["mcc_all"]) <= 0.001, (
        m_zero["mcc_all"], eval_ref["mcc_all"]
    )
    for sym in LABELS[1:]:
        assert m_zero["per_symbol"][sym]["support"] == eval_ref["per_symbol"][sym]["support"]
        for k in ("precision", "recall", "f1"):
            assert abs(m_zero["per_symbol"][sym][k] - eval_ref["per_symbol"][sym][k]) <= 0.005, (
                sym, k, m_zero["per_symbol"][sym][k], eval_ref["per_symbol"][sym][k]
            )
    print(f"VERIFIED baseline fp32 reproduction: micro_no_O={m_zero['micro_no_O']}, mcc={m_zero['mcc_all']} "
          f"(eval_prod.json={eval_ref['micro_no_O']}, mcc={eval_ref['mcc_all']})", flush=True)

    # 2. 多初期点からの座標上昇法
    cw = eval_ref["class_weights"]
    theo_b = np.array([0.0] + [-math.log(cw[c] / cw[0]) for c in range(1, 5)], dtype=np.float64)

    # 一様シフトの最良初期点
    best_uni = (0.0, m_zero["micro_f1_exact"])
    for delta in np.linspace(-3.5, 1.0, 91):
        cand = [0.0, float(delta), float(delta), float(delta), float(delta)]
        f1 = metrics_from_cm(cm_from_bias(cov_logits, cov_gold, uncov_counts, cand))["micro_f1_exact"]
        if f1 > best_uni[1]:
            best_uni = (float(delta), f1)

    inits = [
        ("zero", np.array([0.0, 0.0, 0.0, 0.0, 0.0])),
        ("theoretical_inv_log_cw", theo_b),
        ("uniform_best", np.array([0.0] + [best_uni[0]] * 4)),
    ]
    rng = random.Random(42)
    for k in range(10):
        inits.append((f"random_{k}", np.array([0.0] + [rng.uniform(-3.0, 0.5) for _ in range(4)])))

    t_opt = time.time()
    trials = []
    best_b = None
    best_f1 = -1.0
    best_mcc = -1.0
    for name, init_b in inits:
        opt_b, f1, mcc = optimize_coordinate_ascent(cov_logits, cov_gold, uncov_counts, init_b)
        trials.append({
            "init_name": name,
            "init_bias": [round(float(x), 4) for x in init_b],
            "opt_bias": [round(float(x), 4) for x in opt_b],
            "micro_f1": round(f1, 6),
            "mcc": round(mcc, 6),
        })
        print(f"  [{name}] f1={f1:.6f} mcc={mcc:.6f} b={[round(float(x), 4) for x in opt_b]}", flush=True)
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and mcc > best_mcc + 1e-12):
            best_f1 = f1
            best_mcc = mcc
            best_b = opt_b.copy()

    # 小数点第 3 位丸め＋0.001 刻み微調整
    rounded_b = np.array([0.0] + [round(float(best_b[c]), 3) for c in range(1, 5)])
    best_round_m = metrics_from_cm(cm_from_bias(cov_logits, cov_gold, uncov_counts, rounded_b))
    best_round_f1 = best_round_m["micro_f1_exact"]
    best_round_mcc = best_round_m["mcc_exact"]
    improved = True
    while improved:
        improved = False
        for c in (1, 2, 3, 4):
            orig = rounded_b[c]
            best_val = orig
            for delta in (-0.005, -0.004, -0.003, -0.002, -0.001, 0.001, 0.002, 0.003, 0.004, 0.005):
                rounded_b[c] = round(float(orig + delta), 3)
                m = metrics_from_cm(cm_from_bias(cov_logits, cov_gold, uncov_counts, rounded_b))
                if m["micro_f1_exact"] > best_round_f1 + 1e-12 or (
                    abs(m["micro_f1_exact"] - best_round_f1) <= 1e-12 and m["mcc_exact"] > best_round_mcc + 1e-12
                ):
                    best_round_f1 = m["micro_f1_exact"]
                    best_round_mcc = m["mcc_exact"]
                    best_val = rounded_b[c]
                    improved = True
            rounded_b[c] = best_val

    final_bias = [0.0] + [round(float(rounded_b[c]), 3) for c in range(1, 5)]
    cm_final = cm_from_bias(cov_logits, cov_gold, uncov_counts, final_bias)
    m_final = metrics_from_cm(cm_final)

    # sklearn matthews_corrcoef で検算（混同行列の展開値）
    b_t = torch.tensor(final_bias, dtype=cov_logits.dtype, device=cov_logits.device)
    cov_pred = (cov_logits + b_t).argmax(dim=-1).cpu().numpy()
    cov_gold_np = cov_gold.cpu().numpy()
    # 未カバー文字も含めて展開検算
    uncov_g = np.concatenate([np.full(int(uncov_counts[c]), c, dtype=np.int64) for c in range(5)])
    uncov_p = np.zeros_like(uncov_g)
    sk_mcc = float(matthews_corrcoef(
        np.concatenate([cov_gold_np, uncov_g]),
        np.concatenate([cov_pred, uncov_p]),
    ))
    assert round(sk_mcc, 4) == m_final["mcc_all"], (sk_mcc, m_final["mcc_all"])

    out_dir = Path(args.out)
    logit_bias_cfg = {
        "base": eval_ref["base"],
        "labels": LABELS,
        "logit_bias": final_bias,
        "tuned_on": "data/prod/dev.jsonl",
        "objective": "micro_no_O.f1",
    }
    (out_dir / "logit_bias.json").write_text(
        json.dumps(logit_bias_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "base": eval_ref["base"],
        "dev_windows": len(dev_rows),
        "dev_chars": total_chars,
        "covered_chars": int(cov_gold.numel()),
        "uncovered_chars": int(uncov_counts.sum()),
        "uncovered_gold_counts": uncov_counts.tolist(),
        "elapsed_sec": round(time.time() - t0, 2),
        "optimization_sec": round(time.time() - t_opt, 2),
        "baseline_bias_zero": {
            "logit_bias": [0.0, 0.0, 0.0, 0.0, 0.0],
            "micro_no_O": m_zero["micro_no_O"],
            "micro_f1_exact": round(m_zero["micro_f1_exact"], 6),
            "mcc_all": m_zero["mcc_all"],
            "per_symbol": m_zero["per_symbol"],
        },
        "tuned": {
            "logit_bias": final_bias,
            "micro_no_O": m_final["micro_no_O"],
            "micro_f1_exact": round(m_final["micro_f1_exact"], 6),
            "mcc_all": m_final["mcc_all"],
            "mcc_exact_sklearn": round(sk_mcc, 6),
            "per_symbol": m_final["per_symbol"],
            "confusion_matrix_pred_by_gold": cm_final.tolist(),
        },
        "uniform_best_scan": {"delta": round(best_uni[0], 3), "micro_f1_exact": round(best_uni[1], 6)},
        "trials": trials,
    }
    (out_dir / "bias_tuning.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("FINAL TUNED:", json.dumps(report["tuned"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
