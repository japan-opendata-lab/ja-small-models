#!/usr/bin/env python3
"""第3号 ja-punct-restore-v1: 封印テスト＆実 ASR ベンチマークの最終一括評価。

採用モデル (runs/prod-modernbert/final) と dev で調整したクラス別ロジットバイアス
(runs/prod-modernbert/logit_bias.json) を用い、単一 float32 推論・文字位置復元仕様
(max_len=640, overlap=128, 最初の予測優先) で以下を実施する。
1. data/prod/sealed-test.jsonl (5,003 ウィンドウ) の開封・最終評価
2. data/asr-bench/fleurs-ja-test.jsonl (650 発話) の句読点復元・strict_e2e 採点
"""
import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_asr_bench import LABELS, score_asr_punctuation  # noqa: E402

ID2PUNCT = {1: "、", 2: "。", 3: "？", 4: "！"}


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


def load_jsonl(path: Path) -> list[dict]:
    """JSONL を全行読み込む。"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def extract_char_logits(
    model,
    tok,
    texts: list[str],
    device: str,
    max_len: int = 640,
    overlap: int = 128,
    batch_size: int = 16,
    label: str = "split",
) -> list[np.ndarray]:
    """train_prod_tok.predict_chars と同一優先規則で各テキスト文字位置の生ロジットを抽出する。"""
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

    # start 昇順を第一キーに保つことで、先頭チャンク優先規則を崩さずパディングを最小化する
    chunk_meta.sort(key=lambda x: (x[1], len(x[2]), x[0]))

    total_chunks = len(chunk_meta)
    print(f"[{label}] extracting logits for {len(texts)} items ({total_chunks} chunks)...", flush=True)
    next_pct = 20
    t0 = time.time()

    for b0 in range(0, total_chunks, batch_size):
        batch = chunk_meta[b0 : b0 + batch_size]
        enc = tok(
            [c[2] for c in batch],
            return_offsets_mapping=True,
            padding=True,
        )
        kw = {
            "input_ids": torch.tensor(enc["input_ids"], device=device),
            "attention_mask": torch.tensor(enc["attention_mask"], device=device),
        }
        if "token_type_ids" in enc:
            kw["token_type_ids"] = torch.tensor(enc["token_type_ids"], device=device)

        logits_np = model(**kw).logits.float().cpu().numpy()

        for bi, (wi, start, _) in enumerate(batch):
            arr = win_logits[wi]
            offsets = enc["offset_mapping"][bi]
            lg = logits_np[bi]
            for ti, (s, e) in enumerate(offsets):
                if e > s:
                    pos = start + e - 1
                    if np.isnan(arr[pos, 0]):
                        arr[pos] = lg[ti]

        done = min(total_chunks, b0 + len(batch))
        pct = int(done * 100 / total_chunks) if total_chunks else 100
        if pct >= next_pct or done == total_chunks:
            print(f"[{label}] {done}/{total_chunks} chunks ({pct}%, {time.time() - t0:.1f}s)", flush=True)
            while next_pct <= pct:
                next_pct += 20

    return win_logits


def decode_labels_from_logits(
    win_logits: list[np.ndarray],
    bias_vec: list[float],
) -> list[list[int]]:
    """文字単位生ロジットにクラス別バイアスを加算し、予測ラベル列へ復号する。"""
    b_arr = np.asarray(bias_vec, dtype=np.float32)
    out: list[list[int]] = []
    for arr in win_logits:
        n = len(arr)
        lbs = np.zeros(n, dtype=np.int64)
        if n > 0:
            cov_mask = ~np.isnan(arr[:, 0])
            if cov_mask.any():
                lbs[cov_mask] = (arr[cov_mask] + b_arr).argmax(axis=-1)
        out.append(lbs.tolist())
    return out


def restore_text(clean_text: str, labels: list[int]) -> str:
    """句読点なしテキストと文字ラベル列から復元済み日本語テキストを生成する。"""
    assert len(clean_text) == len(labels), (len(clean_text), len(labels))
    buf: list[str] = []
    for ch, lb in zip(clean_text, labels):
        buf.append(ch)
        if lb > 0:
            buf.append(ID2PUNCT[lb])
    return "".join(buf)


def eval_sealed_split(
    win_logits: list[np.ndarray],
    golds: list[list[int]],
    bias_vec: list[float],
) -> dict:
    """封印テスト分割の混同行列・micro-F1(O除く)・MCC・記号別 P/R/F1 を集計する。"""
    pred_list = decode_labels_from_logits(win_logits, bias_vec)
    assert len(pred_list) == len(golds)
    for p, g in zip(pred_list, golds):
        assert len(p) == len(g)

    pred_flat = np.concatenate([np.asarray(p, dtype=np.int64) for p in pred_list])
    gold_flat = np.concatenate([np.asarray(g, dtype=np.int64) for g in golds])
    total_chars = sum(len(g) for g in golds)
    assert len(pred_flat) == total_chars

    cm = np.bincount(pred_flat * 5 + gold_flat, minlength=25).reshape(5, 5)
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

    c_correct = float(tp_all.sum())
    cov_xy = c_correct * total - float(np.dot(pred_sum, gold_sum))
    cov_xx = total * total - float(np.dot(pred_sum, pred_sum))
    cov_yy = total * total - float(np.dot(gold_sum, gold_sum))
    denom = math.sqrt(cov_xx * cov_yy)
    mcc = cov_xy / denom if denom > 0 else 0.0
    mcc_sk = float(matthews_corrcoef(gold_flat, pred_flat))
    assert abs(mcc - mcc_sk) <= 1e-6, (mcc, mcc_sk)

    return {
        "logit_bias": [float(x) for x in bias_vec],
        "micro_no_O": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "f1": round(mf, 4),
        },
        "micro_f1_exact": round(mf, 6),
        "mcc_all": round(mcc, 4),
        "mcc_exact_sklearn": round(mcc_sk, 6),
        "per_symbol": per,
        "confusion_matrix_pred_by_gold": cm.tolist(),
    }


def count_pred_symbols(pred_labels_list: list[list[int]]) -> dict:
    """予測ラベル列における記号別の出現総数を集計する。"""
    cnt = Counter()
    for lbs in pred_labels_list:
        for lb in lbs:
            if lb > 0:
                cnt[lb] += 1
    out = {LABELS[c]: int(cnt[c]) for c in range(1, 5)}
    out["total_symbols"] = int(sum(cnt.values()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bias", required=True)
    ap.add_argument("--sealed", required=True)
    ap.add_argument("--asr", required=True)
    ap.add_argument("--asr-baseline", required=True)
    ap.add_argument("--out-sealed", required=True)
    ap.add_argument("--out-asr", required=True)
    ap.add_argument("--out-asr-preds", required=True)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"Device: {device} ({device_name}), dtype=float32, batch_size={args.batch_size}", flush=True)

    bias_obj = json.loads(Path(args.bias).read_text(encoding="utf-8"))
    bias_tuned = [float(x) for x in bias_obj["logit_bias"]]
    bias_zero = [0.0, 0.0, 0.0, 0.0, 0.0]
    assert len(bias_tuned) == 5, bias_tuned

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()

    # ============================================================
    # Part 1: 封印テスト評価 (data/prod/sealed-test.jsonl)
    # ============================================================
    sealed_path = Path(args.sealed)
    sealed_sha = sha256_file(sealed_path)
    sealed_rows = load_jsonl(sealed_path)
    for r in sealed_rows:
        assert len(r["text"]) == len(r["labels"]), "Sealed length mismatch"

    sealed_texts = [r["text"] for r in sealed_rows]
    sealed_golds = [r["labels"] for r in sealed_rows]
    sealed_docs = len({r["doc"] for r in sealed_rows})
    sealed_chars = sum(len(t) for t in sealed_texts)

    t_s = time.time()
    sealed_logits = extract_char_logits(
        model,
        tok,
        sealed_texts,
        device=device,
        max_len=args.max_len,
        overlap=args.overlap,
        batch_size=args.batch_size,
        label="sealed-test",
    )
    if device == "cuda":
        torch.cuda.synchronize()
    sealed_elapsed = time.time() - t_s

    covered_chars = 0
    uncovered_chars = 0
    uncov_gold = np.zeros(5, dtype=np.int64)
    for arr, g in zip(sealed_logits, sealed_golds):
        mask = ~np.isnan(arr[:, 0])
        covered_chars += int(mask.sum())
        uncovered_chars += int((~mask).sum())
        if (~mask).any():
            uncov_gold += np.bincount(np.asarray(g, dtype=np.int64)[~mask], minlength=5)

    sealed_tuned = eval_sealed_split(sealed_logits, sealed_golds, bias_tuned)
    sealed_zero = eval_sealed_split(sealed_logits, sealed_golds, bias_zero)

    sealed_out = {
        "base": bias_obj.get("base", "sbintuitions/modernbert-ja-130m"),
        "model_path": args.model,
        "bias_path": args.bias,
        "environment": {
            "device": device,
            "device_name": device_name,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "dtype": "float32",
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "overlap": args.overlap,
        },
        "sealed_test_sha256": sealed_sha,
        "sealed_windows": len(sealed_rows),
        "sealed_docs": sealed_docs,
        "sealed_chars": sealed_chars,
        "covered_chars": covered_chars,
        "uncovered_chars": uncovered_chars,
        "uncovered_gold_counts": uncov_gold.tolist(),
        "elapsed_sec": round(sealed_elapsed, 2),
        "chars_per_sec": round(sealed_chars / sealed_elapsed, 1) if sealed_elapsed > 0 else 0.0,
        "logit_bias": bias_tuned,
        "tuned": sealed_tuned,
        "bias_zero": sealed_zero,
    }
    Path(args.out_sealed).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_sealed).write_text(json.dumps(sealed_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"SEALED TUNED: micro_no_O={sealed_tuned['micro_no_O']}, mcc={sealed_tuned['mcc_all']}",
        flush=True,
    )

    # ============================================================
    # Part 2: 実 ASR ベンチマーク評価 (FLEURS 650 発話)
    # ============================================================
    asr_records = load_jsonl(Path(args.asr))
    baseline = json.loads(Path(args.asr_baseline).read_text(encoding="utf-8"))
    base_chk = score_asr_punctuation(asr_records, [r["whisper_punct_labels"] for r in asr_records])
    w_all = baseline["whisper_baseline_all"]
    assert base_chk["strict_e2e"]["micro_no_O"]["f1"] == w_all["strict_e2e"]["micro_no_O"]["f1"], (
        base_chk["strict_e2e"]["micro_no_O"]["f1"],
        w_all["strict_e2e"]["micro_no_O"]["f1"],
    )
    assert base_chk["aligned_match_or_sub"]["micro_no_O"]["f1"] == w_all["aligned_match_or_sub"]["micro_no_O"]["f1"]

    asr_texts = [r["asr_text_no_punct"] for r in asr_records]
    asr_chars = sum(len(t) for t in asr_texts)

    t_a = time.time()
    asr_logits = extract_char_logits(
        model,
        tok,
        asr_texts,
        device=device,
        max_len=args.max_len,
        overlap=args.overlap,
        batch_size=args.batch_size,
        label="asr-bench",
    )
    if device == "cuda":
        torch.cuda.synchronize()
    asr_elapsed = time.time() - t_a

    asr_cov = sum(int((~np.isnan(arr[:, 0])).sum()) for arr in asr_logits)
    asr_uncov = sum(int(np.isnan(arr[:, 0]).sum()) for arr in asr_logits)

    pred_labels_tuned = decode_labels_from_logits(asr_logits, bias_tuned)
    pred_labels_zero = decode_labels_from_logits(asr_logits, bias_zero)
    for rec, pt, pz in zip(asr_records, pred_labels_tuned, pred_labels_zero):
        n = len(rec["asr_text_no_punct"])
        assert len(pt) == n and len(pz) == n

    score_tuned = score_asr_punctuation(asr_records, pred_labels_tuned)
    score_zero = score_asr_punctuation(asr_records, pred_labels_zero)

    w_strict = w_all["strict_e2e"]
    m_strict = score_tuned["strict_e2e"]
    delta_vs_whisper = {
        "micro_no_O_f1": round(m_strict["micro_no_O"]["f1"] - w_strict["micro_no_O"]["f1"], 4),
        "mcc_all": round(m_strict["mcc_all"] - w_strict["mcc_all"], 4),
        "COMMA_f1": round(m_strict["per_symbol"]["COMMA"]["f1"] - w_strict["per_symbol"]["COMMA"]["f1"], 4),
        "PERIOD_f1": round(m_strict["per_symbol"]["PERIOD"]["f1"] - w_strict["per_symbol"]["PERIOD"]["f1"], 4),
    }

    asr_out = {
        "dataset": baseline.get("dataset", "google/fleurs (ja_jp test)"),
        "whisper_model": baseline.get("whisper_model", "openai/whisper-large-v3-turbo"),
        "model_path": args.model,
        "bias_path": args.bias,
        "logit_bias": bias_tuned,
        "utterances": len(asr_records),
        "unique_utterance_ids": len({r["id"] for r in asr_records}),
        "asr_no_punct_chars": asr_chars,
        "covered_chars": asr_cov,
        "uncovered_chars": asr_uncov,
        "elapsed_sec": round(asr_elapsed, 2),
        "chars_per_sec": round(asr_chars / asr_elapsed, 1) if asr_elapsed > 0 else 0.0,
        "modernbert_tuned": {
            "strict_e2e": score_tuned["strict_e2e"],
            "aligned_match_or_sub": score_tuned["aligned_match_or_sub"],
            "pred_symbol_counts": count_pred_symbols(pred_labels_tuned),
        },
        "modernbert_bias_zero": {
            "strict_e2e": score_zero["strict_e2e"],
            "aligned_match_or_sub": score_zero["aligned_match_or_sub"],
            "pred_symbol_counts": count_pred_symbols(pred_labels_zero),
        },
        "whisper_baseline_all": w_all,
        "delta_vs_whisper_strict_e2e": delta_vs_whisper,
    }
    Path(args.out_asr).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_asr).write_text(json.dumps(asr_out, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(args.out_asr_preds, "w", encoding="utf-8") as f:
        for rec, pt in zip(asr_records, pred_labels_tuned):
            row = {
                "id": rec["id"],
                "filename": rec["filename"],
                "ref_text": rec["ref_text"],
                "whisper_text": rec["whisper_text"],
                "asr_text_no_punct": rec["asr_text_no_punct"],
                "modernbert_tuned_text": restore_text(rec["asr_text_no_punct"], pt),
                "modernbert_tuned_labels": pt,
                "cer": rec["cer"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"ASR TUNED strict_e2e: micro_no_O={score_tuned['strict_e2e']['micro_no_O']}, "
        f"mcc={score_tuned['strict_e2e']['mcc_all']} "
        f"(Whisper strict_e2e F1={w_strict['micro_no_O']['f1']}, delta={delta_vs_whisper['micro_no_O_f1']:+.4f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
