#!/usr/bin/env python3
"""第3号 ja-punct-restore-v1: FLEURS 日本語テスト分割の実 ASR ベンチマーク構築＆Whisper ベースライン採点。

google/fleurs (ja_jp test split) を openai/whisper-large-v3-turbo で書き起こし、
1. 句読点除去済み ASR テキストと参照文・文字ラベルを組にした data/asr-bench/fleurs-ja-test.jsonl を構築
2. Whisper の句読点つき実出力そのものをベースラインとして End-to-End 文字アライン採点を実施
する。score_asr_punctuation() は次工程の自作モデル / Horizon 評価でもそのまま再利用可能。
"""
import argparse
import io
import json
import math
import re
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

LABELS = ["O", "COMMA", "PERIOD", "QUESTION", "EXCLAIM"]
PUNCT_MAP = {"，": "、", "．": "。"}
TARGETS = {"、": 1, "。": 2, "？": 3, "!": 4, "！": 4}
RAW_PUNCT_CHARS = ("、", "。", "？", "！", "，", "．", ",", ".", "?", "!")


def normalize_text(text: str) -> str:
    """SPEC-v1 / build_punct_data.py と同一の句読点・空白正規化を行う。"""
    for k, v in PUNCT_MAP.items():
        text = text.replace(k, v)
    text = text.replace("!", "！")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_labels(text: str) -> tuple[str, list[int]]:
    """正規化済みテキストから (対象約物除去済みテキスト, 文字ラベル列) を返す。"""
    chars, labels = [], []
    for ch in text:
        if ch in TARGETS:
            if labels:
                labels[-1] = TARGETS[ch]
            continue
        chars.append(ch)
        labels.append(0)
    return "".join(chars), labels


def strip_spaces_preserve_labels(clean_text: str, labels: list[int]) -> tuple[str, list[int]]:
    """アラインメント比較用に空白を除去しつつ、直後の約物ラベルを直前の非空白文字へ保持する。"""
    out_chars, out_labels = [], []
    for ch, lb in zip(clean_text, labels):
        if ch.isspace():
            if lb != 0 and out_labels:
                out_labels[-1] = lb
            continue
        out_chars.append(ch)
        out_labels.append(lb)
    return "".join(out_chars), out_labels


def align_char_sequences(ref_seq: str, hyp_seq: str) -> tuple[list[tuple[str, int, int]], int]:
    """文字単位レーベンシュタイン最適 DP アラインメントを返す。

    戻り値: (ops, edit_distance)
      ops の各要素は (op, ref_idx, hyp_idx) で op in {"MATCH", "SUB", "DEL", "INS"}。
    """
    n, m = len(ref_seq), len(hyp_seq)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1, dtype=np.int32)
    dp[0, :] = np.arange(m + 1, dtype=np.int32)

    for i in range(1, n + 1):
        rc = ref_seq[i - 1]
        row_prev = dp[i - 1]
        row_cur = dp[i]
        for j in range(1, m + 1):
            cost_sub = row_prev[j - 1] + (0 if rc == hyp_seq[j - 1] else 1)
            cost_del = row_prev[j] + 1
            cost_ins = row_cur[j - 1] + 1
            best = cost_sub
            if cost_del < best:
                best = cost_del
            if cost_ins < best:
                best = cost_ins
            row_cur[j] = best

    ops: list[tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        cur = dp[i, j]
        if i > 0 and j > 0:
            is_eq = ref_seq[i - 1] == hyp_seq[j - 1]
            if cur == dp[i - 1, j - 1] + (0 if is_eq else 1):
                ops.append(("MATCH" if is_eq else "SUB", i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and cur == dp[i - 1, j] + 1:
            ops.append(("DEL", i - 1, -1))
            i -= 1
        else:
            ops.append(("INS", -1, j - 1))
            j -= 1
    ops.reverse()
    return ops, int(dp[n, m])


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
        "mcc_all": round(mcc, 4),
        "per_symbol": per,
    }


def score_asr_punctuation(records: list[dict], pred_labels_list: list[list[int]]) -> dict:
    """ASR テキスト上の予測文字ラベル列を、参照文との最適文字アラインメントで採点する。

    - strict_e2e: MATCH 文字位置上でのみ TP を認める厳格 End-to-End 採点
      （SUB/DEL/INS 文字に付いた約物は FP/FN として計上する）
    - aligned_match_or_sub: MATCH および SUB 位置を対応文字対とみなす参考採点
    """
    assert len(records) == len(pred_labels_list)
    cm_strict = np.zeros((5, 5), dtype=np.int64)
    cm_sub_ok = np.zeros((5, 5), dtype=np.int64)
    total_ref_chars = 0
    total_hyp_chars = 0
    total_edit = 0

    for rec, pred_lbs in zip(records, pred_labels_list):
        ref_seq, ref_lbs = strip_spaces_preserve_labels(rec["ref_clean"], rec["ref_labels"])
        hyp_seq, hyp_lbs = strip_spaces_preserve_labels(rec["asr_text_no_punct"], pred_lbs)
        ops, dist = align_char_sequences(ref_seq, hyp_seq)
        total_ref_chars += len(ref_seq)
        total_hyp_chars += len(hyp_seq)
        total_edit += dist

        for op, ri, hi in ops:
            if op == "MATCH":
                rl = ref_lbs[ri]
                hl = hyp_lbs[hi]
                cm_strict[hl, rl] += 1
                cm_sub_ok[hl, rl] += 1
            elif op == "SUB":
                rl = ref_lbs[ri]
                hl = hyp_lbs[hi]
                cm_sub_ok[hl, rl] += 1
                # 厳格集計では文字誤りのため正解(TP)にせず、約物があれば FP / FN として扱う
                if hl != 0:
                    cm_strict[hl, 0] += 1
                if rl != 0:
                    cm_strict[0, rl] += 1
                if hl == 0 and rl == 0:
                    cm_strict[0, 0] += 1
            elif op == "DEL":
                rl = ref_lbs[ri]
                cm_strict[0, rl] += 1
                cm_sub_ok[0, rl] += 1
            else:  # INS
                hl = hyp_lbs[hi]
                cm_strict[hl, 0] += 1
                cm_sub_ok[hl, 0] += 1

    return {
        "utterances": len(records),
        "total_ref_chars_no_space": total_ref_chars,
        "total_hyp_chars_no_space": total_hyp_chars,
        "total_edit_distance": total_edit,
        "corpus_cer": round(total_edit / total_ref_chars, 4) if total_ref_chars else 0.0,
        "strict_e2e": metrics_from_cm(cm_strict),
        "aligned_match_or_sub": metrics_from_cm(cm_sub_ok),
    }


def load_fleurs_ja_test() -> tuple[list[dict], dict[str, tuple[np.ndarray, int, float]]]:
    """google/fleurs の data/ja_jp/test.tsv と audio/test.tar.gz をダウンロードして読み込む。"""
    import soundfile as sf

    tsv_path = hf_hub_download("google/fleurs", "data/ja_jp/test.tsv", repo_type="dataset")
    tar_path = hf_hub_download("google/fleurs", "data/ja_jp/audio/test.tar.gz", repo_type="dataset")

    meta_rows = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            meta_rows.append({
                "id": cols[0],
                "filename": cols[1],
                "ref_text": cols[2],
            })

    audios: dict[str, tuple[np.ndarray, int, float]] = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith(".wav"):
                fname = Path(m.name).name
                raw_bytes = tf.extractfile(m).read()
                arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                assert sr == 16000, f"Expected 16000Hz wav, got {sr} for {fname}"
                dur = float(len(arr)) / float(sr)
                audios[fname] = (arr, sr, dur)

    for r in meta_rows:
        assert r["filename"] in audios, f"Missing wav file: {r['filename']}"
    return meta_rows, audios


def transcribe_batch(
    model,
    processor,
    batch_arrs: list[np.ndarray],
    device: str,
    dtype: torch.dtype,
    gen_kwargs: dict,
    batch_durs: list[float] | None = None,
) -> list[str]:
    """Whisper model.generate でバッチ書き起こしを行う（30秒超のみ return_timestamps=True にフォールバック）。"""
    if batch_durs is not None and any(d > 30.0 for d in batch_durs):
        results = [""] * len(batch_arrs)
        short_idx = [i for i, d in enumerate(batch_durs) if d <= 30.0]
        long_idx = [i for i, d in enumerate(batch_durs) if d > 30.0]
        if short_idx:
            short_txts = transcribe_batch(
                model, processor, [batch_arrs[i] for i in short_idx], device, dtype, gen_kwargs, batch_durs=None
            )
            for idx, txt in zip(short_idx, short_txts):
                results[idx] = txt
        for idx in long_idx:
            inp = processor(
                [batch_arrs[idx]], sampling_rate=16000, return_tensors="pt", return_attention_mask=True
            )
            inp["input_features"] = inp["input_features"].to(device, dtype=dtype)
            if "attention_mask" in inp:
                inp["attention_mask"] = inp["attention_mask"].to(device)
            with torch.no_grad():
                ids = model.generate(**inp, **gen_kwargs, return_timestamps=True)
            results[idx] = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return results

    inputs = processor(
        batch_arrs, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
    )
    inputs["input_features"] = inputs["input_features"].to(device, dtype=dtype)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, **gen_kwargs)
    return [t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--out", default="data/asr-bench")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    print("Downloading / loading FLEURS ja_jp test split...", flush=True)
    meta_rows, audios = load_fleurs_ja_test()
    print(f"Loaded {len(meta_rows)} utterances ({len(set(r['id'] for r in meta_rows))} unique ids).", flush=True)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Loading ASR model {args.whisper} on {device} ({dtype})...", flush=True)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.whisper, dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.whisper)

    gen_kwargs = {
        "language": "ja",
        "task": "transcribe",
        "num_beams": 5,
        "compression_ratio_threshold": 2.4,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }

    # 先頭 3 発話のスモークテスト
    smoke_arrs = [audios[r["filename"]][0] for r in meta_rows[:3]]
    smoke_durs = [audios[r["filename"]][2] for r in meta_rows[:3]]
    smoke_out = transcribe_batch(model, processor, smoke_arrs, device, dtype, gen_kwargs, batch_durs=smoke_durs)
    for i, txt in enumerate(smoke_out):
        assert len(txt) > 0, f"Empty whisper output on smoke {i}"
        print(f"[SMOKE {i}] dur={smoke_durs[i]:.2f}s hyp={txt[:60]}", flush=True)

    # 全発話の書き起こし
    t_asr = time.time()
    all_texts: list[str] = []
    n_total = len(meta_rows)
    next_log = 100
    for start in range(0, n_total, args.batch_size):
        batch = meta_rows[start:start + args.batch_size]
        batch_arrs = [audios[r["filename"]][0] for r in batch]
        batch_durs = [audios[r["filename"]][2] for r in batch]
        all_texts.extend(
            transcribe_batch(model, processor, batch_arrs, device, dtype, gen_kwargs, batch_durs=batch_durs)
        )
        done = len(all_texts)
        if done >= next_log or done == n_total:
            print(f"Transcribed {done}/{n_total} utterances ({time.time() - t_asr:.1f}s)", flush=True)
            next_log += 100
    asr_sec = time.time() - t_asr
    print(f"Transcribed {len(all_texts)} utterances in {asr_sec:.1f}s.", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    ref_raw_punct = Counter()
    hyp_raw_punct = Counter()
    durations = []
    utt_cers = []

    for r, hyp_text in zip(meta_rows, all_texts):
        _, _, dur = audios[r["filename"]]
        durations.append(dur)
        ref_text = r["ref_text"]
        for ch in ref_text:
            if ch in RAW_PUNCT_CHARS:
                ref_raw_punct[ch] += 1
        for ch in hyp_text:
            if ch in RAW_PUNCT_CHARS:
                hyp_raw_punct[ch] += 1

        ref_norm = normalize_text(ref_text)
        ref_clean, ref_labels = make_labels(ref_norm)

        hyp_norm = normalize_text(hyp_text)
        asr_no_punct, whisper_punct_labels = make_labels(hyp_norm)

        # 発話単体の CER（空白・記号除去後）
        ref_ns, _ = strip_spaces_preserve_labels(ref_clean, ref_labels)
        hyp_ns, _ = strip_spaces_preserve_labels(asr_no_punct, whisper_punct_labels)
        _, ed = align_char_sequences(ref_ns, hyp_ns)
        utt_cer = round(ed / len(ref_ns), 4) if ref_ns else 0.0
        utt_cers.append(utt_cer)

        records.append({
            "id": r["id"],
            "filename": r["filename"],
            "duration_sec": round(dur, 3),
            "ref_text": ref_text,
            "ref_norm": ref_norm,
            "ref_clean": ref_clean,
            "ref_labels": ref_labels,
            "whisper_text": hyp_text,
            "asr_text_no_punct": asr_no_punct,
            "whisper_punct_labels": whisper_punct_labels,
            "cer": utt_cer,
        })

    jsonl_path = out_dir / "fleurs-ja-test.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Whisper 句読点つき出力そのものをベースライン採点（全件＋<=30秒サブセット）
    score_all = score_asr_punctuation(records, [r["whisper_punct_labels"] for r in records])
    rec_le30 = [r for r in records if r["duration_sec"] <= 30.0]
    score_le30 = score_asr_punctuation(rec_le30, [r["whisper_punct_labels"] for r in rec_le30])

    # 参考：もし Whisper 出力の半角 '?' も '？' として読んだ場合の比較
    alt_preds = []
    for rec in records:
        alt_norm = normalize_text(rec["whisper_text"]).replace("?", "？")
        alt_clean, alt_lbs = make_labels(alt_norm)
        if alt_clean == rec["asr_text_no_punct"]:
            alt_preds.append(alt_lbs)
        else:
            alt_preds.append(rec["whisper_punct_labels"])
    score_q_norm = score_asr_punctuation(records, alt_preds) if hyp_raw_punct.get("?", 0) > 0 else None

    summary = {
        "dataset": "google/fleurs (ja_jp test)",
        "whisper_model": args.whisper,
        "whisper_decode_config": {
            "method": "model.generate",
            "return_timestamps": False,
            "num_beams": 5,
            "compression_ratio_threshold": 2.4,
            "temperature": list(gen_kwargs["temperature"]),
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "language": "ja",
            "task": "transcribe",
            "batch_size": args.batch_size,
            "seed": 42,
        },
        "utterances": len(records),
        "unique_utterance_ids": len({r["id"] for r in records}),
        "utterances_le_30s": len(rec_le30),
        "utterances_gt_30s": len(records) - len(rec_le30),
        "duration_sec": {
            "min": round(min(durations), 2),
            "median": round(float(np.median(durations)), 2),
            "max": round(max(durations), 2),
            "total": round(sum(durations), 1),
        },
        "cer_stats": {
            "median_utt_cer": round(float(np.median(utt_cers)), 4),
            "max_utt_cer": round(max(utt_cers), 4),
            "utterances_cer_gt_0_5": sum(c > 0.5 for c in utt_cers),
            "utterances_cer_gt_1_0": sum(c > 1.0 for c in utt_cers),
        },
        "asr_elapsed_sec": round(asr_sec, 1),
        "total_elapsed_sec": round(time.time() - t0, 1),
        "raw_punct_counts_ref": {ch: int(ref_raw_punct[ch]) for ch in RAW_PUNCT_CHARS if ref_raw_punct[ch]},
        "raw_punct_counts_whisper": {ch: int(hyp_raw_punct[ch]) for ch in RAW_PUNCT_CHARS if hyp_raw_punct[ch]},
        "whisper_baseline_all": score_all,
        "whisper_baseline_le_30s": score_le30,
        "whisper_baseline_halfwidth_q_mapped": score_q_norm,
        "jsonl_bytes": jsonl_path.stat().st_size,
    }
    (out_dir / "whisper-baseline-eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("BENCHMARK SUMMARY:", json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
