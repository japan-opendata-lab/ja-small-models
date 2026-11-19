#!/usr/bin/env python3
"""第8号 ja-toxicity-guard-v1 封印テスト最終評価スクリプト（1回限り実行）。

dev のみで確定・固定された採用閾値（adopted_thresholds.json）を用い、
data/prod/sealed-test.jsonl（5,000件）を 1 回限り評価して eval_sealed.json に保存する。

安全ガード:
  - 出力ファイル (--out) が既に存在する場合は即座に RuntimeError で終了し再評価を防ぐ。
  - --model に "modernbert" を含む場合は即座に RuntimeError で終了する。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toxicity_prod_templates as pt  # noqa: E402
import train_prod_tox as tpt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--sealed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preds-out", required=True)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        raise RuntimeError(f"封印テストは1回限りです。出力ファイルが既に存在します: {out_path}")
    if "modernbert" in args.model.lower():
        raise RuntimeError(f"非対象モデルでの封印評価は禁止です: {args.model}")

    thr_payload = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    adopted_thrs = {lb: float(thr_payload["thresholds"][lb]) for lb in pt.LABELS}

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    max_pos = getattr(model.config, "max_position_embeddings", args.max_len)
    max_len = min(args.max_len, int(max_pos))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    sealed_ds = tpt.ToxicityDataset(Path(args.sealed), tok, max_len=max_len)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_chars = sum(len(r["text"]) for r in sealed_ds.raw_rows)

    t0 = time.time()
    probs = tpt.predict_probs(model, sealed_ds, pad_id, device=device, batch_size=32)
    eval_sec = time.time() - t0

    metrics = tpt.evaluate_with_thresholds(sealed_ds.raw_rows, probs, adopted_thrs)
    default_thrs = {lb: 0.50 for lb in pt.LABELS}
    changed = any(abs(adopted_thrs[lb] - 0.50) > 1e-9 for lb in pt.LABELS)
    ref_at_05 = tpt.evaluate_with_thresholds(sealed_ds.raw_rows, probs, default_thrs) if changed else None

    preds_out = Path(args.preds_out)
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    with open(preds_out, "w", encoding="utf-8") as pf:
        for row, p_vec in zip(sealed_ds.raw_rows, probs):
            pred_lbs = [pt.LABELS[j] for j in range(len(pt.LABELS)) if p_vec[j] >= adopted_thrs[pt.LABELS[j]]]
            pf.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "is_toxic": row["is_toxic"],
                        "benign_subtype": row["benign_subtype"],
                        "active_labels": row["active_labels"],
                        "pred_labels": pred_lbs,
                        "max_prob": round(float(p_vec.max()), 6),
                        "probs": {pt.LABELS[j]: round(float(p_vec[j]), 6) for j in range(len(pt.LABELS))},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "model_dir": args.model,
        "sealed_path": args.sealed,
        "max_len": max_len,
        "sealed_rows": len(sealed_ds),
        "sealed_truncated": sealed_ds.truncated_count,
        "eval_sec": round(eval_sec, 3),
        "eval_chars": total_chars,
        "eval_chars_per_sec": round(total_chars / eval_sec, 1) if eval_sec > 0 else 0.0,
        "eval_ms_per_doc": round(eval_sec * 1000.0 / len(sealed_ds), 3),
        "adopted_thresholds": adopted_thrs,
        "thresholds_changed_from_0.5": changed,
        **metrics,
        "reference_at_0.5": ref_at_05,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "sealed_rows": report["sealed_rows"],
                "sealed_truncated": report["sealed_truncated"],
                "eval_sec": report["eval_sec"],
                "macro_f1": report["macro_f1"],
                "micro_f1": report["micro"]["f1"],
                "exact_match_accuracy": report["exact_match_accuracy"],
                "crit_news_fpr": report["benign_fpr"]["criticism_and_news_quote"]["fpr"],
                "all_benign_fpr": report["benign_fpr"]["all_benign"]["fpr"],
                "per_label_f1": {k: v["f1"] for k, v in report["per_label"].items()},
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
