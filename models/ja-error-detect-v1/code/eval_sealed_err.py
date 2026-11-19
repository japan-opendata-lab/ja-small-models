#!/usr/bin/env python
"""第6号 ja-error-detect-v1 封印テスト最終評価スクリプト（1回限り）。

dev で確定・固定された単一の (method, threshold) のみを使用し、
sealed-test.jsonl (5,000件) を 1 回だけ評価して eval_sealed.json と
正常文検出全件リスト (sealed_clean_detections.json) を書き出す。

既に eval_sealed.json が存在する場合は重複評価を防ぐため即エラー終了する。
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as ec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sealed", required=True)
    ap.add_argument("--threshold-config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_out = out_dir / "eval_sealed.json"
    clean_det_out = out_dir / "sealed_clean_detections.json"

    if eval_out.exists():
        raise RuntimeError(
            f"封印評価の重複実行を禁止しました（既に存在します）: {eval_out}"
        )

    cfg_path = Path(args.model) / "config.json"
    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg_data.get("model_type") == "bert", (
        f"勝者は bert-v3 (model_type=bert) のみ許可: {cfg_data.get('model_type')}"
    )

    thr_cfg = json.loads(Path(args.threshold_config).read_text(encoding="utf-8"))
    locked = thr_cfg["locked"]
    method = str(locked["method"])
    threshold = float(locked["threshold"])

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    assert model.config.model_type == "bert", (
        f"勝者は bert-v3 (model_type=bert) のみ許可: {model.config.model_type}"
    )

    max_pos = getattr(model.config, "max_position_embeddings", args.max_len)
    max_len = min(args.max_len, int(max_pos))

    raw_rows, enc_items, offsets_list, gold_tok_labels, trunc = ec.load_and_encode_eval(
        Path(args.sealed), tok, max_len=max_len
    )

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probs_list, eval_sec = ec.infer_token_probs(
        model, enc_items, pad_id=pad_id, device=device, batch_size=args.bs
    )
    total_chars = sum(len(r["text"]) for r in raw_rows)

    pred_spans_all: List[List[Tuple[int, int]]] = []
    char_preds_all: List[List[int]] = []
    clean_detections: List[dict] = []

    for idx, (row, offs, probs) in enumerate(zip(raw_rows, offsets_list, probs_list)):
        sp, cp, scs = ec.decode_with_method(row["text"], offs, probs, method, threshold)
        pred_spans_all.append(sp)
        char_preds_all.append(cp)

        if (not row["has_error"]) and sp:
            clean_detections.append(
                {
                    "idx": idx,
                    "id": row.get("id"),
                    "text": row["text"],
                    "pred_spans": [
                        {
                            "start": int(a),
                            "end": int(b),
                            "substr": row["text"][a:b],
                            "score": round(float(sc), 6),
                        }
                        for (a, b), sc in zip(sp, scs)
                    ],
                }
            )

    metrics = ec.evaluate_predictions(raw_rows, pred_spans_all, char_preds_all)
    oracle_metrics = ec.compute_tokenizer_oracle(raw_rows, offsets_list, gold_tok_labels)

    report = {
        "model_dir": str(args.model),
        "model_type": model.config.model_type,
        "sealed_path": str(args.sealed),
        "threshold_config": str(args.threshold_config),
        "locked_method": method,
        "locked_threshold": threshold,
        "sealed_rows": len(raw_rows),
        "sealed_chars": total_chars,
        "sealed_truncated": trunc,
        "eval_sec": round(eval_sec, 3),
        "eval_chars_per_sec": round(total_chars / eval_sec, 1) if eval_sec > 0 else 0.0,
        "eval_ms_per_sent": round(eval_sec * 1000.0 / len(raw_rows), 3),
        "exact_span": metrics["exact_span"],
        "overlap_span": metrics["overlap_span"],
        "char_binary": metrics["char_binary"],
        "per_type_exact": metrics["per_type_exact"],
        "per_type_overlap": metrics["per_type_overlap"],
        "clean_fpr": metrics["clean_fpr"],
        "tokenizer_oracle": oracle_metrics,
        "clean_detections_count": len(clean_detections),
    }

    eval_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    clean_det_out.write_text(
        json.dumps(clean_detections, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "locked_method": method,
        "locked_threshold": threshold,
        "sealed_rows": report["sealed_rows"],
        "sealed_truncated": trunc,
        "exact_span": report["exact_span"],
        "overlap_span": report["overlap_span"],
        "clean_fpr": report["clean_fpr"],
        "clean_detections_count": len(clean_detections),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
