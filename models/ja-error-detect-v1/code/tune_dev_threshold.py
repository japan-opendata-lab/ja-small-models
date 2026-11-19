#!/usr/bin/env python
"""第6号 ja-error-detect-v1 dev 確信度閾値調整・閾値固定スクリプト。

学習済み bert-v3 モデルを用いて dev.jsonl (5,000件) で 1 回推論を行い、
span_mean_err / token_err の 2 方式について閾値スイープを実施する。
公開ゲート案 (1,000文字あたり 0.50 件以下・文単位 FPR 2.0% 以下) および
標本揺らぎ余裕付きバッファ条件 (0.45 件以下・1.8% 以下) に基づき採用閾値を確定する。

使い方:
  python code/tune_dev_threshold.py --model runs/prod-bert-v3/final \
      --dev data/prod/dev.jsonl --out runs/prod-bert-v3/dev_threshold_tuning.json
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as ec  # noqa: E402
import offsets_util as ou  # noqa: E402

THRESHOLD_GRID: List[float] = [
    0.0,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.88,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.975,
    0.98,
    0.985,
    0.99,
    0.993,
    0.995,
    0.997,
]


def _pick_best(candidates: List[dict]) -> Optional[dict]:
    if not candidates:
        return None
    # exact_span.f1 最大、同点時は fp_per_1000_chars 最小・sentence_fpr 最小・span_mean_err 優先
    return max(
        candidates,
        key=lambda c: (
            c["exact_span"]["f1"],
            -c["clean_fpr"]["fp_per_1000_chars"],
            -c["clean_fpr"]["sentence_fpr"],
            1 if c["method"] == "span_mean_err" else 0,
            c["threshold"],
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    if "sealed" in Path(args.dev).name.lower():
        raise RuntimeError(f"封印テストファイルは閾値調整に入力禁止です: {args.dev}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    assert model.config.model_type == "bert", (
        f"勝者は bert-v3 (model_type=bert) のみ許可: {model.config.model_type}"
    )

    max_pos = getattr(model.config, "max_position_embeddings", args.max_len)
    max_len = min(args.max_len, int(max_pos))
    raw_rows, enc_items, offsets_list, gold_tok_labels, trunc = ec.load_and_encode_eval(
        Path(args.dev), tok, max_len=max_len
    )

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probs_list, eval_sec = ec.infer_token_probs(
        model, enc_items, pad_id=pad_id, device=device, batch_size=args.bs
    )
    total_chars = sum(len(r["text"]) for r in raw_rows)

    # 整合検証: span_mean_err (tau=0.0) と offsets_util.decode_token_preds_to_spans が全件完全一致すること
    for row, offs, probs in zip(raw_rows, offsets_list, probs_list):
        sp0, cp0, _ = ec.decode_span_mean_err(row["text"], offs, probs, threshold=0.0)
        sp_ref, cp_ref = ou.decode_token_preds_to_spans(
            row["text"], offs, probs.argmax(axis=-1).tolist()
        )
        assert sp0 == sp_ref and cp0 == cp_ref, (
            f"tau=0.0 復号不一致 (id={row.get('id')}): {sp0} != {sp_ref}"
        )

    oracle_metrics = ec.compute_tokenizer_oracle(raw_rows, offsets_list, gold_tok_labels)

    sweep: List[dict] = []
    for method in ("span_mean_err", "token_err"):
        for thr in THRESHOLD_GRID:
            pred_spans_all = []
            char_preds_all = []
            for row, offs, probs in zip(raw_rows, offsets_list, probs_list):
                sp, cp, _ = ec.decode_with_method(row["text"], offs, probs, method, thr)
                pred_spans_all.append(sp)
                char_preds_all.append(cp)
            ev = ec.evaluate_predictions(raw_rows, pred_spans_all, char_preds_all)
            entry = {"method": method, "threshold": thr, **ev}
            sweep.append(entry)

    gate_cands = [
        c
        for c in sweep
        if c["clean_fpr"]["fp_per_1000_chars"] <= 0.50
        and c["clean_fpr"]["sentence_fpr"] <= 0.020
    ]
    buf_cands = [
        c
        for c in sweep
        if c["clean_fpr"]["fp_per_1000_chars"] <= 0.45
        and c["clean_fpr"]["sentence_fpr"] <= 0.018
    ]

    best_gate = _pick_best(gate_cands)
    best_buffered = _pick_best(buf_cands)

    if best_gate is None:
        print("[error] 厳密ゲート (<=0.50/1000字 かつ <=2.0%) を満たす閾値が見つかりませんでした。")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "error": "no_point_satisfies_strict_gate",
                    "argmax_baseline": sweep[0],
                    "sweep": sweep,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 2

    if (
        best_buffered is not None
        and best_buffered["exact_span"]["f1"] >= best_gate["exact_span"]["f1"] - 0.015
    ):
        chosen = best_buffered
        reason = "buffered_gate (<=0.45/1000chars & <=1.8% sent FPR, F1 drop <=0.015)"
    else:
        chosen = best_gate
        reason = "strict_gate (<=0.50/1000chars & <=2.0% sent FPR)"

    argmax_baseline = sweep[0]  # span_mean_err @ 0.0

    result = {
        "model_dir": str(args.model),
        "dev_path": str(args.dev),
        "dev_rows": len(raw_rows),
        "eval_chars": total_chars,
        "dev_truncated": trunc,
        "eval_sec": round(eval_sec, 3),
        "eval_chars_per_sec": round(total_chars / eval_sec, 1) if eval_sec > 0 else 0.0,
        "selection_rule": reason,
        "argmax_baseline": argmax_baseline,
        "best_gate": best_gate,
        "best_buffered": best_buffered,
        "locked": {
            "method": chosen["method"],
            "threshold": chosen["threshold"],
            "dev_metrics": chosen,
        },
        "tokenizer_oracle": oracle_metrics,
        "sweep": sweep,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== dev 閾値調整結果 ===")
    print(
        f"  argmax (0.0): exact_F1={argmax_baseline['exact_span']['f1']} "
        f"(P={argmax_baseline['exact_span']['precision']}, R={argmax_baseline['exact_span']['recall']}), "
        f"FP/1000={argmax_baseline['clean_fpr']['fp_per_1000_chars']}, "
        f"sent_FPR={argmax_baseline['clean_fpr']['sentence_fpr']:.4f}"
    )
    print(
        f"  locked ({chosen['method']} @ {chosen['threshold']} | {reason}): "
        f"exact_F1={chosen['exact_span']['f1']} "
        f"(P={chosen['exact_span']['precision']}, R={chosen['exact_span']['recall']}), "
        f"FP/1000={chosen['clean_fpr']['fp_per_1000_chars']}, "
        f"sent_FPR={chosen['clean_fpr']['sentence_fpr']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
