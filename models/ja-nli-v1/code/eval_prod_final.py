#!/usr/bin/env python3
"""第11号 ja-nli-v1 本番最終評価スクリプト（封印テスト1回限り・許可3モデル限定）。

対象許可モデル:
1. ja-nli-v1-prod (runs/prod-modernbert/best, max_len=1024)
2. mdeberta-2mil7 (MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7, max_len=512)
3. mdeberta-mnli-xnli (MoritzLaurer/mDeBERTa-v3-base-mnli-xnli, max_len=512)

※ tohoku-nlp/bert-base-japanese-v3 の封印評価は禁止ロックにより遮断する。
※ data/prod/sealed-test.jsonl へのアクセスはモデルごとに1回限りのロックファイルで保護する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(".")
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import eval_nli  # noqa: E402

LOCK_PATH = ROOT / "data" / "prod" / ".sealed_evaluated.lock"
SEALED_PATH = ROOT / "data" / "prod" / "sealed-test.jsonl"


def load_jsonl_stream(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def enforce_security_and_lock(model_path: str, run_name: str, eval_sets: List[str], allow_sealed_once: bool) -> dict:
    low = (model_path + " " + run_name).lower()
    if "bert-base-japanese" in low:
        raise ValueError("SECURITY GUARD: tohoku-nlp/bert-base-japanese-v3 must NEVER be evaluated on Stage 2 sealed-test!")

    if "sealed_test" in eval_sets:
        if not allow_sealed_once:
            raise ValueError("SECURITY GUARD: --allow-sealed-once is required to evaluate sealed_test.")
        lock_data = json.loads(LOCK_PATH.read_text(encoding="utf-8")) if LOCK_PATH.exists() else {}
        if run_name in lock_data:
            raise RuntimeError(f"SEALED TEST LOCKED: '{run_name}' already evaluated once at {lock_data[run_name]['evaluated_at']}")
        return lock_data
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--eval-sets", default="sealed_test,human_eval,dev")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--allow-sealed-once", action="store_true")
    args = ap.parse_args()

    eval_sets = [x.strip() for x in args.eval_sets.split(",") if x.strip()]
    lock_data = enforce_security_and_lock(args.model, args.name, eval_sets, args.allow_sealed_once)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, reference_compile=False).to(device)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)

    model_max_pos = getattr(model.config, "max_position_embeddings", None)
    eff_max_len = (
        min(args.max_len, int(model_max_pos))
        if (model_max_pos and int(model_max_pos) > 0)
        else args.max_len
    )
    remap, orig_id2label = eval_nli.resolve_label_remap(model.config)
    n_params = sum(p.numel() for p in model.parameters())
    sanity = eval_nli.sanity_check(model, tok, eff_max_len, remap, device)
    print(f"[{args.name}] eff_max_len={eff_max_len}, id2label={orig_id2label} -> remap={remap} | sanity_ok={all(x['ok'] for x in sanity)}")
    for sx in sanity:
        print("  sanity:", sx)

    set_paths = {
        "sealed_test": SEALED_PATH,
        "human_eval": ROOT / "data" / "eval" / "human-eval.jsonl",
        "dev": ROOT / "data" / "prod" / "dev.jsonl",
    }

    results = {
        "run_name": args.name,
        "model_path": args.model,
        "num_params": n_params,
        "max_len": eff_max_len,
        "model_max_position_embeddings": model_max_pos,
        "orig_id2label": orig_id2label,
        "canonical_remap_indices": remap,
        "sanity_check": sanity,
        "datasets": {},
    }

    for ds_name in eval_sets:
        p = set_paths[ds_name]
        rows = load_jsonl_stream(p)
        bert_lens = eval_nli.compute_bert_token_lengths(rows)
        ds_res, preds = eval_nli.evaluate_dataset(
            model, tok, rows, eff_max_len, args.batch_size, remap, device, bert_lens
        )
        results["datasets"][ds_name] = ds_res
        (out_dir / f"preds_{ds_name}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in preds) + "\n",
            encoding="utf-8",
        )
        ov = ds_res["overall"]
        print(
            f"[{args.name}] {ds_name} (n={ov['n']}): "
            f"acc={ov['accuracy']:.4f}, macro_f1={ov['macro_f1']:.4f}, "
            f"ec_flip_all={ov['polarity_flip']['ec_flip_rate_all']:.4f}, "
            f"rag_f1={ov['rag_binary_entailment']['f1']:.4f}, "
            f"trunc={ov['truncated_count']}, {ov['ms_per_pair']:.2f}ms/pair"
        )

    out_json = out_dir / "eval_final_results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if "sealed_test" in eval_sets:
        sealed_sha = hashlib.sha256(SEALED_PATH.read_bytes()).hexdigest()
        lock_data[args.name] = {
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            "model_path": args.model,
            "sealed_sha256": sealed_sha,
            "sealed_accuracy": results["datasets"]["sealed_test"]["overall"]["accuracy"],
            "sealed_macro_f1": results["datasets"]["sealed_test"]["overall"]["macro_f1"],
        }
        LOCK_PATH.write_text(json.dumps(lock_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[LOCK UPDATED] {LOCK_PATH} recorded one-shot sealed_test eval for {args.name}")


if __name__ == "__main__":
    main()
