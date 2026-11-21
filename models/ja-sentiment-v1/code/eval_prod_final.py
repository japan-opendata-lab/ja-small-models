#!/usr/bin/env python3
"""第12号 ja-sentiment-v1 本番ワンショット最終評価スクリプト。

安全装置:
1. out_dir / "sealed_eval_receipt.json" が既に存在する場合は即停止（封印テスト二重評価防止）。
2. bert-base-japanese-v3 が指定された場合は即停止（仕様上 BERT-v3 の封印評価は禁止）。
3. 評価実行前に data/prod/sealed-test.jsonl のバイト数および SHA256 を
   data/prod/build-stats.json と照合し、未開封同一性を検証する。
4. sealed-test のテキスト本文はコンソールおよび予測JSONLへ一切出力しない。
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_sentiment  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def forbid_bert_v3(model_id: str, config=None) -> None:
    low = model_id.lower()
    if "bert-base-japanese-v3" in low or "pilot-bert-v3" in low:
        raise ValueError(f"SECURITY GUARD: Evaluation of {model_id} on sealed-test is FORBIDDEN!")
    if config is not None:
        name_or_path = str(getattr(config, "_name_or_path", "")).lower()
        if "bert-base-japanese-v3" in name_or_path or "pilot-bert-v3" in name_or_path:
            raise ValueError(f"SECURITY GUARD: Loaded model config originates from {name_or_path}, FORBIDDEN!")


def verify_sealed_integrity(sealed_path: Path, build_stats_path: Path) -> dict:
    st = json.loads(build_stats_path.read_text(encoding="utf-8"))["splits"]["sealed_test"]
    raw = sealed_path.read_bytes()
    cur_bytes = len(raw)
    cur_sha = hashlib.sha256(raw).hexdigest()
    if cur_bytes != st["bytes"] or cur_sha != st["sha256"]:
        raise RuntimeError(
            f"Sealed test integrity mismatch! expected=({st['bytes']}, {st['sha256']}), "
            f"actual=({cur_bytes}, {cur_sha})"
        )
    return {"path": str(sealed_path), "bytes": cur_bytes, "sha256": cur_sha, "rows_expected": st["rows"]}


def eval_single_model(
    model_path: str,
    run_name: str,
    datasets: Dict[str, List[dict]],
    out_dir: Path,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    forbid_bert_v3(model_path)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    forbid_bert_v3(model_path, model.config)

    model_max_pos = getattr(model.config, "max_position_embeddings", None)
    eff_max_len = min(max_len, int(model_max_pos)) if (model_max_pos and int(model_max_pos) > 0) else max_len
    mode, remap, orig_id2label = eval_sentiment.resolve_label_remap(model.config)
    n_params = sum(p.numel() for p in model.parameters())

    sanity = eval_sentiment.sanity_check(model, tok, eff_max_len, mode, remap, device)
    print(f"[{run_name}] mode={mode}, id2label={orig_id2label} -> remap={remap} | sanity_ok={all(x['ok'] for x in sanity)}")
    for sx in sanity:
        print("  sanity:", sx)

    results = {
        "run_name": run_name,
        "model_path": model_path,
        "label_mode": mode,
        "num_params": n_params,
        "max_len": eff_max_len,
        "model_max_position_embeddings": model_max_pos,
        "orig_id2label": orig_id2label,
        "canonical_remap_indices": remap,
        "sanity_check": sanity,
        "datasets": {},
    }

    for ds_name, rows in datasets.items():
        ds_res, preds = eval_sentiment.evaluate_dataset(
            model, tok, rows, eff_max_len, batch_size, mode, remap, device, include_calibration=True
        )
        results["datasets"][ds_name] = ds_res
        # sealed を含む各データセットとも予測JSONLにはテキスト本文を入れない
        (out_dir / f"preds_{ds_name}_{run_name}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in preds) + "\n",
            encoding="utf-8",
        )
        ov = ds_res["overall"]
        cal = ov["calibration"]
        bin_m = ov["binary_no_neutral"]
        print(
            f"[{run_name}] {ds_name} (n={ov['n']}): "
            f"3cls_acc={ov['accuracy']:.4f}, macro_f1={ov['macro_f1']:.4f}, "
            f"pn_flip={ov['polarity_flip']['pn_flip_count']} ({ov['polarity_flip']['pn_flip_rate_all']:.4f}), "
            f"nll={cal['nll']:.5f}, brier={cal['brier_score']:.5f}, ece={cal['ece_15bins']:.5f} | "
            f"bin2cls_acc={bin_m['accuracy']:.4f}, bin2cls_f1={bin_m['macro_f1']:.4f}, roc_auc={bin_m['roc_auc_pos']:.4f} | "
            f"trunc={ov['truncated_count']}, {ov['ms_per_sample']:.2f}ms/sample",
            flush=True,
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out_name = f"eval_{run_name.replace('-', '_')}.json"
    (out_dir / out_name).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-model", default=str(ROOT / "runs" / "prod-modernbert" / "best"))
    ap.add_argument("--baseline-model", default="christian-phu/bert-finetuned-japanese-sentiment")
    ap.add_argument("--sealed-file", default=str(ROOT / "data" / "prod" / "sealed-test.jsonl"))
    ap.add_argument("--build-stats", default=str(ROOT / "data" / "prod" / "build-stats.json"))
    ap.add_argument("--human-eval-file", default=str(ROOT / "data" / "eval" / "human-eval.jsonl"))
    ap.add_argument("--dev-file", default=str(ROOT / "data" / "prod" / "dev.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "prod-final-eval"))
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    receipt_path = out_dir / "sealed_eval_receipt.json"
    if receipt_path.exists():
        raise RuntimeError(
            f"SECURITY GUARD: {receipt_path} already exists! "
            "prod sealed-test can only be evaluated ONCE."
        )

    sealed_path = Path(args.sealed_file)
    sealed_meta = verify_sealed_integrity(sealed_path, Path(args.build_stats))

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = {
        "prod_sealed_test": [json.loads(ln) for ln in sealed_path.read_text(encoding="utf-8").splitlines() if ln.strip()],
        "human_eval": [json.loads(ln) for ln in Path(args.human_eval_file).read_text(encoding="utf-8").splitlines() if ln.strip()],
        "prod_dev": [json.loads(ln) for ln in Path(args.dev_file).read_text(encoding="utf-8").splitlines() if ln.strip()],
    }

    t0 = time.perf_counter()
    res_prod = eval_single_model(
        args.prod_model, "prod-modernbert", datasets, out_dir, args.max_len, args.batch_size, device
    )
    res_base = eval_single_model(
        args.baseline_model, "baseline-christian-phu", datasets, out_dir, args.max_len, args.batch_size, device
    )
    elapsed = round(time.perf_counter() - t0, 2)

    receipt = {
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "elapsed_sec": elapsed,
        "sealed_test_integrity": sealed_meta,
        "evaluated_models": [
            {"run_name": "prod-modernbert", "model_path": args.prod_model},
            {"run_name": "baseline-christian-phu", "model_path": args.baseline_model},
        ],
        "bert_v3_sealed_evaluated": False,
        "dataset_counts": {k: len(v) for k, v in datasets.items()},
        "summary_prod_sealed_test": {
            "prod-modernbert": {
                "3cls_acc": res_prod["datasets"]["prod_sealed_test"]["overall"]["accuracy"],
                "3cls_macro_f1": res_prod["datasets"]["prod_sealed_test"]["overall"]["macro_f1"],
                "pn_flip_count": res_prod["datasets"]["prod_sealed_test"]["overall"]["polarity_flip"]["pn_flip_count"],
                "bin2cls_acc": res_prod["datasets"]["prod_sealed_test"]["overall"]["binary_no_neutral"]["accuracy"],
            },
            "baseline-christian-phu": {
                "3cls_acc": res_base["datasets"]["prod_sealed_test"]["overall"]["accuracy"],
                "3cls_macro_f1": res_base["datasets"]["prod_sealed_test"]["overall"]["macro_f1"],
                "pn_flip_count": res_base["datasets"]["prod_sealed_test"]["overall"]["polarity_flip"]["pn_flip_count"],
                "bin2cls_acc": res_base["datasets"]["prod_sealed_test"]["overall"]["binary_no_neutral"]["accuracy"],
            },
        },
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FINAL EVAL COMPLETE] receipt saved -> {receipt_path}")


if __name__ == "__main__":
    main()
