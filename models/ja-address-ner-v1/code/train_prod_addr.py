#!/usr/bin/env python3
"""第7号 ja-address-ner-v1 本番学習・dev 評価スクリプト（勝者 tohoku-nlp/bert-base-japanese-v3 専用）。

data/prod/train.jsonl (100,000件) で 2 エポック学習し、
dev.jsonl (5,000件) で第1段階（抽出・分解）および第2段階（辞書リンカー接続）を評価する。
共通の推論・評価関数は eval_sealed_addr.py からも再利用する。

注意: 学習入力に封印テスト（sealed-test.jsonl）は一切使用禁止。
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_linker as el  # noqa: E402
import offsets_util as ou  # noqa: E402
import train_pilot_addr as tpa  # noqa: E402
from linker import AddressLinker  # noqa: E402


def run_inference_and_eval(
    model,
    ds: tpa.AddrDataset,
    pad_id: int,
    device: str,
    linker: AddressLinker,
    pred_out_path: Optional[Path] = None,
    eval_bs: int = 64,
) -> Tuple[dict, dict]:
    """モデル推論を実行し、第1段階評価（抽出・分解＋理論上限）と第2段階リンカー評価を返す。"""
    model.to(device).eval()

    pred_comps_all: List[List[dict]] = []
    pred_addrs_all: List[List[dict]] = []
    oracle_comps_all: List[List[dict]] = []
    oracle_addrs_all: List[List[dict]] = []

    total_chars = sum(len(r["text"]) for r in ds.raw_rows)
    t0 = time.time()
    with torch.no_grad():
        for start_i in range(0, len(ds), eval_bs):
            batch_items = [ds[j] for j in range(start_i, min(start_i + eval_bs, len(ds)))]
            batch_tensors = tpa.collate_fn(batch_items, pad_id)
            kw = {
                "input_ids": batch_tensors["input_ids"].to(device),
                "attention_mask": batch_tensors["attention_mask"].to(device),
            }
            if "token_type_ids" in batch_tensors:
                kw["token_type_ids"] = batch_tensors["token_type_ids"].to(device)
            logits = model(**kw).logits.argmax(dim=-1).cpu().tolist()

            for k, item in enumerate(batch_items):
                g_idx = start_i + k
                seq_len = len(item["input_ids"])
                pred_ids = logits[k][:seq_len]
                raw_text = ds.raw_rows[g_idx]["text"]
                offsets = ds.offsets_list[g_idx]

                p_comps = ou.decode_token_preds_to_components(raw_text, offsets, pred_ids)
                p_addrs = ou.group_components_to_addresses(raw_text, p_comps)
                pred_comps_all.append(p_comps)
                pred_addrs_all.append(p_addrs)

                oracle_ids = [0 if lb == -100 else lb for lb in item["labels"]]
                o_comps = ou.decode_token_preds_to_components(raw_text, offsets, oracle_ids)
                o_addrs = ou.group_components_to_addresses(raw_text, o_comps)
                oracle_comps_all.append(o_comps)
                oracle_addrs_all.append(o_addrs)

    eval_sec = time.time() - t0

    if pred_out_path is not None:
        with open(pred_out_path, "w", encoding="utf-8") as pf:
            for row, p_comps, p_addrs in zip(ds.raw_rows, pred_comps_all, pred_addrs_all):
                pf.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "category": row["category"],
                            "pred_components": p_comps,
                            "pred_addresses": p_addrs,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    stage1 = tpa.evaluate_predictions(ds.raw_rows, pred_comps_all, pred_addrs_all)
    oracle_stage1 = tpa.evaluate_predictions(ds.raw_rows, oracle_comps_all, oracle_addrs_all)

    stage1_report = {
        "eval_rows": len(ds),
        "truncated": ds.truncated_count,
        "eval_sec": round(eval_sec, 3),
        "eval_chars": total_chars,
        "eval_chars_per_sec": round(total_chars / eval_sec, 1) if eval_sec > 0 else 0.0,
        "eval_ms_per_sent": round(eval_sec * 1000.0 / len(ds), 3),
        "address_exact_span": stage1["address_exact_span"],
        "address_overlap_span": stage1["address_overlap_span"],
        "per_component_exact": stage1["per_component_exact"],
        "full_decomp_exact_match_on_extracted": stage1[
            "full_decomp_exact_match_on_extracted"
        ],
        "full_decomp_exact_match_on_matched": stage1[
            "full_decomp_exact_match_on_matched"
        ],
        "no_address_fpr": stage1["no_address_fpr"],
        "per_category": stage1["per_category"],
        "tokenizer_oracle": {
            "address_exact_span": oracle_stage1["address_exact_span"],
            "per_component_exact": oracle_stage1["per_component_exact"],
            "full_decomp_exact_match_on_extracted": oracle_stage1[
                "full_decomp_exact_match_on_extracted"
            ],
        },
        "no_address_fp_examples": stage1["no_address_fp_examples"],
    }

    preds_by_id: Dict[str, List[dict]] = {
        row["id"]: p_addrs for row, p_addrs in zip(ds.raw_rows, pred_addrs_all)
    }
    oracle_linker = el.evaluate_mode(ds.raw_rows, {}, linker, is_oracle=True)
    assert oracle_linker["resolvable"]["accuracy"] == 1.0, oracle_linker["resolvable"]
    assert oracle_linker["unresolvable"]["strict_rejection_rate"] == 1.0, oracle_linker[
        "unresolvable"
    ]
    assert oracle_linker["unresolvable"]["false_assertion_rate"] == 0.0, oracle_linker[
        "unresolvable"
    ]

    model_linker = el.evaluate_mode(ds.raw_rows, preds_by_id, linker, is_oracle=False)
    linker_report = {
        "eval_rows": len(ds),
        "oracle_gold_components": oracle_linker,
        "bert_v3_plus_linker": model_linker,
    }
    return stage1_report, linker_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="tohoku-nlp/bert-base-japanese-v3")
    ap.add_argument("--train", default=r"data/prod/train.jsonl")
    ap.add_argument("--dev", default=r"data/prod/dev.jsonl")
    ap.add_argument("--out", default=r"runs/prod-bert-v3")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if "modernbert" in args.base.lower():
        raise RuntimeError(f"本番対象外モデルが指定されました: {args.base}")
    for p_str in (args.train, args.dev):
        if "sealed" in Path(p_str).name.lower():
            raise RuntimeError(f"封印テストファイルは学習・調整に入力禁止です: {p_str}")

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 本番学習開始: base={args.base}, epochs={args.epochs}, bs={args.bs} ===")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base,
        num_labels=len(ou.LABELS),
        id2label=ou.ID2LABEL,
        label2id=ou.LABEL2ID,
    )
    max_pos = getattr(model.config, "max_position_embeddings", args.max_len)
    max_len = min(args.max_len, int(max_pos))

    t_ds0 = time.time()
    train_ds = tpa.AddrDataset(Path(args.train), tok, max_len=max_len, verify_n=300)
    dev_ds = tpa.AddrDataset(Path(args.dev), tok, max_len=max_len, verify_n=999999)
    print(
        f"  データ準備完了 ({time.time() - t_ds0:.1f}s): "
        f"train={len(train_ds)} (trunc={train_ds.truncated_count}), "
        f"dev={len(dev_ds)} (trunc={dev_ds.truncated_count}), max_len={max_len}"
    )

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=64,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=lambda b: tpa.collate_fn(b, pad_id),
    )

    t0 = time.time()
    train_res = trainer.train()
    train_sec = time.time() - t0

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))

    has_nan = False
    log_lines = []
    for entry in trainer.state.log_history:
        for _, v in entry.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                has_nan = True
        log_lines.append(json.dumps(entry, ensure_ascii=False))
    (out_dir / "train.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    assert not has_nan, "学習ログに NaN/Inf が検出されました"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    linker = AddressLinker()
    stage1_dev, linker_dev = run_inference_and_eval(
        model=model,
        ds=dev_ds,
        pad_id=pad_id,
        device=device,
        linker=linker,
        pred_out_path=out_dir / "dev_predictions.jsonl",
        eval_bs=64,
    )

    final_dir_bytes = sum(f.stat().st_size for f in final_dir.rglob("*") if f.is_file())
    st_path = final_dir / "model.safetensors"
    safetensors_bytes = st_path.stat().st_size if st_path.exists() else 0
    model_params = sum(p.numel() for p in model.parameters())

    eval_dev = {
        "base": args.base,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.bs,
        "max_len": max_len,
        "fp16": bool(torch.cuda.is_available()),
        "train_rows": len(train_ds),
        "train_truncated": train_ds.truncated_count,
        "train_sec": round(train_sec, 2),
        "train_loss": round(float(train_res.training_loss), 5),
        "has_nan_in_train_log": has_nan,
        "model_params": int(model_params),
        "final_dir_bytes": int(final_dir_bytes),
        "model_safetensors_bytes": int(safetensors_bytes),
        **stage1_dev,
    }
    (out_dir / "eval_dev.json").write_text(
        json.dumps(eval_dev, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "linker_eval_dev.json").write_text(
        json.dumps(linker_dev, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "base": eval_dev["base"],
        "train_sec": eval_dev["train_sec"],
        "train_loss": eval_dev["train_loss"],
        "dev_address_exact_f1": eval_dev["address_exact_span"]["f1"],
        "dev_component_micro_f1": eval_dev["per_component_exact"]["micro"]["f1"],
        "dev_component_macro_f1": eval_dev["per_component_exact"]["macro_f1"],
        "dev_no_addr_sentence_fpr": eval_dev["no_address_fpr"]["sentence_fpr"],
        "dev_linker_resolvable_acc": linker_dev["bert_v3_plus_linker"]["resolvable"]["accuracy"],
        "dev_linker_unres_reject": linker_dev["bert_v3_plus_linker"]["unresolvable"]["strict_rejection_rate"],
        "dev_linker_unres_false_assert": linker_dev["bert_v3_plus_linker"]["unresolvable"]["false_assertion_rate"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
