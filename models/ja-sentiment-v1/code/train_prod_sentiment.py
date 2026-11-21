#!/usr/bin/env python3
"""第12号 ja-sentiment-v1 本番学習スクリプト (sbintuitions/modernbert-ja-130m 3エポック)。

各エポック終了時に data/prod/dev.jsonl (5,000件) で評価し、
事前固定基準 (dev_3cls_macro_f1 最大, -dev_pn_flip_rate_all 最大, -dev_nll 最大) で
最良エポックを選定して runs/prod-modernbert/best/ へ保存する。
※ sealed-test.jsonl および human-eval.jsonl には一切アクセスしない。
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_sentiment  # noqa: E402
import train_pilot_sentiment as tps  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def guard_path_safe(path_str: str, role: str) -> None:
    low = path_str.lower()
    if "sealed" in low or "human-eval" in low or "human_eval" in low:
        raise ValueError(f"SECURITY GUARD: {role} path must NOT reference sealed or human_eval: {path_str}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="sbintuitions/modernbert-ja-130m")
    ap.add_argument("--name", default="prod-modernbert")
    ap.add_argument("--train-file", default=str(ROOT / "data" / "prod" / "train.jsonl"))
    ap.add_argument("--dev-file", default=str(ROOT / "data" / "prod" / "dev.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "prod-modernbert"))
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--per-device-bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    guard_path_safe(args.train_file, "train-file")
    guard_path_safe(args.dev_file, "dev-file")

    tps.set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    tok = AutoTokenizer.from_pretrained(args.base_model)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=3,
            label2id=tps.LABEL2ID,
            id2label=tps.ID2LABEL,
            reference_compile=False,
        ).to(device)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=3,
            label2id=tps.LABEL2ID,
            id2label=tps.ID2LABEL,
        ).to(device)

    model_max_pos = getattr(model.config, "max_position_embeddings", None)
    eff_max_len = min(args.max_len, int(model_max_pos)) if (model_max_pos and int(model_max_pos) > 0) else args.max_len
    n_params = sum(p.numel() for p in model.parameters())

    train_rows = [json.loads(ln) for ln in Path(args.train_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    dev_rows = [json.loads(ln) for ln in Path(args.dev_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    train_ds = tps.SentimentDataset(train_rows, tok, eff_max_len)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    loader = DataLoader(
        train_ds,
        batch_size=args.per_device_bs,
        shuffle=True,
        collate_fn=tps.make_collate_fn(pad_id),
        generator=torch.Generator().manual_seed(args.seed),
    )

    steps_per_epoch_micro = len(loader)
    steps_per_epoch_opt = math.ceil(steps_per_epoch_micro / args.grad_accum)
    total_opt_steps = steps_per_epoch_opt * args.epochs
    warmup_steps = int(total_opt_steps * args.warmup_ratio)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    print(
        f"[PROD TRAIN {args.name}] base={args.base_model}, params={n_params:,}, "
        f"train={len(train_ds)}, dev={len(dev_rows)}, epochs={args.epochs}, "
        f"eff_max_len={eff_max_len}, train_trunc={train_ds.truncated_count}, "
        f"opt_steps={total_opt_steps} ({steps_per_epoch_opt}/ep), warmup={warmup_steps}, bf16={use_bf16}",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_all = time.perf_counter()

    opt_step = 0
    has_nan = False
    epoch_summaries: List[dict] = []

    for ep in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ep_losses: List[float] = []
        ep_step_logs: List[dict] = []
        acc_loss = 0.0
        micro_in_acc = 0

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_ep = time.perf_counter()

        for m_idx, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(use_bf16 or use_fp16)):
                out = model(**batch)
                loss = out.loss / args.grad_accum
            loss_val = float(out.loss.detach().item())
            if math.isnan(loss_val) or math.isinf(loss_val):
                has_nan = True
                raise RuntimeError(f"NaN/Inf loss at epoch={ep}, micro={m_idx}")
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            acc_loss += loss_val
            micro_in_acc += 1

            if m_idx % args.grad_accum == 0 or m_idx == steps_per_epoch_micro:
                if use_fp16:
                    scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item())
                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    has_nan = True
                    raise RuntimeError(f"NaN/Inf grad_norm at opt_step={opt_step+1}")
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                mean_step_loss = acc_loss / micro_in_acc
                ep_losses.append(mean_step_loss)
                acc_loss = 0.0
                micro_in_acc = 0

                if opt_step % 250 == 0 or m_idx == steps_per_epoch_micro:
                    avg50 = sum(ep_losses[-50:]) / len(ep_losses[-50:])
                    log_item = {
                        "epoch": ep,
                        "opt_step": opt_step,
                        "loss": round(mean_step_loss, 5),
                        "loss_avg50": round(avg50, 5),
                        "grad_norm": round(grad_norm, 4),
                        "lr": round(scheduler.get_last_lr()[0], 8),
                    }
                    ep_step_logs.append(log_item)
                    print(
                        f"  [ep {ep}/{args.epochs}] step {opt_step:4d}/{total_opt_steps} "
                        f"| loss_avg50={avg50:.5f} | grad_norm={grad_norm:.4f} | lr={log_item['lr']:.2e}",
                        flush=True,
                    )

        if device.type == "cuda":
            torch.cuda.synchronize()
        ep_train_sec = time.perf_counter() - t_ep

        ep_dir = out_dir / f"epoch_{ep}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ep_dir)
        tok.save_pretrained(ep_dir)

        # prod dev (5,000件) でエポック評価
        dev_res, _ = eval_sentiment.evaluate_dataset(
            model, tok, dev_rows, eff_max_len, 32, "3class", [0, 1, 2], device, include_calibration=True
        )
        (ep_dir / "dev_eval.json").write_text(json.dumps(dev_res, ensure_ascii=False, indent=2), encoding="utf-8")

        ov = dev_res["overall"]
        cal = ov["calibration"]
        bin_m = ov["binary_no_neutral"]
        ep_rec = {
            "epoch": ep,
            "opt_step_end": opt_step,
            "train_sec": round(ep_train_sec, 2),
            "samples_per_sec": round(len(train_ds) / max(1e-6, ep_train_sec), 2),
            "train_loss_mean": round(sum(ep_losses) / max(1, len(ep_losses)), 5),
            "train_loss_end_avg50": round(sum(ep_losses[-50:]) / max(1, len(ep_losses[-50:])), 5),
            "dev_3cls_acc": ov["accuracy"],
            "dev_3cls_macro_f1": ov["macro_f1"],
            "dev_per_class_f1": {lbl: ov["per_class"][lbl]["f1"] for lbl in eval_sentiment.CANONICAL_LABELS},
            "dev_pn_flip_count": ov["polarity_flip"]["pn_flip_count"],
            "dev_pn_flip_rate_all": ov["polarity_flip"]["pn_flip_rate_all"],
            "dev_nll": cal["nll"],
            "dev_brier": cal["brier_score"],
            "dev_ece_15bins": cal["ece_15bins"],
            "dev_bin2cls_acc": bin_m["accuracy"],
            "dev_bin2cls_macro_f1": bin_m["macro_f1"],
            "dev_bin2cls_roc_auc": bin_m["roc_auc_pos"],
            "dev_truncated_count": ov["truncated_count"],
            "dir_bytes": tps.dir_size_bytes(ep_dir),
            "step_logs": ep_step_logs,
        }
        epoch_summaries.append(ep_rec)
        print(
            f"[EPOCH {ep} EVAL dev n=5000] loss={ep_rec['train_loss_mean']:.5f} | "
            f"acc={ep_rec['dev_3cls_acc']:.4f}, macro_f1={ep_rec['dev_3cls_macro_f1']:.4f}, "
            f"pn_flip={ep_rec['dev_pn_flip_count']} ({ep_rec['dev_pn_flip_rate_all']:.4f}), "
            f"nll={ep_rec['dev_nll']:.5f}, bin2cls_acc={ep_rec['dev_bin2cls_acc']:.4f}",
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_sec = time.perf_counter() - t_all

    # 事前固定基準で最良エポックを選定: (dev_3cls_macro_f1 最大, -dev_pn_flip_rate_all 最大, -dev_nll 最大)
    best_rec = max(
        epoch_summaries,
        key=lambda e: (e["dev_3cls_macro_f1"], -e["dev_pn_flip_rate_all"], -e["dev_nll"]),
    )
    best_ep = best_rec["epoch"]
    best_src = out_dir / f"epoch_{best_ep}"
    best_dst = out_dir / "best"
    if best_dst.exists():
        shutil.rmtree(best_dst)
    shutil.copytree(best_src, best_dst)

    st_path = best_dst / "model.safetensors"
    summary = {
        "name": args.name,
        "base_model": args.base_model,
        "num_params": n_params,
        "eff_max_len": eff_max_len,
        "train_rows": len(train_ds),
        "dev_rows": len(dev_rows),
        "train_truncated": train_ds.truncated_count,
        "epochs": args.epochs,
        "per_device_bs": args.per_device_bs,
        "grad_accum": args.grad_accum,
        "effective_bs": args.per_device_bs * args.grad_accum,
        "steps_per_epoch_opt": steps_per_epoch_opt,
        "total_opt_steps": total_opt_steps,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "seed": args.seed,
        "total_train_eval_sec": round(total_sec, 2),
        "has_nan_in_train_log": has_nan,
        "selection_criterion": "(dev_3cls_macro_f1 max, -dev_pn_flip_rate_all max, -dev_nll max)",
        "selected_best_epoch": best_ep,
        "best_dir": str(best_dst),
        "best_dir_bytes": tps.dir_size_bytes(best_dst),
        "best_model_safetensors_bytes": st_path.stat().st_size if st_path.exists() else 0,
        "epochs_history": epoch_summaries,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PROD TRAIN DONE] total_sec={total_sec:.2f}s | selected_best_epoch={best_ep} -> {best_dst}")


if __name__ == "__main__":
    main()
