#!/usr/bin/env python3
"""第11号 ja-nli-v1 本番学習スクリプト（sbintuitions/modernbert-ja-130m, 3エポック dev 最良選定）。

各エポック終了時に dev.jsonl (5,000件) を評価し、(macro_f1, -ec_flip_rate_all, rag_pr_auc) の順で
最良チェックポイントを best/ へ保存する。
※ sealed-test.jsonl および human-eval.jsonl へのアクセス・評価は一切禁止。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Dict, List

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_nli  # noqa: E402

CANONICAL_LABELS = eval_nli.CANONICAL_LABELS
LABEL2ID = {l: i for i, l in enumerate(CANONICAL_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(CANONICAL_LABELS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NliPairDataset(Dataset):
    """トークナイズ済み (premise, hypothesis) ペアデータセット。"""

    def __init__(self, rows: List[dict], tokenizer, max_len: int):
        self.truncated_count = 0
        self.encodings: List[dict] = []
        for st in range(0, len(rows), 256):
            b = rows[st : st + 256]
            prems = [r["premise"] for r in b]
            hyps = [r["hypothesis"] for r in b]
            raw_enc = tokenizer(prems, hyps, truncation=False, padding=False)
            for ids in raw_enc["input_ids"]:
                if len(ids) > max_len:
                    self.truncated_count += 1
            enc = tokenizer(
                prems,
                hyps,
                truncation="only_first",
                max_length=max_len,
                padding=False,
            )
            keys = list(enc.keys())
            for i in range(len(b)):
                item = {k: enc[k][i] for k in keys}
                item["labels"] = LABEL2ID[b[i]["label"]]
                self.encodings.append(item)

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict:
        return self.encodings[idx]


def make_collate_fn(pad_token_id: int):
    def collate(batch: List[dict]) -> Dict[str, torch.Tensor]:
        max_l = max(len(x["input_ids"]) for x in batch)
        if max_l % 8 != 0:
            max_l = ((max_l // 8) + 1) * 8
        keys = [k for k in batch[0].keys() if k != "labels"]
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            pad_val = pad_token_id if k == "input_ids" else 0
            out[k] = torch.tensor(
                [x[k] + [pad_val] * (max_l - len(x[k])) for x in batch],
                dtype=torch.long,
            )
        out["labels"] = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
        return out

    return collate


def dir_size_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="sbintuitions/modernbert-ja-130m")
    ap.add_argument("--name", default="ja-nli-v1-prod")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--dev-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-len-cap", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--per-device-bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--eval-bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 学習工程での sealed-test / human-eval 接触を構造的に禁止
    for p_str in (args.train_file, args.dev_file):
        low = Path(p_str).name.lower()
        if "sealed" in low or "human" in low:
            raise ValueError(f"SECURITY GUARD: forbidden dataset in training: {p_str}")

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    best_dir = out_dir / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=3,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            reference_compile=False,
        ).to(device)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=3,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
        ).to(device)

    model_max_pos = getattr(model.config, "max_position_embeddings", None)
    eff_max_len = (
        min(args.max_len_cap, int(model_max_pos))
        if (model_max_pos and int(model_max_pos) > 0)
        else args.max_len_cap
    )
    n_params = sum(p.numel() for p in model.parameters())
    remap, _ = eval_nli.resolve_label_remap(model.config)

    with Path(args.train_file).open("r", encoding="utf-8") as f:
        train_rows = [json.loads(ln) for ln in f if ln.strip()]
    with Path(args.dev_file).open("r", encoding="utf-8") as f:
        dev_rows = [json.loads(ln) for ln in f if ln.strip()]
    print(f"[DATA] train={len(train_rows)}, dev={len(dev_rows)} -> tokenizing...")

    train_ds = NliPairDataset(train_rows, tok, eff_max_len)
    dev_bert_lens = eval_nli.compute_bert_token_lengths(dev_rows)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_ds,
        batch_size=args.per_device_bs,
        shuffle=True,
        collate_fn=make_collate_fn(pad_id),
        generator=gen,
    )

    steps_per_epoch = len(loader)
    total_opt_steps = math.ceil(steps_per_epoch / args.grad_accum) * args.epochs
    warmup_steps = int(total_opt_steps * args.warmup_ratio)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
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

    print(
        f"[TRAIN PROD] base={args.base_model}, params={n_params:,}, eff_max_len={eff_max_len}, "
        f"train_trunc={train_ds.truncated_count}/{len(train_ds)}, bs={args.per_device_bs}x{args.grad_accum}, "
        f"epochs={args.epochs}, opt_steps={total_opt_steps}, warmup={warmup_steps}, bf16={use_bf16}"
    )

    losses_all: List[float] = []
    step_logs: List[dict] = []
    epoch_history: List[dict] = []
    has_nan = False
    best_score = (-1.0, -999.0, -1.0)
    best_epoch = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_total_start = time.perf_counter()
    opt_step = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ep_t0 = time.perf_counter()
        ep_losses: List[float] = []
        micro_step = 0
        acc_loss = 0.0

        for batch in loader:
            micro_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(**batch)
                loss = out.loss / args.grad_accum
            loss_val = float(out.loss.detach().item())
            if math.isnan(loss_val) or math.isinf(loss_val):
                has_nan = True
                raise RuntimeError(f"NaN/Inf loss at epoch={ep}, micro={micro_step}")
            loss.backward()
            acc_loss += loss_val

            if micro_step % args.grad_accum == 0 or micro_step == len(loader):
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item())
                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    has_nan = True
                    raise RuntimeError(f"NaN/Inf grad_norm at epoch={ep}, step={opt_step+1}")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                divisor = args.grad_accum if micro_step % args.grad_accum == 0 else (micro_step % args.grad_accum)
                mean_step_loss = acc_loss / divisor
                losses_all.append(mean_step_loss)
                ep_losses.append(mean_step_loss)
                acc_loss = 0.0

                if opt_step % 250 == 0 or micro_step == len(loader):
                    avg100 = sum(losses_all[-100:]) / len(losses_all[-100:])
                    entry = {
                        "epoch": ep,
                        "step": opt_step,
                        "loss": round(mean_step_loss, 5),
                        "loss_avg100": round(avg100, 5),
                        "grad_norm": round(grad_norm, 4),
                        "lr": round(scheduler.get_last_lr()[0], 8),
                    }
                    step_logs.append(entry)
                    print(
                        f"  ep {ep}/{args.epochs} step {opt_step:5d}/{total_opt_steps} | "
                        f"loss_avg100={entry['loss_avg100']:.5f} | grad_norm={entry['grad_norm']:.4f}"
                    )

        if device.type == "cuda":
            torch.cuda.synchronize()
        ep_train_sec = round(time.perf_counter() - ep_t0, 2)

        # エポック終了時 dev (5,000件) フル評価
        sanity = eval_nli.sanity_check(model, tok, eff_max_len, remap, device)
        sanity_ok = all(x["ok"] for x in sanity)
        dev_res, _ = eval_nli.evaluate_dataset(
            model, tok, dev_rows, eff_max_len, args.eval_bs, remap, device, dev_bert_lens
        )
        ov = dev_res["overall"]
        score = (
            float(ov["macro_f1"]),
            -float(ov["polarity_flip"]["ec_flip_rate_all"]),
            float(ov["rag_binary_entailment"]["pr_auc"] or 0.0),
        )
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = ep
            model.save_pretrained(best_dir)
            tok.save_pretrained(best_dir)

        ep_record = {
            "epoch": ep,
            "train_sec": ep_train_sec,
            "train_loss_mean": round(sum(ep_losses) / max(1, len(ep_losses)), 5),
            "sanity_ok": sanity_ok,
            "sanity_check": sanity,
            "dev_overall": ov,
            "dev_stratified": dev_res["stratified"],
            "is_best_so_far": is_best,
        }
        epoch_history.append(ep_record)
        (out_dir / f"eval_dev_ep{ep}.json").write_text(
            json.dumps(ep_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[EPOCH {ep} DEV] acc={ov['accuracy']:.4f}, macro_f1={ov['macro_f1']:.4f}, "
            f"ec_flip_all={ov['polarity_flip']['ec_flip_rate_all']:.4f}, "
            f"rag_f1={ov['rag_binary_entailment']['f1']:.4f}, "
            f"pr_auc={ov['rag_binary_entailment']['pr_auc']:.4f}, "
            f"sanity_ok={sanity_ok}, is_best={is_best}"
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_sec = round(time.perf_counter() - t_total_start, 2)

    selection = {
        "name": args.name,
        "base_model": args.base_model,
        "selection_criterion": "(dev_macro_f1, -dev_ec_flip_rate_all, dev_rag_pr_auc)",
        "best_epoch": best_epoch,
        "best_dev_overall": epoch_history[best_epoch - 1]["dev_overall"],
        "best_sanity_ok": epoch_history[best_epoch - 1]["sanity_ok"],
        "epochs": [
            {
                "epoch": h["epoch"],
                "train_loss_mean": h["train_loss_mean"],
                "train_sec": h["train_sec"],
                "sanity_ok": h["sanity_ok"],
                "accuracy": h["dev_overall"]["accuracy"],
                "macro_f1": h["dev_overall"]["macro_f1"],
                "entailment_f1": h["dev_overall"]["per_class"]["entailment"]["f1"],
                "neutral_f1": h["dev_overall"]["per_class"]["neutral"]["f1"],
                "contradiction_f1": h["dev_overall"]["per_class"]["contradiction"]["f1"],
                "ec_flip_rate_all": h["dev_overall"]["polarity_flip"]["ec_flip_rate_all"],
                "rag_f1": h["dev_overall"]["rag_binary_entailment"]["f1"],
                "rag_pr_auc": h["dev_overall"]["rag_binary_entailment"]["pr_auc"],
                "rag_roc_auc": h["dev_overall"]["rag_binary_entailment"]["roc_auc"],
            }
            for h in epoch_history
        ],
    }
    (out_dir / "epoch_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    st_path = best_dir / "model.safetensors"
    summary = {
        "name": args.name,
        "base_model": args.base_model,
        "num_params": n_params,
        "eff_max_len": eff_max_len,
        "train_rows": len(train_ds),
        "dev_rows": len(dev_rows),
        "train_truncated": train_ds.truncated_count,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "per_device_bs": args.per_device_bs,
        "grad_accum": args.grad_accum,
        "effective_bs": args.per_device_bs * args.grad_accum,
        "opt_steps": opt_step,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "bf16": use_bf16,
        "seed": args.seed,
        "total_sec": total_sec,
        "train_loss_mean": round(sum(losses_all) / max(1, len(losses_all)), 5),
        "has_nan_in_train_log": has_nan,
        "best_dir_bytes": dir_size_bytes(best_dir),
        "model_safetensors_bytes": st_path.stat().st_size if st_path.exists() else 0,
        "step_logs": step_logs,
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ALL DONE] total={total_sec}s | best_epoch={best_epoch} saved -> {best_dir}")


if __name__ == "__main__":
    main()
