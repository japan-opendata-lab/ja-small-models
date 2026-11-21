#!/usr/bin/env python
"""第11号 ja-nli-v1 ベース候補2種 同一条件1エポック パイロット学習スクリプト。

対象候補:
1. sbintuitions/modernbert-ja-130m (MIT, max_position_embeddings=8192 -> max_len=1024)
2. tohoku-nlp/bert-base-japanese-v3 (Apache-2.0, max_position_embeddings=512 -> max_len=512)

同一ハイパーパラメータ:
- epochs=1.0, 実効 batch_size=16 (per_device_batch_size=4, grad_accum=4 -> 1,250 steps)
- lr=2e-5, AdamW, linear warmup=0.05, weight_decay=0.01, max_grad_norm=1.0, bf16=True, seed=42
- truncation="only_first" (前提末尾優先、仮説完全保持)
※ sealed-test.jsonl には一切触れない。
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import os
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

ROOT = Path(__file__).resolve().parent.parent
CANONICAL_LABELS = ["entailment", "neutral", "contradiction"]
LABEL2ID = {l: i for i, l in enumerate(CANONICAL_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(CANONICAL_LABELS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NliPairDataset(Dataset):
    def __init__(self, rows: List[dict], tokenizer, max_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        # トークン長統計と切り詰め件数を計測
        self.truncated_count = 0
        self.encodings = []
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
        # 8の倍数パディングで Tensor Core 効率化
        if max_l % 8 != 0:
            max_l = ((max_l // 8) + 1) * 8
        keys = [k for k in batch[0].keys() if k != "labels"]
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            pad_val = pad_token_id if k == "input_ids" else 0
            padded = [x[k] + [pad_val] * (max_l - len(x[k])) for x in batch]
            out[k] = torch.tensor(padded, dtype=torch.long)
        out["labels"] = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
        return out

    return collate


def dir_size_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-file", default=str(ROOT / "data" / "pilot" / "train.jsonl"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-len-cap", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per-device-bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    final_dir = out_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

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
    eff_max_len = min(args.max_len_cap, int(model_max_pos)) if (model_max_pos and int(model_max_pos) > 0) else args.max_len_cap
    n_params = sum(p.numel() for p in model.parameters())

    train_rows = [
        json.loads(ln)
        for ln in Path(args.train_file).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    train_ds = NliPairDataset(train_rows, tok, eff_max_len)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    loader = DataLoader(
        train_ds,
        batch_size=args.per_device_bs,
        shuffle=True,
        collate_fn=make_collate_fn(pad_id),
        generator=torch.Generator().manual_seed(args.seed),
    )

    total_micro_steps = math.ceil(len(loader) * args.epochs)
    total_opt_steps = math.ceil(total_micro_steps / args.grad_accum)
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
        f"[TRAIN {args.name}] base={args.base_model}, params={n_params:,}, "
        f"eff_max_len={eff_max_len}, train_trunc={train_ds.truncated_count}/{len(train_ds)}, "
        f"opt_steps={total_opt_steps}, warmup={warmup_steps}, bf16={use_bf16}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses_all: List[float] = []
    step_logs: List[dict] = []
    has_nan = False

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    opt_step = 0
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
            raise RuntimeError(f"NaN/Inf loss encountered at micro_step={micro_step}")
        loss.backward()
        acc_loss += loss_val

        if micro_step % args.grad_accum == 0 or micro_step == len(loader):
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item())
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                has_nan = True
                raise RuntimeError(f"NaN/Inf grad_norm at opt_step={opt_step+1}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            mean_step_loss = acc_loss / args.grad_accum
            losses_all.append(mean_step_loss)
            acc_loss = 0.0

            if opt_step % 50 == 0 or opt_step == total_opt_steps:
                avg50 = sum(losses_all[-50:]) / len(losses_all[-50:])
                entry = {
                    "step": opt_step,
                    "loss": round(mean_step_loss, 5),
                    "loss_avg50": round(avg50, 5),
                    "grad_norm": round(grad_norm, 4),
                    "lr": round(scheduler.get_last_lr()[0], 8),
                }
                step_logs.append(entry)
                print(
                    f"  step {opt_step:4d}/{total_opt_steps} | loss_avg50={entry['loss_avg50']:.5f} "
                    f"| grad_norm={entry['grad_norm']:.4f}"
                )

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0

    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    final_bytes = dir_size_bytes(final_dir)
    safetensors_path = final_dir / "model.safetensors"
    st_bytes = safetensors_path.stat().st_size if safetensors_path.exists() else 0

    train_summary = {
        "name": args.name,
        "base_model": args.base_model,
        "num_params": n_params,
        "eff_max_len": eff_max_len,
        "train_rows": len(train_ds),
        "train_truncated": train_ds.truncated_count,
        "epochs": args.epochs,
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
        "train_sec": round(train_sec, 2),
        "samples_per_sec": round(len(train_ds) / max(1e-6, train_sec), 2),
        "train_loss_mean": round(sum(losses_all) / max(1, len(losses_all)), 5),
        "has_nan_in_train_log": has_nan,
        "final_dir_bytes": final_bytes,
        "model_safetensors_bytes": st_bytes,
        "step_logs": step_logs,
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(train_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[TRAIN DONE {args.name}] {train_sec:.2f}s ({train_summary['samples_per_sec']} samples/s), mean_loss={train_summary['train_loss_mean']:.5f}")

    # 続けて human_eval と dev を評価 (sealed-test は絶対に触らない)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    eval_nli.run_eval_cli(
        str(final_dir),
        args.name,
        ["human_eval", "dev"],
        out_dir,
        max_len=eff_max_len,
        batch_size=16,
    )


if __name__ == "__main__":
    main()
