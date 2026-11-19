#!/usr/bin/env python
"""第6号 ja-error-detect-v1 本番学習スクリプト（tohoku-nlp/bert-base-japanese-v3 専用）。

本番 train.jsonl (100,000件) で bert-base-japanese-v3 を 2 エポック学習し、
最終モデル・トークナイザ・学習ログ・学習メタ情報を保存する。
封印テスト（sealed-test.jsonl）には一切アクセスしない。

使い方:
  python code/train_prod_err.py --train data/prod/train.jsonl \
      --dev data/prod/dev.jsonl --out runs/prod-bert-v3
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offsets_util as ou  # noqa: E402


class ProdErrDataset(Dataset):
    """本番学習用トークン分類データセット（省メモリ版）。"""

    def __init__(
        self,
        path: Path,
        tok,
        max_len: int = 256,
        verify_n: int = 200,
    ) -> None:
        self.rows: List[dict] = []
        self.truncated_count = 0

        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                r = json.loads(line)
                enc, offsets = ou.encode_with_offsets(tok, r["text"], max_len=max_len)
                if idx < verify_n:
                    ref = tok(r["text"], truncation=True, max_length=max_len)
                    assert enc["input_ids"] == ref["input_ids"], (
                        f"input_ids 不一致 (row={idx}, id={r.get('id')}): "
                        f"{enc['input_ids'][:20]} != {ref['input_ids'][:20]}"
                    )
                non_empty_ends = [b for a, b in offsets if b > a]
                if r["text"] and (
                    not non_empty_ends or max(non_empty_ends) < len(r["text"].rstrip())
                ):
                    self.truncated_count += 1

                labels = ou.align_spans_to_tokens(r["text"], offsets, r["spans"])
                enc["labels"] = labels
                self.rows.append(enc)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def collate_fn(batch: List[dict], pad_id: int) -> dict:
    """動的パディング付きミニバッチ生成。"""
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, labels = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad_n = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad_n)
        attn.append(b["attention_mask"] + [0] * pad_n)
        labels.append(b["labels"] + [-100] * pad_n)
    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = torch.tensor(
            [
                b["token_type_ids"] + [0] * (maxlen - len(b["token_type_ids"]))
                for b in batch
            ],
            dtype=torch.long,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="tohoku-nlp/bert-base-japanese-v3")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", default="runs/prod-bert-v3")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 封印テスト誤読込防止ガード
    for p_str in (args.train, args.dev):
        if "sealed" in Path(p_str).name.lower():
            raise RuntimeError(f"封印テストファイルは学習入力に禁止です: {p_str}")

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 本番学習開始: base={args.base} ===")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base,
        num_labels=len(ou.LABELS),
        id2label=ou.ID2LABEL,
        label2id=ou.LABEL2ID,
    )
    assert model.config.model_type == "bert", (
        f"勝者は bert-v3 (model_type=bert) のみ許可: {model.config.model_type}"
    )

    max_pos = getattr(model.config, "max_position_embeddings", args.max_len)
    max_len = min(args.max_len, int(max_pos))

    t_prep = time.time()
    train_ds = ProdErrDataset(Path(args.train), tok, max_len=max_len, verify_n=200)
    # dev は学習へは渡さず、オフセット整列の全件検証と切り詰め数記録のみ実施
    dev_ds = ProdErrDataset(Path(args.dev), tok, max_len=max_len, verify_n=999999)
    print(
        f"  データ準備完了 ({time.time() - t_prep:.1f}s): "
        f"train={len(train_ds)} (trunc={train_ds.truncated_count}), "
        f"dev検証={len(dev_ds)} (trunc={dev_ds.truncated_count}), max_len={max_len}"
    )

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=lambda b: collate_fn(b, pad_id),
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

    final_dir_bytes = sum(f.stat().st_size for f in final_dir.rglob("*") if f.is_file())
    st_path = final_dir / "model.safetensors"
    safetensors_bytes = st_path.stat().st_size if st_path.exists() else 0
    model_params = sum(p.numel() for p in model.parameters())

    meta = {
        "base": args.base,
        "model_type": model.config.model_type,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.bs,
        "max_len": max_len,
        "fp16": bool(torch.cuda.is_available()),
        "train_rows": len(train_ds),
        "dev_rows_verified": len(dev_ds),
        "train_truncated": train_ds.truncated_count,
        "dev_truncated": dev_ds.truncated_count,
        "train_sec": round(train_sec, 2),
        "train_loss": round(float(train_res.training_loss), 5),
        "has_nan_in_train_log": has_nan,
        "model_params": int(model_params),
        "final_dir_bytes": int(final_dir_bytes),
        "model_safetensors_bytes": int(safetensors_bytes),
    }
    (out_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
