#!/usr/bin/env python3
"""第8号 ja-toxicity-guard-v1 パイロット学習・dev 評価スクリプト（両ベース共通）。

7ラベル多ラベル分類（BCEWithLogitsLoss）でモデルを1エポック学習し、
dev.jsonl 上で以下を算出・出力する:
  - 7ラベルごとの Precision / Recall / F1 / PR-AUC / ROC-AUC（閾値 0.5）
  - 全体 Macro F1 / Micro F1 / Exact Match / Any-Toxic 検知性能
  - 単一有害 vs 多ラベル共起有害サブセット評価
  - 無害文での誤検出率（全体 / 正当な批判＋報道引用 / ハードネガティブ / 5サブカテゴリ別）
  - 参考: 正当な批判＋報道引用 FPR <= 1% / <= 2% 閾値での有害発見率
  - 学習/推論速度・サイズ・dev_predictions.jsonl

注意: 封印テスト（sealed-test.jsonl）は一切入力禁止。
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toxicity_templates as tt  # noqa: E402


class ToxicityDataset(Dataset):
    """7ラベル多ラベル有害検知用データセット。"""

    def __init__(self, path: Path, tok, max_len: int = 256) -> None:
        self.rows: List[dict] = []
        self.raw_rows: List[dict] = []
        self.truncated_count = 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                full_ids = tok(r["text"], truncation=False, add_special_tokens=True)["input_ids"]
                if len(full_ids) > max_len:
                    self.truncated_count += 1
                enc = tok(r["text"], truncation=True, max_length=max_len)
                enc["labels"] = [float(r["labels"][lb]) for lb in tt.LABELS]
                self.rows.append(enc)
                self.raw_rows.append(r)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def collate_fn(batch: List[dict], pad_id: int) -> dict:
    """動的パディング付きミニバッチ生成（多ラベル用に labels は float32）。"""
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, labels = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad_n = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad_n)
        attn.append(b["attention_mask"] + [0] * pad_n)
        labels.append(b["labels"])

    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
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


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def _find_fpr_threshold(neg_scores: np.ndarray, max_fpr: float) -> float:
    """neg_scores 上で (neg_scores >= thr).mean() <= max_fpr を満たす最小閾値付近を返す。"""
    n = len(neg_scores)
    if n == 0:
        return 0.5
    max_fp = int(math.floor(max_fpr * n))
    sorted_desc = np.sort(neg_scores)[::-1]
    if max_fp >= n:
        return 0.0
    cand = float(sorted_desc[max_fp]) + 1e-6
    # 同値境界でわずかに max_fp を超える場合の安全チェック
    while (neg_scores >= cand).sum() > max_fp and cand < 1.0:
        cand += 1e-5
    return min(1.0, cand)


def evaluate_dev(raw_rows: List[dict], probs: np.ndarray) -> dict:
    """dev 全件に対する多ラベル・誤検出率・FPRゲート指標を算出する。"""
    n_rows = len(raw_rows)
    y_true = np.array([[int(r["labels"][lb]) for lb in tt.LABELS] for r in raw_rows], dtype=int)
    y_pred = (probs >= 0.5).astype(int)

    # 1. 7ラベル別指標（閾値 0.5 ＋ 閾値非依存 AUC）
    per_label: Dict[str, dict] = {}
    f1_list = []
    pr_auc_list = []
    for j, lb in enumerate(tt.LABELS):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        m = _prf(tp, fp, fn)
        m["support"] = int(yt.sum())
        if yt.sum() > 0 and (1 - yt).sum() > 0:
            ap = float(average_precision_score(yt, probs[:, j]))
            ra = float(roc_auc_score(yt, probs[:, j]))
        else:
            ap, ra = 0.0, 0.0
        m["pr_auc"] = round(ap, 4)
        m["roc_auc"] = round(ra, 4)
        per_label[lb] = m
        f1_list.append(m["f1"])
        pr_auc_list.append(ap)

    # 2. 全体集約指標
    micro_tp = int(((y_pred == 1) & (y_true == 1)).sum())
    micro_fp = int(((y_pred == 1) & (y_true == 0)).sum())
    micro_fn = int(((y_pred == 0) & (y_true == 1)).sum())
    micro_metrics = _prf(micro_tp, micro_fp, micro_fn)
    macro_f1 = round(float(np.mean(f1_list)), 4)
    macro_pr_auc = round(float(np.mean(pr_auc_list)), 4)
    exact_match_acc = round(float(np.all(y_pred == y_true, axis=1).mean()), 4)

    # Any-Toxic（7ラベルのうち1つ以上陽性か）
    y_any_true = (y_true.sum(axis=1) > 0).astype(int)
    y_any_pred = (y_pred.sum(axis=1) > 0).astype(int)
    any_tp = int(((y_any_pred == 1) & (y_any_true == 1)).sum())
    any_fp = int(((y_any_pred == 1) & (y_any_true == 0)).sum())
    any_fn = int(((y_any_pred == 0) & (y_any_true == 1)).sum())
    any_tn = int(((y_any_pred == 0) & (y_any_true == 0)).sum())
    any_toxic = _prf(any_tp, any_fp, any_fn)
    any_toxic["overall_benign_fpr"] = round(any_fp / max(1, any_fp + any_tn), 4)

    # 単一有害 vs 多ラベル共起有害サブセット
    source_kinds = np.array([r["source_kind"] for r in raw_rows])
    subset_metrics: Dict[str, dict] = {}
    for sk in ("single_toxic", "multi_toxic"):
        mask = source_kinds == sk
        if mask.any():
            sub_tp = int(((y_pred[mask] == 1) & (y_true[mask] == 1)).sum())
            sub_fp = int(((y_pred[mask] == 1) & (y_true[mask] == 0)).sum())
            sub_fn = int(((y_pred[mask] == 0) & (y_true[mask] == 1)).sum())
            sm = _prf(sub_tp, sub_fp, sub_fn)
            sm["count"] = int(mask.sum())
            sm["exact_match_acc"] = round(float(np.all(y_pred[mask] == y_true[mask], axis=1).mean()), 4)
            sm["any_toxic_recall"] = round(float((y_any_pred[mask] == 1).mean()), 4)
            subset_metrics[sk] = sm

    # 3. 無害文での誤検出率（看板指標）
    benign_subtypes = np.array([r.get("benign_subtype") or "" for r in raw_rows])

    def _benign_group_stats(mask: np.ndarray) -> dict:
        cnt = int(mask.sum())
        if cnt == 0:
            return {"count": 0, "fp_docs": 0, "fpr": 0.0, "fp_label_hits": 0, "fp_by_label": {}}
        fp_doc_mask = y_any_pred[mask] == 1
        fp_docs = int(fp_doc_mask.sum())
        fp_by_label = {
            lb: int(y_pred[mask, j].sum())
            for j, lb in enumerate(tt.LABELS)
        }
        return {
            "count": cnt,
            "fp_docs": fp_docs,
            "fpr": round(fp_docs / cnt, 4),
            "fp_label_hits": int(sum(fp_by_label.values())),
            "fp_by_label": fp_by_label,
        }

    benign_mask = y_any_true == 0
    crit_news_mask = np.isin(benign_subtypes, ["criticism", "news_quote"])
    hard_benign_mask = np.isin(benign_subtypes, ["criticism", "news_quote", "debate", "support_legal"])

    benign_fpr = {
        "all_benign": _benign_group_stats(benign_mask),
        "criticism_and_news_quote": _benign_group_stats(crit_news_mask),
        "hard_benign_all": _benign_group_stats(hard_benign_mask),
        "by_subtype": {
            sub: _benign_group_stats(benign_subtypes == sub)
            for sub in tt.BENIGN_SUBTYPES
        },
    }

    # 4. 参考: criticism_and_news_quote 上で FPR <= 1% / <= 2% 閾値での有害検知率
    max_probs = probs.max(axis=1)
    cn_risks = max_probs[crit_news_mask]
    toxic_mask = y_any_true == 1

    def _gate_eval(max_fpr: float) -> dict:
        thr = _find_fpr_threshold(cn_risks, max_fpr)
        cn_fp = int((cn_risks >= thr).sum())
        all_b_fp = int((max_probs[benign_mask] >= thr).sum())
        tox_det = int((max_probs[toxic_mask] >= thr).sum())
        per_lb_rec_any = {}
        per_lb_rec_self = {}
        for j, lb in enumerate(tt.LABELS):
            m_lb = y_true[:, j] == 1
            if m_lb.any():
                per_lb_rec_any[lb] = round(float((max_probs[m_lb] >= thr).mean()), 4)
                per_lb_rec_self[lb] = round(float((probs[m_lb, j] >= thr).mean()), 4)
        return {
            "threshold": round(float(thr), 4),
            "crit_news_fpr": round(cn_fp / max(1, int(crit_news_mask.sum())), 4),
            "crit_news_fp_docs": cn_fp,
            "all_benign_fpr": round(all_b_fp / max(1, int(benign_mask.sum())), 4),
            "any_toxic_recall": round(tox_det / max(1, int(toxic_mask.sum())), 4),
            "per_label_recall_via_max_prob": per_lb_rec_any,
            "per_label_recall_via_own_prob": per_lb_rec_self,
        }

    # 5. 無害文での誤検出上位 15 件サンプル
    benign_indices = np.where(benign_mask)[0]
    benign_sorted = sorted(benign_indices, key=lambda idx: float(max_probs[idx]), reverse=True)
    fp_examples = []
    for idx in benign_sorted[:15]:
        r = raw_rows[int(idx)]
        p_vec = probs[int(idx)]
        pred_lbs = [tt.LABELS[j] for j in range(len(tt.LABELS)) if p_vec[j] >= 0.5]
        fp_examples.append(
            {
                "id": r["id"],
                "benign_subtype": r["benign_subtype"],
                "max_prob": round(float(max_probs[idx]), 4),
                "pred_labels_at_0.5": pred_lbs,
                "probs": {tt.LABELS[j]: round(float(p_vec[j]), 4) for j in range(len(tt.LABELS))},
                "text": r["text"],
            }
        )

    return {
        "dev_rows": n_rows,
        "toxic_rows": int(toxic_mask.sum()),
        "benign_rows": int(benign_mask.sum()),
        "per_label_at_0.5": per_label,
        "macro_f1": macro_f1,
        "macro_pr_auc": macro_pr_auc,
        "micro_at_0.5": micro_metrics,
        "exact_match_accuracy": exact_match_acc,
        "any_toxic_at_0.5": any_toxic,
        "subset_metrics_at_0.5": subset_metrics,
        "benign_fpr_at_0.5": benign_fpr,
        "at_crit_news_fpr_1pct": _gate_eval(0.01),
        "at_crit_news_fpr_2pct": _gate_eval(0.02),
        "benign_fp_examples": fp_examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 封印テスト誤読込防止ガード
    for p_str in (args.train, args.dev):
        if "sealed" in Path(p_str).name.lower() or "sealed" in str(p_str).lower():
            raise RuntimeError(f"封印テストファイルは入力禁止です: {p_str}")

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== パイロット学習開始: base={args.base} ===")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=len(tt.LABELS),
        problem_type="multi_label_classification",
        id2label=tt.ID2LABEL,
        label2id=tt.LABEL2ID,
    )
    max_pos = getattr(model.config, "max_position_embeddings", args.max_len)
    max_len = min(args.max_len, int(max_pos))

    train_ds = ToxicityDataset(Path(args.train), tok, max_len=max_len)
    dev_ds = ToxicityDataset(Path(args.dev), tok, max_len=max_len)
    print(
        f"  データ準備完了: train={len(train_ds)} (trunc={train_ds.truncated_count}), "
        f"dev={len(dev_ds)} (trunc={dev_ds.truncated_count}), max_len={max_len}"
    )

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
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
        for _k, v in entry.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                has_nan = True
        log_lines.append(json.dumps(entry, ensure_ascii=False))
    (out_dir / "train.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    # dev バッチ推論
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    probs_list: List[np.ndarray] = []
    eval_bs = 32
    total_chars = sum(len(r["text"]) for r in dev_ds.raw_rows)
    t_eval0 = time.time()
    with torch.no_grad():
        for start_i in range(0, len(dev_ds), eval_bs):
            batch_items = [
                dev_ds[j] for j in range(start_i, min(start_i + eval_bs, len(dev_ds)))
            ]
            batch_tensors = collate_fn(batch_items, pad_id)
            kw = {
                "input_ids": batch_tensors["input_ids"].to(device),
                "attention_mask": batch_tensors["attention_mask"].to(device),
            }
            if "token_type_ids" in batch_tensors:
                kw["token_type_ids"] = batch_tensors["token_type_ids"].to(device)
            logits = model(**kw).logits
            p = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(p)
    eval_sec = time.time() - t_eval0
    probs = np.vstack(probs_list)

    # dev_predictions.jsonl 出力
    with open(out_dir / "dev_predictions.jsonl", "w", encoding="utf-8") as pf:
        for row, p_vec in zip(dev_ds.raw_rows, probs):
            pred_lbs = [tt.LABELS[j] for j in range(len(tt.LABELS)) if p_vec[j] >= 0.5]
            pf.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "is_toxic": row["is_toxic"],
                        "benign_subtype": row["benign_subtype"],
                        "active_labels": row["active_labels"],
                        "pred_labels": pred_lbs,
                        "max_prob": round(float(p_vec.max()), 5),
                        "probs": {tt.LABELS[j]: round(float(p_vec[j]), 5) for j in range(len(tt.LABELS))},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    eval_metrics = evaluate_dev(dev_ds.raw_rows, probs)

    final_dir_bytes = sum(f.stat().st_size for f in final_dir.rglob("*") if f.is_file())
    st_path = final_dir / "model.safetensors"
    safetensors_bytes = st_path.stat().st_size if st_path.exists() else 0
    model_params = sum(p.numel() for p in model.parameters())

    report = {
        "base": args.base,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.bs,
        "max_len": max_len,
        "fp16": bool(torch.cuda.is_available()),
        "train_rows": len(train_ds),
        "dev_rows": len(dev_ds),
        "train_truncated": train_ds.truncated_count,
        "dev_truncated": dev_ds.truncated_count,
        "train_sec": round(train_sec, 2),
        "train_loss": round(float(train_res.training_loss), 5),
        "has_nan_in_train_log": has_nan,
        "eval_sec": round(eval_sec, 3),
        "eval_chars": total_chars,
        "eval_chars_per_sec": round(total_chars / eval_sec, 1) if eval_sec > 0 else 0.0,
        "eval_ms_per_doc": round(eval_sec * 1000.0 / len(dev_ds), 3),
        "model_params": int(model_params),
        "final_dir_bytes": int(final_dir_bytes),
        "model_safetensors_bytes": int(safetensors_bytes),
        **eval_metrics,
    }

    (out_dir / "eval_pilot.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "base": report["base"],
        "train_sec": report["train_sec"],
        "train_loss": report["train_loss"],
        "has_nan": report["has_nan_in_train_log"],
        "macro_f1": report["macro_f1"],
        "micro_f1": report["micro_at_0.5"]["f1"],
        "crit_news_fpr": report["benign_fpr_at_0.5"]["criticism_and_news_quote"]["fpr"],
        "all_benign_fpr": report["benign_fpr_at_0.5"]["all_benign"]["fpr"],
        "per_label_f1": {k: v["f1"] for k, v in report["per_label_at_0.5"].items()},
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
