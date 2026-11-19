#!/usr/bin/env python3
"""本番データによる日本語プロンプトインジェクション検知ベース選定学習・評価スクリプト。

仕様 (SPEC-v1 / ベース選定指示) に基づき、train.jsonl 全件で学習し、
dev.jsonl で以下を一括計測して eval_select_cls.json に保存する。
- 閾値 0.5 固定での attack_recall / benign_fpr / precision / f1_binary
- benign_fpr <= 0.01 (1% 以下) 制約下での最大 attack_recall とその閾値
- サブカテゴリ別 (攻撃6種・通常3種) の発見率・正答率・誤検出率
- 出典別 (llmail / synth / browsesafe / notinject / xstest) の内訳
- 長文スライス (512文字超) での発見率と誤検出率
- パラメータ数、保存モデルサイズ、学習時間、GPU / VRAM 情報
"""
import argparse
import json
import os
import time
from collections import defaultdict

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


def load_jsonl(path: str):
    """JSONLファイルを読み込む。"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def source_group(src: str) -> str:
    """データ行の source 列を報告用の出典グループ名に集約する。"""
    if src.startswith("llmail"):
        return "llmail"
    if src.startswith("synth"):
        return "synth"
    if src.startswith("browsesafe"):
        return "browsesafe"
    if src == "notinject":
        return "notinject"
    if src.startswith("xstest"):
        return "xstest"
    raise ValueError(f"未知の source 値です: {src}")


class ClsDataset(Dataset):
    """ピークメモリを抑えつつチャンク単位でトークナイズする分類データセット。"""

    def __init__(self, rows, tok, max_len: int, chunk_size: int = 5000):
        self.input_ids = []
        self.attention_mask = []
        self.labels = [int(r["label"]) for r in rows]
        texts = [r["text"] for r in rows]
        for start in range(0, len(texts), chunk_size):
            enc = tok(
                texts[start : start + chunk_size],
                truncation=True,
                max_length=max_len,
            )
            self.input_ids.extend(enc["input_ids"])
            self.attention_mask.extend(enc["attention_mask"])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


def binary_metrics(probs: np.ndarray, gold: np.ndarray, threshold: float) -> dict:
    """指定閾値における混同行列および二値分類指標を計算する。"""
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (gold == 1)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    tn = int(((pred == 0) & (gold == 0)).sum())

    n_attack = tp + fn
    n_benign = fp + tn
    n_total = len(gold)

    attack_recall = (tp / n_attack) if n_attack > 0 else None
    benign_fpr = (fp / n_benign) if n_benign > 0 else None
    benign_accuracy = (tn / n_benign) if n_benign > 0 else None
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else (0.0 if n_attack > 0 else None)

    if attack_recall is not None and precision is not None:
        f1_binary = (
            (2.0 * precision * attack_recall / (precision + attack_recall))
            if (precision + attack_recall) > 0
            else 0.0
        )
    else:
        f1_binary = None

    accuracy = ((tp + tn) / n_total) if n_total > 0 else None

    def _r(val):
        return round(float(val), 6) if val is not None else None

    return {
        "n_total": int(n_total),
        "n_attack": int(n_attack),
        "n_benign": int(n_benign),
        "attack_recall": _r(attack_recall),
        "benign_fpr": _r(benign_fpr),
        "benign_accuracy": _r(benign_accuracy),
        "precision": _r(precision),
        "f1_binary": _r(f1_binary),
        "accuracy": _r(accuracy),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def find_threshold_fpr_1pct(probs: np.ndarray, gold: np.ndarray) -> dict:
    """dev 上で benign_fpr <= 0.01 を満たし attack_recall を最大化する閾値を求める。"""
    benign_scores = probs[gold == 0]
    n_benign = int(len(benign_scores))
    max_fp = int(np.floor(0.01 * n_benign))

    b_desc = np.sort(benign_scores)[::-1]
    cutoff_benign = float(b_desc[max_fp])

    higher = probs[probs > cutoff_benign]
    if len(higher) > 0:
        min_above = float(higher.min())
        t = 0.5 * (cutoff_benign + min_above)
        if int((benign_scores >= t).sum()) > max_fp:
            t = min_above
    else:
        t = float(np.nextafter(cutoff_benign, 2.0))

    actual_fp = int((benign_scores >= t).sum())
    assert (actual_fp / n_benign) <= 0.01, (
        f"FPR <= 0.01 制約違反: actual_fp={actual_fp}, n_benign={n_benign}"
    )

    res = {
        "threshold": float(t),
        "threshold_round6": round(float(t), 6),
        "cutoff_benign": float(cutoff_benign),
        "max_allowed_fp": int(max_fp),
    }
    res.update(binary_metrics(probs, gold, t))
    return res


def dir_total_bytes(path: str) -> int:
    """ディレクトリ配下の全ファイル合計バイト数を返す。"""
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.base)
    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)
    assert len(train_rows) > 0 and len(dev_rows) > 0, "学習または検証データが空です"

    train_ds = ClsDataset(train_rows, tok, args.max_len)
    dev_ds = ClsDataset(dev_rows, tok, args.max_len)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def collate(feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in feats:
            n = maxlen - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * n)
            batch["attention_mask"].append(f["attention_mask"] + [0] * n)
            batch["labels"].append(f["labels"])
        return {k: torch.tensor(v) for k, v in batch.items()}

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=2,
        id2label={0: "benign", 1: "injection"},
        label2id={"benign": 0, "injection": 1},
    )
    num_parameters = int(sum(p.numel() for p in model.parameters()))

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        per_device_eval_batch_size=args.eval_batch,
        learning_rate=args.lr,
        fp16=use_cuda,
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        eval_strategy="no",
        report_to=[],
        disable_tqdm=True,
        seed=args.seed,
        data_seed=args.seed,
        label_smoothing_factor=args.label_smoothing,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collate,
    )

    final_dir = os.path.join(args.out, "final")
    final_safetensors = os.path.join(final_dir, "model.safetensors")
    last_ckpt = (
        get_last_checkpoint(args.out)
        if os.path.isdir(args.out) and not os.path.exists(final_safetensors)
        else None
    )
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)

    # 学習履歴を保存
    with open(os.path.join(args.out, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    # モデルとトークナイザを保存
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)

    # dev.jsonl に対する予測とロジット・確率の保存
    preds = trainer.predict(dev_ds)
    logits = (
        preds.predictions[0]
        if isinstance(preds.predictions, tuple)
        else preds.predictions
    )
    logits_arr = np.asarray(logits, dtype=np.float64)
    probs = torch.softmax(torch.tensor(logits_arr, dtype=torch.float64), dim=-1)[:, 1].numpy()
    gold = np.array([int(r["label"]) for r in dev_rows], dtype=int)

    np.save(os.path.join(args.out, "dev_logits.npy"), logits_arr)
    np.save(os.path.join(args.out, "dev_probs.npy"), probs)

    # 1. 閾値 0.5 固定および FPR <= 1% 動作点の算出
    m_05 = binary_metrics(probs, gold, 0.5)
    m_fpr1 = find_threshold_fpr_1pct(probs, gold)
    t_fpr1 = m_fpr1["threshold"]

    # 2. サブカテゴリ別内訳
    subcat_indices = defaultdict(list)
    for idx, r in enumerate(dev_rows):
        subcat_indices[r.get("subcategory", "?")].append(idx)

    per_subcategory = {}
    for cat, idxs in sorted(subcat_indices.items()):
        idx_arr = np.array(idxs, dtype=int)
        p_sub = probs[idx_arr]
        g_sub = gold[idx_arr]
        lbl_vals = sorted(set(int(x) for x in g_sub))
        lbl = lbl_vals[0] if len(lbl_vals) == 1 else -1

        def _sub_stat(thr: float):
            pred = (p_sub >= thr).astype(int)
            correct = int((pred == g_sub).sum())
            n = int(len(g_sub))
            bm = binary_metrics(p_sub, g_sub, thr)
            bm["correct"] = correct
            bm["label"] = lbl
            return bm

        per_subcategory[cat] = {
            "n": int(len(idx_arr)),
            "label": lbl,
            "at_0_5": _sub_stat(0.5),
            "at_fpr_1pct": _sub_stat(t_fpr1),
        }

    # 3. 出典グループ別・生出典別内訳
    src_grp_indices = defaultdict(list)
    src_raw_indices = defaultdict(list)
    for idx, r in enumerate(dev_rows):
        src = r["source"]
        src_grp_indices[source_group(src)].append(idx)
        src_raw_indices[src].append(idx)

    per_source_group = {}
    for grp, idxs in sorted(src_grp_indices.items()):
        idx_arr = np.array(idxs, dtype=int)
        per_source_group[grp] = {
            "at_0_5": binary_metrics(probs[idx_arr], gold[idx_arr], 0.5),
            "at_fpr_1pct": binary_metrics(probs[idx_arr], gold[idx_arr], t_fpr1),
        }

    per_source_raw = {}
    for src, idxs in sorted(src_raw_indices.items()):
        idx_arr = np.array(idxs, dtype=int)
        per_source_raw[src] = {
            "source_group": source_group(src),
            "at_0_5": binary_metrics(probs[idx_arr], gold[idx_arr], 0.5),
            "at_fpr_1pct": binary_metrics(probs[idx_arr], gold[idx_arr], t_fpr1),
        }

    # 4. 長文スライス (512文字超)
    mask_long = np.array([len(r["text"]) > 512 for r in dev_rows], dtype=bool)
    long_slice = {
        "rule": "len(text) > 512 chars",
        "n_total": int(mask_long.sum()),
        "n_attack": int((gold[mask_long] == 1).sum()),
        "n_benign": int((gold[mask_long] == 0).sum()),
        "at_0_5": binary_metrics(probs[mask_long], gold[mask_long], 0.5),
        "at_fpr_1pct": binary_metrics(probs[mask_long], gold[mask_long], t_fpr1),
    }

    # 5. 高確信誤検出 (通常文で攻撃スコア >= 0.99) の集計
    mask_ben_ge099 = (gold == 0) & (probs >= 0.99)
    subcat_ge099 = defaultdict(int)
    src_grp_ge099 = defaultdict(int)
    for idx in np.where(mask_ben_ge099)[0]:
        subcat_ge099[dev_rows[int(idx)].get("subcategory", "?")] += 1
        src_grp_ge099[source_group(dev_rows[int(idx)]["source"])] += 1

    fp_high_conf_ge_099 = {
        "at_0_5": {
            "fp_total": int(((gold == 0) & (probs >= 0.5)).sum()),
            "fp_prob_ge_099": int(((gold == 0) & (probs >= 0.5) & (probs >= 0.99)).sum()),
        },
        "at_fpr_1pct": {
            "fp_total": int(((gold == 0) & (probs >= t_fpr1)).sum()),
            "fp_prob_ge_099": int(((gold == 0) & (probs >= t_fpr1) & (probs >= 0.99)).sum()),
        },
        "total_benign_prob_ge_099": int(mask_ben_ge099.sum()),
        "by_subcategory_prob_ge_099": dict(sorted(subcat_ge099.items())),
        "by_source_group_prob_ge_099": dict(sorted(src_grp_ge099.items())),
    }

    # ハードウェア・サイズ情報
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    gpu_total_vram_mib = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2)
        if use_cuda
        else 0.0
    )
    peak_vram_mib = (
        round(torch.cuda.max_memory_allocated() / (1024**2), 2) if use_cuda else 0.0
    )
    model_sf_bytes = (
        os.path.getsize(final_safetensors) if os.path.exists(final_safetensors) else 0
    )
    final_bytes = dir_total_bytes(final_dir)
    elapsed_sec = round(time.time() - t0, 2)
    train_metrics = getattr(train_result, "metrics", {}) or {}

    result = {
        "base": args.base,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "batch": args.batch,
        "accum": args.accum,
        "effective_batch": args.batch * args.accum,
        "eval_batch": args.eval_batch,
        "lr": args.lr,
        "seed": args.seed,
        "label_smoothing_factor": args.label_smoothing,
        "fp16": use_cuda,
        "train_n": len(train_rows),
        "dev_n": len(dev_rows),
        "resumed_from_checkpoint": last_ckpt,
        "elapsed_sec": elapsed_sec,
        "train_runtime_sec": round(float(train_metrics.get("train_runtime", 0.0)), 2),
        "train_samples_per_sec": round(
            float(train_metrics.get("train_samples_per_second", 0.0)), 3
        ),
        "train_loss": round(float(train_metrics.get("train_loss", 0.0)), 6),
        "num_parameters": num_parameters,
        "model_safetensors_bytes": int(model_sf_bytes),
        "final_dir_bytes": int(final_bytes),
        "gpu_name": gpu_name,
        "gpu_total_vram_mib": gpu_total_vram_mib,
        "peak_vram_allocated_mib": peak_vram_mib,
        "threshold_0_5": m_05,
        "fpr_1pct_operating_point": m_fpr1,
        "per_subcategory": per_subcategory,
        "per_source_group": per_source_group,
        "per_source_raw": per_source_raw,
        "long_slice_gt512chars": long_slice,
        "fp_high_conf_ge_099": fp_high_conf_ge_099,
    }

    eval_path = os.path.join(args.out, "eval_select_cls.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "base": args.base,
                "elapsed_sec": elapsed_sec,
                "train_runtime_sec": result["train_runtime_sec"],
                "threshold_0_5": m_05,
                "fpr_1pct_operating_point": m_fpr1,
                "long_slice_gt512chars": long_slice,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
