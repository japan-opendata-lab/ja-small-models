#!/usr/bin/env python3
"""
NLIパイロットデータ・モデル向けのショートカット診断スクリプト。

1. Hypothesis-only 線形プローブ (char n-gram TF-IDF + LogisticRegression):
   - train -> dev
   - train -> human-eval
   - human-eval 5-fold Stratified CV
2. Premise-derangement (前提シャッフル) 評価 on human-eval:
   - 各サンプルの premise を別文書由来の他サンプルの premise と入れ替え (seed=42)
   - 通常評価との Accuracy / Macro-F1 / クラス予測分布 / 予測維持率 (same_pred_rate) を算出
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(".")
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import eval_nli  # noqa: E402


DEFAULT_MODELS = [
    {
        "name": "modernbert-ja-130m",
        "model_path": str(ROOT / "runs" / "pilot-modernbert" / "final"),
        "eval_dir": str(ROOT / "runs" / "pilot-modernbert"),
        "max_len": 1024,
    },
    {
        "name": "bert-base-japanese-v3",
        "model_path": str(ROOT / "runs" / "pilot-bert-v3" / "final"),
        "eval_dir": str(ROOT / "runs" / "pilot-bert-v3"),
        "max_len": 512,
    },
    {
        "name": "mdeberta-2mil7",
        "model_path": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        "eval_dir": str(ROOT / "runs" / "zero-shot-mdeberta-2mil7"),
        "max_len": 512,
    },
    {
        "name": "mdeberta-mnli-xnli",
        "model_path": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "eval_dir": str(ROOT / "runs" / "zero-shot-mdeberta-mnli-xnli"),
        "max_len": 512,
    },
]


def load_jsonl(path: Path) -> List[dict]:
    """JSONLファイルを読み込む。"""
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def summarize_probe_preds(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """線形プローブの予測結果を主要指標辞書へまとめる。"""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0))
    p_arr, r_arr, f_arr, s_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "n": int(len(y_true)),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            lbl: {
                "precision": round(float(p_arr[i]), 4),
                "recall": round(float(r_arr[i]), 4),
                "f1": round(float(f_arr[i]), 4),
                "support": int(s_arr[i]),
            }
            for i, lbl in enumerate(eval_nli.CANONICAL_LABELS)
        },
        "confusion_matrix": cm.tolist(),
    }


def extract_top_features(vec: TfidfVectorizer, clf: LogisticRegression, top_k: int = 15) -> Dict[str, List[dict]]:
    """各クラスの上位正重み文字n-gram特徴を抽出する。"""
    feat_names = np.array(vec.get_feature_names_out())
    out: Dict[str, List[dict]] = {}
    for cls_idx in clf.classes_:
        row_idx = list(clf.classes_).index(cls_idx)
        weights = clf.coef_[row_idx]
        top_ids = np.argsort(weights)[::-1][:top_k]
        lbl = eval_nli.CANONICAL_LABELS[int(cls_idx)]
        out[lbl] = [
            {"ngram": str(feat_names[j]), "weight": round(float(weights[j]), 4)}
            for j in top_ids
        ]
    return out


def run_hypothesis_only_diagnostics(
    train_rows: List[dict],
    dev_rows: List[dict],
    he_rows: List[dict],
) -> dict:
    """hypothesis-only TF-IDF + LogisticRegression の診断を実行する。"""
    vec_cfg = {
        "analyzer": "char_wb",
        "ngram_range": [2, 5],
        "min_df": 3,
        "max_features": 40000,
        "sublinear_tf": True,
    }
    clf_cfg = {"C": 2.0, "max_iter": 1000, "random_state": 42}

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        max_features=40000,
        sublinear_tf=True,
    )
    x_tr = vec.fit_transform([r["hypothesis"] for r in train_rows])
    y_tr = np.array([eval_nli.LABEL2ID[r["label"]] for r in train_rows], dtype=np.int64)

    clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    clf.fit(x_tr, y_tr)

    x_dev = vec.transform([r["hypothesis"] for r in dev_rows])
    y_dev = np.array([eval_nli.LABEL2ID[r["label"]] for r in dev_rows], dtype=np.int64)
    dev_summary = summarize_probe_preds(y_dev, clf.predict(x_dev))

    x_he = vec.transform([r["hypothesis"] for r in he_rows])
    y_he = np.array([eval_nli.LABEL2ID[r["label"]] for r in he_rows], dtype=np.int64)
    he_from_train_summary = summarize_probe_preds(y_he, clf.predict(x_he))

    # human-eval 内での 5-fold Stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    he_hyps = [r["hypothesis"] for r in he_rows]
    oof_preds = np.zeros(len(he_rows), dtype=np.int64)
    fold_accs: List[float] = []
    fold_f1s: List[float] = []
    for tr_idx, va_idx in skf.split(he_hyps, y_he):
        cv_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=2,
            max_features=20000,
            sublinear_tf=True,
        )
        x_cv_tr = cv_vec.fit_transform([he_hyps[i] for i in tr_idx])
        x_cv_va = cv_vec.transform([he_hyps[i] for i in va_idx])
        cv_clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
        cv_clf.fit(x_cv_tr, y_he[tr_idx])
        preds_va = cv_clf.predict(x_cv_va)
        oof_preds[va_idx] = preds_va
        fold_accs.append(float(accuracy_score(y_he[va_idx], preds_va)))
        fold_f1s.append(float(f1_score(y_he[va_idx], preds_va, labels=[0, 1, 2], average="macro", zero_division=0)))

    he_cv_summary = summarize_probe_preds(y_he, oof_preds)
    he_cv_summary["fold_acc_mean"] = round(float(np.mean(fold_accs)), 4)
    he_cv_summary["fold_acc_std"] = round(float(np.std(fold_accs)), 4)
    he_cv_summary["fold_f1_mean"] = round(float(np.mean(fold_f1s)), 4)
    he_cv_summary["fold_f1_std"] = round(float(np.std(fold_f1s)), 4)

    return {
        "config": {"vectorizer": vec_cfg, "classifier": clf_cfg},
        "train_to_dev": dev_summary,
        "train_to_human_eval": he_from_train_summary,
        "human_eval_5fold_cv": he_cv_summary,
        "train_top_positive_features": extract_top_features(vec, clf, top_k=15),
    }


def make_premise_derangement(rows: List[dict], seed: int = 42) -> List[dict]:
    """全サンプルが自身の前提および同一文書の前提を受け取らない完全錯乱配列を生成する。"""
    n = len(rows)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    def ok(pos: int, donor: int) -> bool:
        return (
            donor != pos
            and rows[donor]["premise"] != rows[pos]["premise"]
            and rows[donor].get("doc_id") != rows[pos].get("doc_id")
        )

    for _ in range(10):
        changed = False
        for i in range(n):
            if ok(i, indices[i]):
                continue
            candidates = [j for j in range(n) if j != i and ok(i, indices[j]) and ok(j, indices[i])]
            if not candidates:
                raise RuntimeError(f"Swap-repair failed at index {i}")
            j = rng.choice(candidates)
            indices[i], indices[j] = indices[j], indices[i]
            changed = True
        if not changed:
            break

    assert all(ok(i, indices[i]) for i in range(n)), "Premise derangement invariant failed"
    shuffled = []
    for i, row in enumerate(rows):
        new_row = dict(row)
        new_row["premise"] = rows[indices[i]]["premise"]
        new_row["shuffled_from_id"] = rows[indices[i]]["id"]
        shuffled.append(new_row)
    return shuffled


def count_label_dist(preds: List[str]) -> Dict[str, int]:
    """ラベル予測分布を正規順で集計する。"""
    cnt = Counter(preds)
    return {lbl: int(cnt.get(lbl, 0)) for lbl in eval_nli.CANONICAL_LABELS}


def evaluate_premise_shuffle(
    model_specs: List[dict],
    he_rows: List[dict],
    shuffled_rows: List[dict],
    batch_size: int = 16,
) -> Dict[str, dict]:
    """全対象モデルで前提シャッフル時の劣化幅と予測維持率を実測する。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true = np.array([eval_nli.LABEL2ID[r["label"]] for r in he_rows], dtype=np.int64)
    results: Dict[str, dict] = {}

    for spec in model_specs:
        name = spec["name"]
        eval_dir = Path(spec["eval_dir"])
        eval_res_path = eval_dir / "eval_results.json"
        preds_he_path = eval_dir / "preds_human_eval.jsonl"
        if not eval_res_path.exists() or not preds_he_path.exists():
            raise FileNotFoundError(f"Missing eval artifacts for {name} in {eval_dir}")

        normal_res = json.loads(eval_res_path.read_text(encoding="utf-8"))
        normal_overall = normal_res["datasets"]["human_eval"]["overall"]
        normal_preds_rows = load_jsonl(preds_he_path)
        normal_preds = [r["pred"] for r in normal_preds_rows]

        tok = AutoTokenizer.from_pretrained(spec["model_path"])
        model = AutoModelForSequenceClassification.from_pretrained(spec["model_path"]).to(device)
        model_max_pos = getattr(model.config, "max_position_embeddings", None)
        eff_max_len = (
            min(int(spec["max_len"]), int(model_max_pos))
            if (model_max_pos and int(model_max_pos) > 0)
            else int(spec["max_len"])
        )
        remap, _ = eval_nli.resolve_label_remap(model.config)

        probs_shuf, truncated, elapsed = eval_nli.predict_batches(
            model, tok, shuffled_rows, eff_max_len, batch_size, remap, device
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        shuf_overall = eval_nli.compute_metrics_subset(y_true, probs_shuf)
        shuf_overall["eval_sec"] = round(elapsed, 3)
        shuf_overall["truncated_count"] = truncated

        shuf_preds = [eval_nli.CANONICAL_LABELS[int(i)] for i in np.argmax(probs_shuf, axis=1)]
        same_all = sum(1 for a, b in zip(normal_preds, shuf_preds) if a == b)
        same_by_gold: Dict[str, float] = {}
        for lbl in eval_nli.CANONICAL_LABELS:
            idxs = [i for i, r in enumerate(he_rows) if r["label"] == lbl]
            same_cnt = sum(1 for i in idxs if normal_preds[i] == shuf_preds[i])
            same_by_gold[lbl] = round(same_cnt / max(1, len(idxs)), 4)

        results[name] = {
            "max_len": eff_max_len,
            "normal": {
                "accuracy": normal_overall["accuracy"],
                "macro_f1": normal_overall["macro_f1"],
                "pred_dist": count_label_dist(normal_preds),
            },
            "shuffled": {
                "accuracy": shuf_overall["accuracy"],
                "macro_f1": shuf_overall["macro_f1"],
                "pred_dist": count_label_dist(shuf_preds),
                "per_class": shuf_overall["per_class"],
                "polarity_flip": shuf_overall["polarity_flip"],
                "truncated_count": truncated,
            },
            "delta_drop": {
                "accuracy_drop": round(float(normal_overall["accuracy"] - shuf_overall["accuracy"]), 4),
                "macro_f1_drop": round(float(normal_overall["macro_f1"] - shuf_overall["macro_f1"]), 4),
            },
            "same_pred_rate_overall": round(same_all / max(1, len(he_rows)), 4),
            "same_pred_rate_by_gold": same_by_gold,
        }
        print(
            f"[SHUFFLE {name}] norm_acc={normal_overall['accuracy']:.4f} -> "
            f"shuf_acc={shuf_overall['accuracy']:.4f} "
            f"(drop={results[name]['delta_drop']['accuracy_drop']:.4f}) | "
            f"same_pred={results[name]['same_pred_rate_overall']:.4f} | "
            f"shuf_dist={results[name]['shuffled']['pred_dist']}"
        )

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(ROOT / "runs" / "shortcut_diagnostics.json"),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    train_rows = load_jsonl(ROOT / "data" / "pilot" / "train.jsonl")
    dev_rows = load_jsonl(ROOT / "data" / "pilot" / "dev.jsonl")
    he_rows = load_jsonl(ROOT / "data" / "eval" / "human-eval.jsonl")

    hyp_only = run_hypothesis_only_diagnostics(train_rows, dev_rows, he_rows)
    print(
        f"[HYP-ONLY] train->dev acc={hyp_only['train_to_dev']['accuracy']:.4f} | "
        f"train->he acc={hyp_only['train_to_human_eval']['accuracy']:.4f} | "
        f"he_5cv acc={hyp_only['human_eval_5fold_cv']['accuracy']:.4f}"
    )

    shuffled_rows = make_premise_derangement(he_rows, seed=args.seed)
    premise_shuffle = evaluate_premise_shuffle(DEFAULT_MODELS, he_rows, shuffled_rows, batch_size=args.batch_size)

    payload = {
        "seed": args.seed,
        "human_eval_n": len(he_rows),
        "hypothesis_only": hyp_only,
        "premise_shuffle_on_human_eval": premise_shuffle,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Saved shortcut diagnostics -> {out_path}")


if __name__ == "__main__":
    main()
