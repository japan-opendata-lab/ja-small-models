#!/usr/bin/env python3
"""第11号 ja-nli-v1 本番モデル・ゼロショット2種向けショートカット診断再実施スクリプト。

1. Hypothesis-only 線形プローブ (char_wb 2-5 TF-IDF + LogisticRegression):
   - prod train -> prod dev (5,000)
   - prod train -> prod sealed-test (5,000)
   - prod train -> human-eval (600)
   - prod dev 5-fold Stratified CV
   - human-eval 5-fold Stratified CV
2. Premise-derangement (前提シャッフル) 評価:
   - human-eval (600) および prod sealed-test (5,000) 上で、各行へ別文書の premise を割当 (seed=42)
   - 対象は本番許可3モデル (ja-nli-v1-prod / mdeberta-2mil7 / mdeberta-mnli-xnli) のみ
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(".")
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import diagnose_shortcuts as ds  # noqa: E402
import eval_nli  # noqa: E402

PROD_MODELS = [
    {
        "name": "ja-nli-v1-prod",
        "model_path": str(ROOT / "runs" / "prod-modernbert" / "best"),
        "eval_dir": str(ROOT / "runs" / "prod-modernbert"),
        "max_len": 1024,
    },
    {
        "name": "mdeberta-2mil7",
        "model_path": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        "eval_dir": str(ROOT / "runs" / "prod-zero-shot-mdeberta-2mil7"),
        "max_len": 512,
    },
    {
        "name": "mdeberta-mnli-xnli",
        "model_path": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "eval_dir": str(ROOT / "runs" / "prod-zero-shot-mdeberta-mnli-xnli"),
        "max_len": 512,
    },
]


def load_jsonl_stream(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def run_cv_probe(rows: List[dict], min_df: int = 2, max_features: int = 20000) -> dict:
    y = np.array([eval_nli.LABEL2ID[r["label"]] for r in rows], dtype=np.int64)
    hyps = [r["hypothesis"] for r in rows]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(rows), dtype=np.int64)
    fold_accs: List[float] = []
    fold_f1s: List[float] = []
    for tr_i, va_i in skf.split(hyps, y):
        vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=min_df, max_features=max_features, sublinear_tf=True
        )
        x_tr = vec.fit_transform([hyps[i] for i in tr_i])
        x_va = vec.transform([hyps[i] for i in va_i])
        clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
        clf.fit(x_tr, y[tr_i])
        preds = clf.predict(x_va)
        oof[va_i] = preds
        fold_accs.append(float(accuracy_score(y[va_i], preds)))
        fold_f1s.append(float(f1_score(y[va_i], preds, labels=[0, 1, 2], average="macro", zero_division=0)))
    res = ds.summarize_probe_preds(y, oof)
    res["fold_acc_mean"] = round(float(np.mean(fold_accs)), 4)
    res["fold_acc_std"] = round(float(np.std(fold_accs)), 4)
    res["fold_f1_mean"] = round(float(np.mean(fold_f1s)), 4)
    res["fold_f1_std"] = round(float(np.std(fold_f1s)), 4)
    return res


def run_hypothesis_only_prod(
    train_rows: List[dict], dev_rows: List[dict], sealed_rows: List[dict], he_rows: List[dict]
) -> dict:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=40000, sublinear_tf=True)
    x_tr = vec.fit_transform([r["hypothesis"] for r in train_rows])
    y_tr = np.array([eval_nli.LABEL2ID[r["label"]] for r in train_rows], dtype=np.int64)
    clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    clf.fit(x_tr, y_tr)

    out = {
        "config": {
            "vectorizer": {"analyzer": "char_wb", "ngram_range": [2, 5], "min_df": 3, "max_features": 40000, "sublinear_tf": True},
            "classifier": {"C": 2.0, "max_iter": 1000, "random_state": 42},
        },
        "prod_train_top_positive_features": ds.extract_top_features(vec, clf, top_k=15),
        "prod_dev_5fold_cv": run_cv_probe(dev_rows, min_df=3, max_features=30000),
        "human_eval_5fold_cv": run_cv_probe(he_rows, min_df=2, max_features=20000),
    }

    for key, rows in [
        ("prod_train_to_dev", dev_rows),
        ("prod_train_to_sealed_test", sealed_rows),
        ("prod_train_to_human_eval", he_rows),
    ]:
        x_ev = vec.transform([r["hypothesis"] for r in rows])
        y_ev = np.array([eval_nli.LABEL2ID[r["label"]] for r in rows], dtype=np.int64)
        out[key] = ds.summarize_probe_preds(y_ev, clf.predict(x_ev))
    return out


def evaluate_shuffle_for_dataset(
    model_specs: List[dict],
    rows: List[dict],
    shuffled_rows: List[dict],
    ds_key: str,
    batch_size: int = 16,
) -> Dict[str, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true = np.array([eval_nli.LABEL2ID[r["label"]] for r in rows], dtype=np.int64)
    results: Dict[str, dict] = {}

    for spec in model_specs:
        name = spec["name"]
        if "bert-base-japanese" in (name + spec["model_path"]).lower():
            raise ValueError("SECURITY GUARD: forbidden model in Stage 2 final eval")
        eval_dir = Path(spec["eval_dir"])
        norm_res = json.loads((eval_dir / "eval_final_results.json").read_text(encoding="utf-8"))
        normal_ov = norm_res["datasets"][ds_key]["overall"]
        norm_preds = [r["pred"] for r in load_jsonl_stream(eval_dir / f"preds_{ds_key}.jsonl")]

        tok = AutoTokenizer.from_pretrained(spec["model_path"])
        try:
            model = AutoModelForSequenceClassification.from_pretrained(spec["model_path"], reference_compile=False).to(device)
        except TypeError:
            model = AutoModelForSequenceClassification.from_pretrained(spec["model_path"]).to(device)
        mpos = getattr(model.config, "max_position_embeddings", None)
        eff_max_len = min(int(spec["max_len"]), int(mpos)) if (mpos and int(mpos) > 0) else int(spec["max_len"])
        remap, _ = eval_nli.resolve_label_remap(model.config)

        probs_shuf, truncated, elapsed = eval_nli.predict_batches(
            model, tok, shuffled_rows, eff_max_len, batch_size, remap, device
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        shuf_ov = eval_nli.compute_metrics_subset(y_true, probs_shuf)
        shuf_ov["eval_sec"] = round(elapsed, 3)
        shuf_ov["truncated_count"] = truncated

        shuf_preds = [eval_nli.CANONICAL_LABELS[int(i)] for i in np.argmax(probs_shuf, axis=1)]
        same_all = sum(1 for a, b in zip(norm_preds, shuf_preds) if a == b)
        same_by_gold = {}
        for lbl in eval_nli.CANONICAL_LABELS:
            idxs = [i for i, r in enumerate(rows) if r["label"] == lbl]
            same_by_gold[lbl] = round(sum(1 for i in idxs if norm_preds[i] == shuf_preds[i]) / max(1, len(idxs)), 4)

        results[name] = {
            "max_len": eff_max_len,
            "normal": {
                "accuracy": normal_ov["accuracy"],
                "macro_f1": normal_ov["macro_f1"],
                "pred_dist": ds.count_label_dist(norm_preds),
            },
            "shuffled": {
                "accuracy": shuf_ov["accuracy"],
                "macro_f1": shuf_ov["macro_f1"],
                "pred_dist": ds.count_label_dist(shuf_preds),
                "per_class": shuf_ov["per_class"],
                "polarity_flip": shuf_ov["polarity_flip"],
                "truncated_count": truncated,
            },
            "delta_drop": {
                "accuracy_drop": round(float(normal_ov["accuracy"] - shuf_ov["accuracy"]), 4),
                "macro_f1_drop": round(float(normal_ov["macro_f1"] - shuf_ov["macro_f1"]), 4),
            },
            "same_pred_rate_overall": round(same_all / max(1, len(rows)), 4),
            "same_pred_rate_by_gold": same_by_gold,
        }
        print(
            f"[SHUFFLE {ds_key} | {name}] {normal_ov['accuracy']:.4f} -> {shuf_ov['accuracy']:.4f} "
            f"(drop={results[name]['delta_drop']['accuracy_drop']:.4f}) | same={results[name]['same_pred_rate_overall']:.4f} | "
            f"dist={results[name]['shuffled']['pred_dist']}"
        )
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "prod_shortcut_diagnostics.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    train_rows = load_jsonl_stream(ROOT / "data" / "prod" / "train.jsonl")
    dev_rows = load_jsonl_stream(ROOT / "data" / "prod" / "dev.jsonl")
    sealed_rows = load_jsonl_stream(ROOT / "data" / "prod" / "sealed-test.jsonl")
    he_rows = load_jsonl_stream(ROOT / "data" / "eval" / "human-eval.jsonl")

    hyp_only = run_hypothesis_only_prod(train_rows, dev_rows, sealed_rows, he_rows)
    print(
        f"[HYP-ONLY PROD] train->dev={hyp_only['prod_train_to_dev']['accuracy']:.4f} | "
        f"train->sealed={hyp_only['prod_train_to_sealed_test']['accuracy']:.4f} | "
        f"train->he={hyp_only['prod_train_to_human_eval']['accuracy']:.4f} | "
        f"dev_5cv={hyp_only['prod_dev_5fold_cv']['accuracy']:.4f} | "
        f"he_5cv={hyp_only['human_eval_5fold_cv']['accuracy']:.4f}"
    )

    he_shuf = ds.make_premise_derangement(he_rows, seed=args.seed)
    st_shuf = ds.make_premise_derangement(sealed_rows, seed=args.seed)

    payload = {
        "seed": args.seed,
        "hypothesis_only": hyp_only,
        "premise_shuffle_on_human_eval": evaluate_shuffle_for_dataset(
            PROD_MODELS, he_rows, he_shuf, "human_eval", batch_size=args.batch_size
        ),
        "premise_shuffle_on_sealed_test": evaluate_shuffle_for_dataset(
            PROD_MODELS, sealed_rows, st_shuf, "sealed_test", batch_size=args.batch_size
        ),
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Saved prod shortcut diagnostics -> {out_p}")


if __name__ == "__main__":
    main()
