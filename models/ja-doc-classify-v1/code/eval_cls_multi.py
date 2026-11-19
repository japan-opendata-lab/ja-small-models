#!/usr/bin/env python3
"""第4号 ja-doc-classify-v1 独立評価スクリプト（dev 詳細 / OOD 両対応）。

学習済みの共有エンコーダ＋3ヘッドモデル（model_state.pt）を読み込み、
dev 詳細評価または OOD 評価を実行して JSON を出力する。
※ train_pilot_multi.py と同一の MultiHead・ラベル順序・max_len ロジックを使用する。
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from torch import nn
from transformers import AutoModel, AutoTokenizer

DOMAINS = ["金融", "法務", "医療", "教育", "EC", "雇用", "通信・IT", "行政"]
FORMATS = ["structured", "unstructured"]
HIGH_RISK = {
    "給付申請書", "入院同意書", "口座開設申込書", "住民異動届", "本人確認書類一覧",
    "振込依頼書", "学生記録票", "保護者連絡票", "処方箋", "検査結果報告書",
    "税務申告関連書類", "診療情報提供書", "成績通知書", "診断書", "本人確認台帳",
    "クレジットカード利用明細", "給与連絡票", "本人確認記録", "海外送金依頼書",
    "ローン審査書類", "奨学金申請書", "退職手続書類",
}


def load_rows(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class MultiHead(nn.Module):
    def __init__(self, base: str, n_types: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        h = self.encoder.config.hidden_size
        self.head_domain = nn.Linear(h, len(DOMAINS))
        self.head_type = nn.Linear(h, n_types)
        self.head_format = nn.Linear(h, len(FORMATS))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kw):
        kw_in = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self.encoder.config.model_type == "bert":
            kw_in["token_type_ids"] = token_type_ids
        out = self.encoder(**kw_in)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]
        return {
            "logits_domain": self.head_domain(pooled),
            "logits_type": self.head_type(pooled),
            "logits_format": self.head_format(pooled),
        }


def run_eval(model, tok, rows, types, device, max_len, is_dev: bool):
    t2i = {t: i for i, t in enumerate(types)}
    yt, pt_, yd, pd_, yf, pf_ = [], [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for r in rows:
            enc = tok(r["text"], truncation=True, max_length=max_len, return_tensors="pt").to(device)
            res = model(**enc)
            pt_.append(int(res["logits_type"].argmax(-1)[0]))
            yt.append(t2i[r["document_type"]])
            pd_.append(int(res["logits_domain"].argmax(-1)[0]))
            yd.append(DOMAINS.index(r["domain"]))
            pf_.append(int(res["logits_format"].argmax(-1)[0]))
            if is_dev:
                yf.append(FORMATS.index(r["document_format"]))
    eval_sec = time.time() - t0
    return eval_sec, np.array(yt), np.array(pt_), np.array(yd), np.array(pd_), np.array(yf), np.array(pf_)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", choices=["dev", "ood"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-pilot-json", default="")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    train_rows = load_rows(args.train)
    dev_rows = load_rows(args.dev)
    types = sorted({r["document_type"] for r in train_rows + dev_rows})
    t2i = {t: i for i, t in enumerate(types)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiHead(args.base, len(types))
    max_len = min(1024, getattr(model.encoder.config, "max_position_embeddings", 1024))
    state_dict = torch.load(args.state, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    rows = dev_rows if args.mode == "dev" else load_rows(args.input)
    eval_sec, yt, pt_, yd, pd_, yf, pf_ = run_eval(
        model, tok, rows, types, device, max_len, is_dev=(args.mode == "dev")
    )

    hr_idx = [t2i[t] for t in HIGH_RISK if t in t2i]
    hr_gold = np.isin(yt, hr_idx).astype(int)
    hr_pred = np.isin(pt_, hr_idx).astype(int)
    hr_rec = round(float(recall_score(hr_gold, hr_pred, zero_division=0)), 4) if hr_gold.any() else None

    if args.mode == "dev":
        per_f1 = f1_score(yt, pt_, average=None, labels=list(range(len(types))), zero_division=0)
        worst = sorted(range(len(types)), key=lambda i: per_f1[i])[:5]
        cls_rep = classification_report(
            yt, pt_, labels=list(range(len(types))), target_names=types, output_dict=True, zero_division=0
        )
        per_type = {
            ty: {
                "precision": round(float(cls_rep[ty]["precision"]), 4),
                "recall": round(float(cls_rep[ty]["recall"]), 4),
                "f1": round(float(cls_rep[ty]["f1-score"]), 4),
                "support": int(cls_rep[ty]["support"]),
            }
            for ty in types
        }
        report = {
            "base": args.base,
            "dev_rows": len(rows),
            "eval_sec": round(eval_sec, 2),
            "eval_sec_per_doc": round(eval_sec / len(rows), 4),
            "type_macro_f1": round(float(f1_score(yt, pt_, average="macro", zero_division=0)), 4),
            "type_accuracy": round(float(accuracy_score(yt, pt_)), 4),
            "domain_accuracy": round(float(accuracy_score(yd, pd_)), 4),
            "domain_macro_f1": round(float(f1_score(yd, pd_, average="macro", zero_division=0)), 4),
            "format_accuracy": round(float(accuracy_score(yf, pf_)), 4),
            "high_risk_recall": hr_rec,
            "worst5_types": [{"type": types[i], "f1": round(float(per_f1[i]), 4)} for i in worst],
            "n_types": len(types),
            "state_bytes": Path(args.state).stat().st_size,
            "per_type": per_type,
        }
        if args.verify_pilot_json:
            ref = json.loads(Path(args.verify_pilot_json).read_text(encoding="utf-8"))
            for k in (
                "type_macro_f1",
                "type_accuracy",
                "domain_accuracy",
                "format_accuracy",
                "high_risk_recall",
                "dev_rows",
                "n_types",
                "state_bytes",
            ):
                assert report[k] == ref[k], f"Mismatch on {k}: {report[k]} != {ref[k]}"
    else:
        present_idx = sorted(set(yt.tolist()))
        appeared_idx = sorted(set(yt.tolist()) | set(pt_.tolist()))
        present_set = set(present_idx)
        out_of_25 = Counter(types[p] for p in pt_.tolist() if p not in present_set)

        per_type = {}
        confusions = Counter()
        for g, p in zip(yt.tolist(), pt_.tolist()):
            if g != p:
                confusions[(types[g], types[p])] += 1

        cls_rep = classification_report(
            yt, pt_, labels=present_idx, target_names=[types[i] for i in present_idx],
            output_dict=True, zero_division=0,
        )
        for idx in present_idx:
            ty = types[idx]
            mask = yt == idx
            preds_for_ty = pt_[mask]
            wrong = Counter(types[p] for p in preds_for_ty.tolist() if p != idx)
            sample_row = next(r for r in rows if r["document_type"] == ty)
            per_type[ty] = {
                "domain": sample_row["domain"],
                "risk_tier": sample_row.get("risk_tier", "高機密" if ty in HIGH_RISK else "標準"),
                "support": int(mask.sum()),
                "correct": int((preds_for_ty == idx).sum()),
                "precision": round(float(cls_rep[ty]["precision"]), 4),
                "recall": round(float(cls_rep[ty]["recall"]), 4),
                "f1": round(float(cls_rep[ty]["f1-score"]), 4),
                "misclassified_as": dict(wrong),
            }

        by_style = {}
        for st in sorted({r["style"] for r in rows}):
            m = np.array([r["style"] == st for r in rows])
            by_style[st] = {
                "rows": int(m.sum()),
                "type_accuracy": round(float(accuracy_score(yt[m], pt_[m])), 4),
                "type_macro_f1_present25": round(float(f1_score(
                    yt[m], pt_[m], labels=present_idx, average="macro", zero_division=0
                )), 4),
                "domain_accuracy": round(float(accuracy_score(yd[m], pd_[m])), 4),
            }

        by_exact = {}
        for flag, label in [(True, "contains_exact_type_name"), (False, "no_exact_type_name")]:
            m = np.array([(r["document_type"] in r["text"]) == flag for r in rows])
            by_exact[label] = {
                "rows": int(m.sum()),
                "type_accuracy": round(float(accuracy_score(yt[m], pt_[m])), 4) if m.any() else None,
                "domain_accuracy": round(float(accuracy_score(yd[m], pd_[m])), 4) if m.any() else None,
            }

        by_domain = {}
        for d_idx, dom in enumerate(DOMAINS):
            m = yd == d_idx
            if m.any():
                by_domain[dom] = {
                    "rows": int(m.sum()),
                    "type_accuracy": round(float(accuracy_score(yt[m], pt_[m])), 4),
                    "domain_accuracy": round(float(accuracy_score(yd[m], pd_[m])), 4),
                }

        macro_f1_cov = round(float(f1_score(
            yt, pt_, labels=present_idx, average="macro", zero_division=0
        )), 4)
        out_cov_cnt = int(sum(out_of_25.values()))

        report = {
            "base": args.base,
            "ood_rows": len(rows),
            "covered_types": len(present_idx),
            "eval_sec": round(eval_sec, 2),
            "eval_sec_per_doc": round(eval_sec / len(rows), 4),
            "type_macro_f1_covered": macro_f1_cov,
            "type_macro_f1_present25": macro_f1_cov,
            "type_macro_f1_appeared": round(float(f1_score(
                yt, pt_, labels=appeared_idx, average="macro", zero_division=0
            )), 4),
            "type_accuracy": round(float(accuracy_score(yt, pt_)), 4),
            "out_of_covered_pred_count": out_cov_cnt,
            "out_of_covered_pred_breakdown": dict(out_of_25),
            "out_of_25_pred_count": out_cov_cnt,
            "out_of_25_pred_breakdown": dict(out_of_25),
            "domain_accuracy": round(float(accuracy_score(yd, pd_)), 4),
            "domain_macro_f1": round(float(f1_score(yd, pd_, average="macro", zero_division=0)), 4),
            "high_risk_recall": hr_rec,
            "high_risk_rows": int(hr_gold.sum()),
            "by_style": by_style,
            "by_domain": by_domain,
            "by_exact_name_presence": by_exact,
            "top_confusions": [
                {"gold": g, "pred": p, "count": c} for (g, p), c in confusions.most_common(20)
            ],
            "per_type": per_type,
        }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "per_type"}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
