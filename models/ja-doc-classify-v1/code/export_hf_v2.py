#!/usr/bin/env python3
"""第4号 ja-doc-classify-v1: 学習済み v2 モデルを Hugging Face 形式で書き出し、確定閾値を同梱するスクリプト。

- 元の学習済み重み (runs/prod-modernbert-v2/model_state.pt) および runs/prod-modernbert-v2/final/ は読み取り専用で不変とする。
- 再学習は一切行わない。
- 封印テスト (data/pilot/sealed-test.jsonl) は一切参照しない。
- 確定既定値:
  - domain_reject_threshold: 0.95
  - type_reject_threshold: 0.80
  - high_risk_prob_sum_threshold: 0.12
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from transformers import AutoConfig, AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_cls_multi import DOMAINS, FORMATS, HIGH_RISK, MultiHead  # noqa: E402

DEFAULT_THRESHOLDS = {
    "domain_reject_threshold": 0.95,
    "type_reject_threshold": 0.80,
    "high_risk_prob_sum_threshold": 0.12,
}

CONFIGURATION_PY = '''"""ja-doc-classify-v1 カスタム Hugging Face 設定クラス。"""
from transformers import ModernBertConfig


class JaDocClassifyConfig(ModernBertConfig):
    """8領域・43文書種別・2形式・確定ルーティング閾値を保持する設定クラス。"""

    model_type = "modernbert"

    def __init__(
        self,
        domain_labels=None,
        type_labels=None,
        format_labels=None,
        high_risk_types=None,
        routing_thresholds=None,
        max_length=1024,
        **kwargs,
    ):
        for k in (
            "domain_id2label",
            "domain_label2id",
            "type_id2label",
            "type_label2id",
            "format_id2label",
            "format_label2id",
        ):
            kwargs.pop(k, None)

        # transformers 4.57.x 系互換のため rope_parameters からの補完も許容
        rp = kwargs.get("rope_parameters") or {}
        if "global_rope_theta" not in kwargs and isinstance(rp, dict) and "full_attention" in rp:
            kwargs["global_rope_theta"] = float(rp["full_attention"].get("rope_theta", 160000.0))
        if "local_rope_theta" not in kwargs and isinstance(rp, dict) and "sliding_attention" in rp:
            kwargs["local_rope_theta"] = float(rp["sliding_attention"].get("rope_theta", 10000.0))

        super().__init__(**kwargs)
        self.domain_labels = list(domain_labels) if domain_labels is not None else []
        self.type_labels = list(type_labels) if type_labels is not None else []
        self.format_labels = (
            list(format_labels) if format_labels is not None else ["structured", "unstructured"]
        )
        self.high_risk_types = list(high_risk_types) if high_risk_types is not None else []
        default_th = {
            "domain_reject_threshold": 0.95,
            "type_reject_threshold": 0.80,
            "high_risk_prob_sum_threshold": 0.12,
        }
        if routing_thresholds is not None:
            default_th.update({k: float(v) for k, v in dict(routing_thresholds).items()})
        self.routing_thresholds = default_th
        self.max_length = int(max_length)

        self.domain_id2label = {str(i): lbl for i, lbl in enumerate(self.domain_labels)}
        self.domain_label2id = {lbl: i for i, lbl in enumerate(self.domain_labels)}
        self.type_id2label = {str(i): lbl for i, lbl in enumerate(self.type_labels)}
        self.type_label2id = {lbl: i for i, lbl in enumerate(self.type_labels)}
        self.format_id2label = {str(i): lbl for i, lbl in enumerate(self.format_labels)}
        self.format_label2id = {lbl: i for i, lbl in enumerate(self.format_labels)}
'''

MODELING_PY = '''"""ja-doc-classify-v1 カスタム Hugging Face モデル＆ルーティング後処理クラス。"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from transformers import ModernBertModel, ModernBertPreTrainedModel
from transformers.utils import ModelOutput

from .configuration_ja_doc_classify import JaDocClassifyConfig


@dataclass
class JaDocClassifyOutput(ModelOutput):
    """3ヘッド（領域・文書種別・形式）のロジット出力。"""

    logits_domain: Optional[torch.FloatTensor] = None
    logits_type: Optional[torch.FloatTensor] = None
    logits_format: Optional[torch.FloatTensor] = None


def _to_numpy_2d(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().to(torch.float64).numpy()
    else:
        arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _softmax_f64(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def route_from_logits(
    logits_domain,
    logits_type,
    logits_format,
    domain_labels: Sequence[str],
    type_labels: Sequence[str],
    format_labels: Sequence[str],
    high_risk_types: Sequence[str],
    domain_reject_threshold: float = 0.95,
    type_reject_threshold: float = 0.80,
    high_risk_prob_sum_threshold: float = 0.12,
) -> List[dict]:
    """3ヘッドの生ロジットから float64 softmax と確定閾値に基づくルーティング結果を返す。"""
    probs_d = _softmax_f64(_to_numpy_2d(logits_domain))
    probs_t = _softmax_f64(_to_numpy_2d(logits_type))
    probs_f = _softmax_f64(_to_numpy_2d(logits_format))

    hr_set = set(high_risk_types)
    hr_indices = [i for i, lbl in enumerate(type_labels) if lbl in hr_set]

    results: List[dict] = []
    for i in range(probs_d.shape[0]):
        pd = probs_d[i]
        pt = probs_t[i]
        pf = probs_f[i]

        d_idx = int(np.argmax(pd))
        t_idx = int(np.argmax(pt))
        f_idx = int(np.argmax(pf))

        d_conf = float(pd[d_idx])
        t_conf = float(pt[t_idx])
        f_conf = float(pf[f_idx])
        hr_sum = float(pt[hr_indices].sum()) if hr_indices else 0.0

        d_top1 = str(domain_labels[d_idx])
        t_top1 = str(type_labels[t_idx])
        f_top1 = str(format_labels[f_idx])

        d_rej = bool(d_conf < domain_reject_threshold)
        t_rej = bool(t_conf < type_reject_threshold)
        hr_flag = bool(hr_sum >= high_risk_prob_sum_threshold)

        results.append(
            {
                "domain": None if d_rej else d_top1,
                "domain_rejected": d_rej,
                "domain_top1": d_top1,
                "domain_confidence": d_conf,
                "document_type": None if t_rej else t_top1,
                "type_rejected": t_rej,
                "document_type_top1": t_top1,
                "type_confidence": t_conf,
                "high_risk": hr_flag,
                "high_risk_prob_sum": hr_sum,
                "document_format": f_top1,
                "format_confidence": f_conf,
            }
        )
    return results


class JaDocClassifyModel(ModernBertPreTrainedModel):
    """共有 ModernBERT エンコーダ＋3線形ヘッドの文書分類・ルーティングモデル。"""

    config_class = JaDocClassifyConfig
    _tied_weights_keys = None

    def __init__(self, config: JaDocClassifyConfig):
        super().__init__(config)
        self.encoder = ModernBertModel(config)
        h = config.hidden_size
        n_domain = len(config.domain_labels) if config.domain_labels else 8
        n_type = len(config.type_labels) if config.type_labels else 43
        n_format = len(config.format_labels) if config.format_labels else 2
        self.head_domain = nn.Linear(h, n_domain)
        self.head_type = nn.Linear(h, n_type)
        self.head_format = nn.Linear(h, n_format)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JaDocClassifyOutput:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]
        return JaDocClassifyOutput(
            logits_domain=self.head_domain(pooled),
            logits_type=self.head_type(pooled),
            logits_format=self.head_format(pooled),
        )

    def postprocess(
        self,
        outputs: JaDocClassifyOutput,
        domain_reject_threshold: Optional[float] = None,
        type_reject_threshold: Optional[float] = None,
        high_risk_prob_sum_threshold: Optional[float] = None,
    ) -> List[dict]:
        """モデル出力ロジットへ config 同梱の確定閾値を適用してルーティング辞書を返す。"""
        th = getattr(self.config, "routing_thresholds", {}) or {}
        d_th = float(
            th.get("domain_reject_threshold", 0.95)
            if domain_reject_threshold is None
            else domain_reject_threshold
        )
        t_th = float(
            th.get("type_reject_threshold", 0.80)
            if type_reject_threshold is None
            else type_reject_threshold
        )
        hr_th = float(
            th.get("high_risk_prob_sum_threshold", 0.12)
            if high_risk_prob_sum_threshold is None
            else high_risk_prob_sum_threshold
        )
        return route_from_logits(
            outputs.logits_domain,
            outputs.logits_type,
            outputs.logits_format,
            domain_labels=self.config.domain_labels,
            type_labels=self.config.type_labels,
            format_labels=self.config.format_labels,
            high_risk_types=self.config.high_risk_types,
            domain_reject_threshold=d_th,
            type_reject_threshold=t_th,
            high_risk_prob_sum_threshold=hr_th,
        )

    def predict(
        self,
        texts: Union[str, Sequence[str]],
        tokenizer,
        max_length: Optional[int] = None,
        batch_size: int = 16,
        domain_reject_threshold: Optional[float] = None,
        type_reject_threshold: Optional[float] = None,
        high_risk_prob_sum_threshold: Optional[float] = None,
    ) -> Union[dict, List[dict]]:
        """テキストまたはテキスト列に対して前処理・推論・閾値判定を一括実行する。"""
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)
        if not text_list:
            return []

        max_len = int(max_length if max_length is not None else getattr(self.config, "max_length", 1024))
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        all_results: List[dict] = []
        with torch.no_grad():
            for i in range(0, len(text_list), batch_size):
                batch = text_list[i : i + batch_size]
                enc = tokenizer(
                    batch,
                    truncation=True,
                    max_length=max_len,
                    padding=len(batch) > 1,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
                out = self.forward(**enc)
                all_results.extend(
                    self.postprocess(
                        out,
                        domain_reject_threshold=domain_reject_threshold,
                        type_reject_threshold=type_reject_threshold,
                        high_risk_prob_sum_threshold=high_risk_prob_sum_threshold,
                    )
                )
        if was_training:
            self.train()
        return all_results[0] if single else all_results
'''


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_fixed_risk_tiers(
    v0_csv_path: Path,
    build_stats_path: Path,
    dev_rows: list[dict],
    out_csv_path: Path,
) -> tuple[list[dict], list[dict]]:
    """risk_tiers_v0.csv の 6 件の領域不一致を build-stats.json 基準で修正し、dev 実データと突合検証する。"""
    stats = json.loads(build_stats_path.read_text(encoding="utf-8"))
    mismatch_list = stats.get("risk_tier_domain_mismatch", [])
    mismatch_map = {m["type"]: m["actual"] for m in mismatch_list}
    assert len(mismatch_list) == 6, f"Expected 6 mismatches, got {len(mismatch_list)}"

    dev_type_to_domain = {}
    for r in dev_rows:
        dt = r["document_type"]
        dm = r["domain"]
        if dt in dev_type_to_domain:
            assert dev_type_to_domain[dt] == dm, f"Inconsistent domain in dev for {dt}"
        dev_type_to_domain[dt] = dm

    with v0_csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 43

    updated_rows = []
    applied_fixes = []
    for row in rows:
        dt = row["document_type"]
        orig_dom = row["domain"]
        new_dom = mismatch_map.get(dt, orig_dom)
        if new_dom != orig_dom:
            applied_fixes.append({"document_type": dt, "from": orig_dom, "to": new_dom})
        assert new_dom == dev_type_to_domain[dt], (
            f"Fixed domain mismatch for {dt}: {new_dom} != {dev_type_to_domain[dt]}"
        )
        assert (row["risk_tier"] == "高機密") == (dt in HIGH_RISK), (
            f"Risk tier mismatch for {dt}: {row['risk_tier']}"
        )
        updated_rows.append(
            {
                "document_type": dt,
                "domain": new_dom,
                "risk_tier": row["risk_tier"],
            }
        )

    assert len(applied_fixes) == 6, f"Expected 6 fixes applied, got {len(applied_fixes)}"
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["document_type", "domain", "risk_tier"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(updated_rows)

    return updated_rows, applied_fixes


def main() -> int:
    root = Path(".")  # project root (run from the model folder)
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sbintuitions/modernbert-ja-130m")
    ap.add_argument("--state", default=str(root / "runs" / "prod-modernbert-v2" / "model_state.pt"))
    ap.add_argument("--tok-dir", default=str(root / "runs" / "prod-modernbert-v2" / "final"))
    ap.add_argument("--eval-dev", default=str(root / "runs" / "prod-modernbert-v2" / "eval_dev.json"))
    ap.add_argument("--dev-jsonl", default=str(root / "data" / "pilot" / "dev.jsonl"))
    ap.add_argument("--ood-v2", default=str(root / "data" / "ood_v2" / "ood-v2-200.jsonl"))
    ap.add_argument("--risk-v0", default=str(root / "risk_tiers_v0.csv"))
    ap.add_argument("--build-stats", default=str(root / "data" / "pilot" / "build-stats.json"))
    ap.add_argument("--out-dir", default=str(root / "runs" / "prod-modernbert-v2" / "final-hf"))
    ap.add_argument(
        "--verify-json",
        default=str(root / "runs" / "prod-modernbert-v2" / "hf_export_verify.json"),
    )
    args = ap.parse_args()

    t0 = time.time()

    # 1. ラベル順序確定（eval_dev.json + dev.jsonl、sealed-test は非参照）
    eval_dev = json.loads(Path(args.eval_dev).read_text(encoding="utf-8"))
    types = sorted(eval_dev["per_type"].keys())
    dev_rows = load_jsonl(Path(args.dev_jsonl))
    dev_types = sorted({r["document_type"] for r in dev_rows})
    assert len(types) == 43 and types == dev_types, "Type label order verification failed"
    high_risk_sorted = sorted(HIGH_RISK)
    assert len(high_risk_sorted) == 22
    hr_indices = [i for i, t in enumerate(types) if t in HIGH_RISK]
    assert len(hr_indices) == 22

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. 修正済み risk_tiers.csv 書き出し
    _, applied_fixes = build_fixed_risk_tiers(
        Path(args.risk_v0), Path(args.build_stats), dev_rows, out_dir / "risk_tiers.csv"
    )

    # 3. カスタム config / modeling コード書き出し
    (out_dir / "configuration_ja_doc_classify.py").write_text(CONFIGURATION_PY, encoding="utf-8", newline="\n")
    (out_dir / "modeling_ja_doc_classify.py").write_text(MODELING_PY, encoding="utf-8", newline="\n")

    # 4. config.json 作成
    base_cfg = AutoConfig.from_pretrained(args.base)
    cfg_dict = base_cfg.to_dict()
    for unused_k in ("id2label", "label2id", "_name_or_path"):
        cfg_dict.pop(unused_k, None)

    domain_id2label = {str(i): l for i, l in enumerate(DOMAINS)}
    domain_label2id = {l: i for i, l in enumerate(DOMAINS)}
    type_id2label = {str(i): l for i, l in enumerate(types)}
    type_label2id = {l: i for i, l in enumerate(types)}
    format_id2label = {str(i): l for i, l in enumerate(FORMATS)}
    format_label2id = {l: i for i, l in enumerate(FORMATS)}

    cfg_dict.update(
        {
            "architectures": ["JaDocClassifyModel"],
            "auto_map": {
                "AutoConfig": "configuration_ja_doc_classify.JaDocClassifyConfig",
                "AutoModel": "modeling_ja_doc_classify.JaDocClassifyModel",
            },
            "dtype": "float32",
            "tie_word_embeddings": False,
            "use_cache": False,
            "max_length": 1024,
            "domain_labels": DOMAINS,
            "domain_id2label": domain_id2label,
            "domain_label2id": domain_label2id,
            "type_labels": types,
            "type_id2label": type_id2label,
            "type_label2id": type_label2id,
            "format_labels": FORMATS,
            "format_id2label": format_id2label,
            "format_label2id": format_label2id,
            "high_risk_types": high_risk_sorted,
            "routing_thresholds": DEFAULT_THRESHOLDS,
        }
    )
    (out_dir / "config.json").write_text(
        json.dumps(cfg_dict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    # 5. routing_config.json（ONNX・外部ランタイム共通純JSON）
    routing_cfg = {
        "model_id": "ja-doc-classify-v1",
        "model_revision": "v2",
        "base_model": args.base,
        "max_length": 1024,
        "unknown_label": None,
        "routing_thresholds": DEFAULT_THRESHOLDS,
        "domain_labels": DOMAINS,
        "type_labels": types,
        "format_labels": FORMATS,
        "high_risk_types": high_risk_sorted,
        "high_risk_type_indices": hr_indices,
    }
    (out_dir / "routing_config.json").write_text(
        json.dumps(routing_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    # 6. トークナイザ複製（元の final/ は不変）
    tok_src = Path(args.tok_dir)
    for fname in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(tok_src / fname, out_dir / fname)

    # 7. 重みを safetensors (float32, CPU) として書き出し
    raw_sd = torch.load(args.state, map_location="cpu", weights_only=True)
    clean_sd = {k: v.detach().cpu().to(torch.float32).contiguous() for k, v in raw_sd.items()}
    assert len(clean_sd) == 122
    save_safetensors(clean_sd, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

    # --- 書き出し後検証 ---
    # (A) 重みビット一致
    loaded_st = load_safetensors(str(out_dir / "model.safetensors"), device="cpu")
    assert set(loaded_st.keys()) == set(raw_sd.keys()), "Tensor key set mismatch"
    max_weight_diff = 0.0
    for k, v in raw_sd.items():
        assert loaded_st[k].dtype == torch.float32
        assert loaded_st[k].shape == v.shape
        assert torch.equal(loaded_st[k], v.cpu().to(torch.float32)), f"Bitwise mismatch on {k}"
        max_weight_diff = max(max_weight_diff, float((loaded_st[k] - v.cpu().to(torch.float32)).abs().max()))
    assert max_weight_diff == 0.0

    # (B) Hugging Face ロード検証（remote code 有・無の両方）
    cfg_remote = AutoConfig.from_pretrained(str(out_dir), trust_remote_code=True, local_files_only=True)
    cfg_plain = AutoConfig.from_pretrained(str(out_dir), local_files_only=True)
    assert cfg_remote.routing_thresholds == DEFAULT_THRESHOLDS
    assert getattr(cfg_plain, "routing_thresholds") == DEFAULT_THRESHOLDS

    tok_hf = AutoTokenizer.from_pretrained(str(out_dir), local_files_only=True)
    model_hf = AutoModel.from_pretrained(
        str(out_dir), trust_remote_code=True, local_files_only=True, dtype=torch.float32
    ).eval()

    # (C) OOD v2 先頭 10 件での元 MultiHead vs HF 書き出しモデル ロジット一致検証
    ref_model = MultiHead(args.base, len(types))
    ref_model.load_state_dict(raw_sd)
    ref_model.cpu().eval()

    sample_rows = load_jsonl(Path(args.ood_v2))[:10]
    max_diffs = {"logits_domain": 0.0, "logits_type": 0.0, "logits_format": 0.0}
    sample_preds = []
    with torch.no_grad():
        for r in sample_rows:
            enc = tok_hf(r["text"], truncation=True, max_length=1024, return_tensors="pt")
            out_ref = ref_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            out_hf = model_hf(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            for k in max_diffs:
                diff = float((out_ref[k] - getattr(out_hf, k)).abs().max().item())
                max_diffs[k] = max(max_diffs[k], diff)
            res = model_hf.predict(r["text"], tok_hf)
            sample_preds.append(
                {
                    "id": r["id"],
                    "gold_domain": r["domain"],
                    "gold_type": r["document_type"],
                    "pred": res,
                }
            )
    for k, diff in max_diffs.items():
        assert diff <= 1e-6, f"HF logit mismatch on {k}: {diff}"

    verify = {
        "hf_dir": str(out_dir),
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        "tensor_count": len(loaded_st),
        "max_weight_abs_diff": max_weight_diff,
        "bitwise_weights_match": True,
        "routing_thresholds": cfg_remote.routing_thresholds,
        "risk_tiers_fixes_applied": applied_fixes,
        "parity_ood_v2_first10_max_abs_logit_diff": max_diffs,
        "sample_predictions_first3": sample_preds[:3],
        "elapsed_sec": round(time.time() - t0, 2),
        "sealed_test_opened": False,
        "retrained": False,
    }
    Path(args.verify_json).write_text(json.dumps(verify, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in verify.items() if k != "sample_predictions_first3"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
