#!/usr/bin/env python3
"""第7号 ja-address-ner-v1 封印テスト最終評価スクリプト（1回限り・勝者 bert-v3 専用）。

data/prod/sealed-test.jsonl (5,000件) 上で、runs/prod-bert-v3/final の
第1段階（住所スパン完全一致 F1・5構成要素別正確さ）および
第2段階接続（AddressLinker によるコード付与正確さ・棄却率・誤断言率）を 1 回だけ評価し、
runs/prod-bert-v3/eval_sealed.json に保存する。

安全ガード:
  - ModernBERT 系モデルの指定を拒否する。
  - eval_sealed.json が既に存在する場合は上書きせず終了する（1回限り保証）。
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pilot_addr as tpa  # noqa: E402
import train_prod_addr as tprod  # noqa: E402
from linker import AddressLinker  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=r"runs/prod-bert-v3/final")
    ap.add_argument("--sealed", default=r"data/prod/sealed-test.jsonl")
    ap.add_argument("--out", default=r"runs/prod-bert-v3/eval_sealed.json")
    ap.add_argument("--max-len", type=int, default=384)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    sealed_path = Path(args.sealed)
    out_path = Path(args.out)

    # 1回限りガード
    if out_path.exists():
        raise RuntimeError(
            f"封印テスト評価結果が既に存在します（1回限り厳守）: {out_path}"
        )

    # ModernBERT 拒否ガード（パス・config.json 双方検査）
    if "modernbert" in str(model_dir).lower():
        raise RuntimeError(f"ModernBERT の封印評価は禁止されています: {model_dir}")
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if "modernbert" in str(cfg.get("model_type", "")).lower() or "modernbert" in str(
        cfg.get("_name_or_path", "")
    ).lower():
        raise RuntimeError(f"ModernBERT 系モデルの封印評価は禁止されています: {cfg}")

    # パイロット封印テストの誤用防止
    if "pilot" in str(sealed_path).lower():
        raise RuntimeError(f"パイロット封印テストは使用禁止です: {sealed_path}")

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    max_pos = getattr(model.config, "max_position_embeddings", args.max_len)
    max_len = min(args.max_len, int(max_pos))

    sealed_ds = tpa.AddrDataset(sealed_path, tok, max_len=max_len, verify_n=999999)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    linker = AddressLinker()

    stage1_sealed, linker_sealed = tprod.run_inference_and_eval(
        model=model,
        ds=sealed_ds,
        pad_id=pad_id,
        device=device,
        linker=linker,
        pred_out_path=None,
        eval_bs=64,
    )

    result = {
        "model_dir": str(model_dir),
        "model_type": cfg.get("model_type"),
        "base_name_or_path": cfg.get("_name_or_path"),
        "sealed_file": str(sealed_path),
        "max_len": max_len,
        "fp16_eval": False,
        "stage1": stage1_sealed,
        "stage2_linker": linker_sealed,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    b_link = linker_sealed["bert_v3_plus_linker"]
    summary = {
        "eval_rows": stage1_sealed["eval_rows"],
        "truncated": stage1_sealed["truncated"],
        "eval_sec": stage1_sealed["eval_sec"],
        "eval_ms_per_sent": stage1_sealed["eval_ms_per_sent"],
        "address_exact_span": stage1_sealed["address_exact_span"],
        "per_component_exact": {
            k: stage1_sealed["per_component_exact"][k]["f1"]
            if isinstance(stage1_sealed["per_component_exact"][k], dict)
            else stage1_sealed["per_component_exact"][k]
            for k in ("PREF", "CITY", "TOWN", "BLOCK", "BUILDING", "micro", "macro_f1")
        },
        "no_address_sentence_fpr": stage1_sealed["no_address_fpr"]["sentence_fpr"],
        "linker_oracle_ok": (
            linker_sealed["oracle_gold_components"]["resolvable"]["accuracy"] == 1.0
            and linker_sealed["oracle_gold_components"]["unresolvable"]["strict_rejection_rate"] == 1.0
        ),
        "linker_resolvable_acc": b_link["resolvable"]["accuracy"],
        "linker_wrong_code_rate": b_link["resolvable"]["wrong_code_rate"],
        "linker_false_reject_rate": b_link["resolvable"]["false_reject_rate"],
        "linker_unres_strict_reject": b_link["unresolvable"]["strict_rejection_rate"],
        "linker_unres_non_assert": b_link["unresolvable"]["non_assertion_rate"],
        "linker_unres_false_assert": b_link["unresolvable"]["false_assertion_rate"],
        "linker_e2e_assert_prec": b_link["assertion_summary"]["assertion_precision_end_to_end"],
        "linker_fp_unmatched_resolved": b_link["assertion_summary"]["fp_unmatched_resolved"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
