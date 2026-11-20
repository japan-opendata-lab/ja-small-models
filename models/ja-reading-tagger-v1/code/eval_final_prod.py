#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 本番封印テスト（1回限り）＆同形異音語最終比較スクリプト。

対象モデル:
- 本番最良モデル: runs/prod-modernbert/best/ (sbintuitions/modernbert-ja-130m + calibration.json)
- 辞書比較 3 手法: SudachiPy Mode C / SudachiPy Mode A / UniDic (unidic-lite)
※ tohoku-nlp/bert-base-japanese-v3 の封印評価は一切行わない。
※ data/pilot/sealed-test.jsonl は一切参照・開封しない。

評価対象:
1) data/prod/sealed-test.jsonl (5,000件) — 1回限り開封評価
2) data/pilot/homograph-eval.jsonl (1,000件) — 同形異音語評価の最終比較
"""
import json
import sys
import time
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_prod_topup as apt  # noqa: E402
import eval_reading as er  # noqa: E402
from reading_tagger import JaReadingTagger  # noqa: E402
from span_extractor import DictionaryBaselines, SpanExtractor  # noqa: E402
import train_prod_reading as tpv2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def eval_model_split(
    model,
    tok,
    label2id,
    extractor: SpanExtractor,
    rows: List[dict],
    device: str,
    override_tau: float,
    name: str,
) -> dict:
    collected, sec, trunc = tpv2.collect_eval_logits_and_decode(
        model,
        tok,
        rows,
        extractor,
        label2id,
        max_len=256,
        batch_size=32,
        device=device,
        verify_all_offsets=True,
    )
    preds = tpv2.decode_from_collected(collected, extractor, override_tau=override_tau)
    metrics = er.evaluate_predictions(rows, preds, eval_sec=sec)
    metrics["truncated"] = trunc
    metrics["override_tau"] = override_tau
    print(
        f"[{name}] modernbert_prod (tau={override_tau:.2f}): "
        f"sent_read={metrics['sentence_reading_acc']:.4f}, "
        f"sent_pron={metrics['sentence_pron_acc']:.4f}, "
        f"target_homo={metrics['target_homograph_acc']}, sec={metrics['eval_sec']}",
        flush=True,
    )
    return metrics


def main() -> None:
    master_dir = ROOT / "data" / "master"
    best_dir = ROOT / "runs" / "prod-modernbert" / "best"
    out_path = ROOT / "runs" / "prod-modernbert" / "eval_final.json"

    train_rows = load_jsonl(ROOT / "data" / "prod" / "train.jsonl")
    sealed_rows = load_jsonl(ROOT / "data" / "prod" / "sealed-test.jsonl")
    homo_rows = load_jsonl(ROOT / "data" / "pilot" / "homograph-eval.jsonl")

    # 付帯監査: prod/train vs prod/sealed-test の完全一致・LSH (Jaccard >= 0.80) 重複 0 検証
    checker = apt.FastLSHChecker([r["text"] for r in sealed_rows])
    exact_ov, near_ov = checker.check_overlap([r["text"] for r in train_rows])
    assert exact_ov == 0 and near_ov == 0, (
        f"リーク監査エラー (prod/train vs prod/sealed-test): exact={exact_ov}, near_lsh_080={near_ov}"
    )
    leak_audit = {
        "train_rows": len(train_rows),
        "sealed_rows": len(sealed_rows),
        "exact_norm": exact_ov,
        "near_dup_lsh_080": near_ov,
    }
    print(
        f"[Audit] prod/train vs prod/sealed-test: exact_norm={exact_ov}, near_dup_lsh_080={near_ov} [PASS]",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(best_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(best_dir)).to(device)
    with open(best_dir / "label_map.json", encoding="utf-8") as f:
        label2id = json.load(f)["label2id"]

    cal_path = best_dir / "calibration.json"
    if cal_path.exists():
        with open(cal_path, encoding="utf-8") as f:
            override_tau = float(json.load(f).get("override_tau", 0.0))
    else:
        override_tau = 0.0
    print(f"Loaded calibration override_tau={override_tau:.2f}", flush=True)

    extractor = SpanExtractor(master_dir)
    baselines = DictionaryBaselines(master_dir)

    results = {
        "override_tau": override_tau,
        "leak_audit_train_vs_sealed": leak_audit,
        "sealed_test": {},
        "homograph_eval": {},
    }

    # 1. sealed-test (5,000件) — 1回限り評価
    results["sealed_test"]["modernbert_prod"] = eval_model_split(
        model, tok, label2id, extractor, sealed_rows, device, override_tau, "sealed-test"
    )
    for bname, fn in [
        ("sudachi_mode_c", lambda t: baselines.predict_sudachi(t, mode="C")),
        ("sudachi_mode_a", lambda t: baselines.predict_sudachi(t, mode="A")),
        ("unidic_lite", lambda t: baselines.predict_unidic(t)),
    ]:
        t0 = time.time()
        preds = [fn(r["text"]) for r in sealed_rows]
        dt = time.time() - t0
        m = er.evaluate_predictions(sealed_rows, preds, eval_sec=dt)
        results["sealed_test"][bname] = m
        print(
            f"[sealed-test] {bname:16s}: sent_read={m['sentence_reading_acc']:.4f}, "
            f"sent_pron={m['sentence_pron_acc']:.4f}, sec={m['eval_sec']}",
            flush=True,
        )

    # 2. homograph-eval (1,000件)
    results["homograph_eval"]["modernbert_prod"] = eval_model_split(
        model, tok, label2id, extractor, homo_rows, device, override_tau, "homograph-eval"
    )
    for bname, fn in [
        ("sudachi_mode_c", lambda t: baselines.predict_sudachi(t, mode="C")),
        ("sudachi_mode_a", lambda t: baselines.predict_sudachi(t, mode="A")),
        ("unidic_lite", lambda t: baselines.predict_unidic(t)),
    ]:
        t0 = time.time()
        preds = [fn(r["text"]) for r in homo_rows]
        dt = time.time() - t0
        m = er.evaluate_predictions(homo_rows, preds, eval_sec=dt)
        results["homograph_eval"][bname] = m
        print(
            f"[homograph-eval] {bname:16s}: sent_read={m['sentence_reading_acc']:.4f}, "
            f"target_homo={m['target_homograph_acc']}, sec={m['eval_sec']}",
            flush=True,
        )

    # 3. 参照実装テスト（回帰テスト文含む8文）
    tagger = JaReadingTagger(best_dir, master_dir, device=device)
    ref_samples = [
        "人気のない路地と人気のある店を歩いて回った。",
        "工事現場の工夫が工夫を凝らして安全を守る。",
        "重要な会議を行った後に、家族で海へ行った。",
        "ドアが開いた瞬間に新しい店を開く準備を始めた。",
        "彼女は日本画の大家だが、アパートの大家に家賃を払った。",
        "強いてご交際を求めるのもどうかと思って遠慮している。",
        "地球上の動物が二家の主張によって守られている。",
        "煙草を一本吸ってから六個入りの箱を運んだ。",
    ]
    ref_out = []
    for s in ref_samples:
        spans = tagger.predict(s)
        for sp in spans:
            assert s[sp["start"] : sp["end"]] == sp["surface"]
        ref_out.append({"text": s, "spans": spans})
    results["reference_samples"] = ref_out
    results["reference_tagger_tau"] = tagger.override_tau

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved final production evaluation to:", out_path, flush=True)


if __name__ == "__main__":
    main()
