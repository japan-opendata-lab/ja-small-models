#!/usr/bin/env python3
"""第3号 ja-punct-restore-v1: 採用ロジットバイアスの分類ヘッド焼き込み＆dev200件一致検証。

runs/prod-modernbert/final を読み取り専用で読み込み、classifier.bias に採用バイアス
[0.0, -1.037, -1.247, -2.412, -2.786] を加算して runs/prod-modernbert/final-baked に保存する。
保存済み重みの全テンソル差分検証に加え、dev.jsonl 先頭 200 ウィンドウで
(元モデル + 外部バイアス) と (焼き込み後再読込モデル) の予測完全一致を確認する。
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForTokenClassification, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_final import decode_labels_from_logits, extract_char_logits, load_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="runs/prod-modernbert/final")
    ap.add_argument("--bias-json", default="runs/prod-modernbert/logit_bias.json")
    ap.add_argument("--dev", default="data/prod/dev.jsonl")
    ap.add_argument("--out-model", default="runs/prod-modernbert/final-baked")
    ap.add_argument("--out-verify", default="runs/prod-modernbert/bias_bake_verify.json")
    ap.add_argument("--num-windows", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    src_dir = Path(args.src_model)
    out_dir = Path(args.out_model)
    bias_obj = json.loads(Path(args.bias_json).read_text(encoding="utf-8"))
    bias_delta = [float(x) for x in bias_obj["logit_bias"]]
    assert len(bias_delta) == 5, bias_delta

    # 1. 元モデル読み込みとバイアス焼き込み
    model_baked = AutoModelForTokenClassification.from_pretrained(str(src_dir), dtype=torch.float32).eval()
    bias_before = model_baked.classifier.bias.detach().cpu().clone()
    delta_t = torch.tensor(bias_delta, dtype=torch.float32)
    with torch.no_grad():
        model_baked.classifier.bias.add_(delta_t)
    expected_bias_after = bias_before + delta_t

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_baked.save_pretrained(str(out_dir), safe_serialization=True)
    for tok_name in ("tokenizer.json", "tokenizer_config.json"):
        src_tok = src_dir / tok_name
        assert src_tok.exists(), f"Missing tokenizer file: {src_tok}"
        shutil.copy2(src_tok, out_dir / tok_name)
    del model_baked

    # 2. safetensors テンソルレベルの整合検証（classifier.bias 以外はビット完全一致）
    t_orig = load_file(str(src_dir / "model.safetensors"))
    t_baked = load_file(str(out_dir / "model.safetensors"))
    assert set(t_orig.keys()) == set(t_baked.keys()), "Tensor keys mismatch"
    other_diffs = 0
    for k in t_orig:
        if k == "classifier.bias":
            continue
        if not torch.equal(t_orig[k], t_baked[k]):
            other_diffs += 1
    assert other_diffs == 0, f"Non-bias tensors modified: {other_diffs}"
    bias_after_loaded = t_baked["classifier.bias"].detach().cpu()
    bias_tensor_max_diff = float((bias_after_loaded - expected_bias_after).abs().max().item())
    assert bias_tensor_max_diff == 0.0, bias_tensor_max_diff

    # 3. dev 先頭 N 件で (元モデル + 外部バイアス) vs (焼き込み後再読込モデル) を比較
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_rows = load_jsonl(Path(args.dev))[: args.num_windows]
    texts = [r["text"] for r in dev_rows]
    tok_orig = AutoTokenizer.from_pretrained(str(src_dir))
    tok_baked = AutoTokenizer.from_pretrained(str(out_dir))

    model_orig = AutoModelForTokenClassification.from_pretrained(str(src_dir), dtype=torch.float32).to(device).eval()
    logits_orig = extract_char_logits(model_orig, tok_orig, texts, device=device, batch_size=args.batch_size, label="orig+bias")
    del model_orig
    if device == "cuda":
        torch.cuda.empty_cache()

    model_reloaded = AutoModelForTokenClassification.from_pretrained(str(out_dir), dtype=torch.float32).to(device).eval()
    logits_baked = extract_char_logits(model_reloaded, tok_baked, texts, device=device, batch_size=args.batch_size, label="baked")
    del model_reloaded
    if device == "cuda":
        torch.cuda.empty_cache()

    labels_orig = decode_labels_from_logits(logits_orig, bias_delta)
    labels_baked = decode_labels_from_logits(logits_baked, [0.0] * 5)

    delta_np = np.asarray(bias_delta, dtype=np.float32)
    win_exact = 0
    char_exact = 0
    total_chars = 0
    cov_exact = 0
    total_cov = 0
    abs_diffs = []

    for lo, lb, po, pb in zip(logits_orig, logits_baked, labels_orig, labels_baked):
        assert len(lo) == len(lb) == len(po) == len(pb)
        n = len(po)
        total_chars += n
        eq_arr = np.asarray(po, dtype=np.int64) == np.asarray(pb, dtype=np.int64)
        char_exact += int(eq_arr.sum())
        if po == pb:
            win_exact += 1
        cov_mask = ~np.isnan(lo[:, 0])
        assert np.array_equal(cov_mask, ~np.isnan(lb[:, 0]))
        if cov_mask.any():
            total_cov += int(cov_mask.sum())
            cov_exact += int(eq_arr[cov_mask].sum())
            abs_diffs.append(np.abs((lo[cov_mask] + delta_np) - lb[cov_mask]))

    all_diffs = np.concatenate([d.reshape(-1) for d in abs_diffs])
    max_abs_logit_diff = float(all_diffs.max())
    mean_abs_logit_diff = float(all_diffs.mean())

    assert win_exact == len(dev_rows), (win_exact, len(dev_rows))
    assert char_exact == total_chars, (char_exact, total_chars)

    summary = {
        "src_model": str(src_dir),
        "baked_model": str(out_dir),
        "device": device,
        "dtype": "float32",
        "num_windows": len(dev_rows),
        "total_chars": total_chars,
        "covered_chars": total_cov,
        "bias_before": [round(float(x), 8) for x in bias_before.tolist()],
        "bias_delta": bias_delta,
        "bias_after": [round(float(x), 8) for x in bias_after_loaded.tolist()],
        "non_bias_tensors_modified": other_diffs,
        "bias_tensor_max_diff": bias_tensor_max_diff,
        "window_exact_match": win_exact,
        "char_exact_match": char_exact,
        "covered_char_exact_match": cov_exact,
        "max_abs_logit_diff": round(max_abs_logit_diff, 8),
        "mean_abs_logit_diff": round(mean_abs_logit_diff, 10),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    Path(args.out_verify).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
