#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 辞書方式ベースライン評価スクリプト。

dev.jsonl (2,000件) および homograph-eval.jsonl (1,000件) に対し、
SudachiPy SplitMode.C / SplitMode.A / UniDic (unidic-lite) の3手法を
同一の eval_reading.evaluate_predictions で計測し runs/baselines/eval_baselines.json へ出力する。

注意: 封印テスト (sealed-test.jsonl) は一切参照しない。
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_reading as er  # noqa: E402
from span_extractor import DictionaryBaselines  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval_split(
    baselines: DictionaryBaselines, rows: List[dict], split_name: str
) -> dict:
    results = {}
    for method_name, fn in [
        ("sudachi_mode_c", lambda txt: baselines.predict_sudachi(txt, mode="C")),
        ("sudachi_mode_a", lambda txt: baselines.predict_sudachi(txt, mode="A")),
        ("unidic_lite", lambda txt: baselines.predict_unidic(txt)),
    ]:
        t0 = time.time()
        preds = [fn(r["text"]) for r in rows]
        dt = time.time() - t0
        metrics = er.evaluate_predictions(rows, preds, eval_sec=dt)
        results[method_name] = metrics
        print(
            f"[{split_name}] {method_name:16s}: "
            f"sent_read={metrics['sentence_reading_acc']:.4f}, "
            f"sent_pron={metrics['sentence_pron_acc']:.4f}, "
            f"target_homo={metrics['target_homograph_acc']}, "
            f"sec={metrics['eval_sec']}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data" / "pilot"))
    parser.add_argument("--master-dir", type=str, default=str(ROOT / "data" / "master"))
    parser.add_argument("--out", type=str, default=str(ROOT / "runs" / "baselines" / "eval_baselines.json"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    baselines = DictionaryBaselines(Path(args.master_dir))

    dev_rows = load_jsonl(data_dir / "dev.jsonl")
    out_obj = {
        "dev": run_eval_split(baselines, dev_rows, "dev"),
    }
    homo_path = data_dir / "homograph-eval.jsonl"
    if homo_path.exists():
        homo_rows = load_jsonl(homo_path)
        out_obj["homograph_eval"] = run_eval_split(baselines, homo_rows, "homograph-eval")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print("Saved dictionary baselines to:", out_path)


if __name__ == "__main__":
    main()
