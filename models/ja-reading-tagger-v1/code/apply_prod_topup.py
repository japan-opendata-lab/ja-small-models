#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 本番 train.jsonl への Qwen3 補完安全適用＆0リーク再検証スクリプト。

仕様:
- data/raw/qwen3_prod_topup_verified.jsonl を再検証のうえ、data/prod/train.jsonl 内の
  カタログ同形異音語を1つも含まない青空文庫文と末尾から同数差し替え、合計 100,000 件を厳守する。
- data/prod/sealed-test.jsonl および data/prod/dev.jsonl は一切変更せず、
  sealed-test.jsonl 自体は 1 バイトも開かずに、母集団全件 (aozora_prod_clean + qwen3_prod_verified) との
  完全一致・LSH (Jaccard >= 0.80) 0 重複を数学的にアサート検証する。
"""
import argparse
import json
import os
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_reading as er  # noqa: E402
import homograph_catalog as hc  # noqa: E402
from lexicon_util import LexiconValidator  # noqa: E402
import phonology as ph  # noqa: E402
from span_extractor import SpanExtractor  # noqa: E402
import synth_qwen3 as sq  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MASTER_DIR = ROOT / "data" / "master"
RAW_DIR = ROOT / "data" / "raw"
PROD_DIR = ROOT / "data" / "prod"
PILOT_DIR = ROOT / "data" / "pilot"

_NORM_STRIP_RE = re.compile(r"[、。「」『』（）！？…・\s　]")
_NUM_PERM = 64
_BANDS = 16
_ROWS = 4

CORE_12 = [
    ("一日", "ツイタチ"),
    ("十分", "ジュップン"),
    ("一分", "イップン"),
    ("大事", "オオゴト"),
    ("大家", "タイカ"),
    ("上手", "ウワテ"),
    ("御前", "オマエ"),
    ("弾+く", "ハジ"),
    ("止+める", "ヤ"),
    ("解+く", "ホド"),
    ("退+く", "ノ"),
    ("何時", "ナンジ"),
]


def norm_for_dedup(text: str) -> str:
    return _NORM_STRIP_RE.sub("", text)


def char_5grams(norm_txt: str) -> Set[str]:
    if len(norm_txt) <= 5:
        return {norm_txt}
    return {norm_txt[i : i + 5] for i in range(len(norm_txt) - 4)}


def compute_minhash(grams: Set[str]) -> Tuple[int, ...]:
    vals = [zlib.crc32(g.encode("utf-8")) & 0xFFFFFFFF for g in grams]
    sig = []
    for seed in range(_NUM_PERM):
        a = (seed * 2654435761 + 1) & 0xFFFFFFFF
        b = (seed * 2246822519 + 13) & 0xFFFFFFFF
        sig.append(min(((v * a + b) & 0xFFFFFFFF) for v in vals))
    return tuple(sig)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


class FastLSHChecker:
    """正規化完全一致＋5-gram MinHash LSH (Jaccard >= 0.80) 重複検査器。"""

    def __init__(self, texts: List[str]) -> None:
        self.exact_norms: Set[str] = set()
        self.grams_list: List[Set[str]] = []
        self.buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
        for txt in texts:
            nt = norm_for_dedup(txt)
            if not nt or nt in self.exact_norms:
                continue
            self.exact_norms.add(nt)
            g = char_5grams(nt)
            sig = compute_minhash(g)
            idx = len(self.grams_list)
            self.grams_list.append(g)
            for bi in range(_BANDS):
                self.buckets[(bi, sig[bi * _ROWS : (bi + 1) * _ROWS])].append(idx)

    def check_overlap(self, other_texts: List[str]) -> Tuple[int, int]:
        """(exact_norm_overlap, near_dup_lsh_080) を返す。"""
        exact_ov = 0
        near_ov = 0
        for txt in other_texts:
            nt = norm_for_dedup(txt)
            if nt in self.exact_norms:
                exact_ov += 1
                continue
            g = char_5grams(nt)
            sig = compute_minhash(g)
            cands: Set[int] = set()
            for bi in range(_BANDS):
                cands.update(self.buckets.get((bi, sig[bi * _ROWS : (bi + 1) * _ROWS]), ()))
            for idx in cands:
                if jaccard(g, self.grams_list[idx]) >= 0.80:
                    near_ov += 1
                    break
        return exact_ov, near_ov


def load_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    val = LexiconValidator(MASTER_DIR)
    extractor = SpanExtractor(MASTER_DIR)
    catalog_keys: Set[str] = {item["key"] for item in hc.HOMOGRAPH_CATALOG}

    topup_path = RAW_DIR / "qwen3_prod_topup_verified.jsonl"
    if not topup_path.exists():
        raise FileNotFoundError(f"補完検証済みファイルが見つかりません: {topup_path}")
    topup_raw = load_jsonl(topup_path)
    if not topup_raw:
        raise RuntimeError("補完検証済みデータが 0 件です。")

    # 1. topup 全件のスパン・オフセット・辞書検証＆trainスキーマ変換
    topup_clean: List[dict] = []
    seen_topup_norms: Set[str] = set()
    for i, r in enumerate(topup_raw):
        text = r["text"]
        nt = norm_for_dedup(text)
        if nt in seen_topup_norms:
            continue
        seen_topup_norms.add(nt)

        covered = [False] * len(text)
        clean_spans = []
        for sp in r["spans"]:
            st, ed, surf = int(sp["start"]), int(sp["end"]), sp["surface"]
            assert text[st:ed] == surf, f"オフセット不一致: {text[st:ed]!r} != {surf!r}"
            for p in range(st, ed):
                covered[p] = True
            rk = ph.normalize_kana(sp.get("reading_kata") or sp["reading"])
            lead = text[max(0, st - 2) : st]
            trail = text[ed : ed + 4]
            assert val.verify_span_reading(surf, rk, trailing_text=trail, leading_text=lead), (
                f"辞書未検証スパン検出: {surf} / {rk} in {text}"
            )
            clean_spans.append(
                {
                    "start": st,
                    "end": ed,
                    "surface": surf,
                    "reading": ph.kata_to_hira(rk),
                    "reading_kata": rk,
                    "pron": val.to_pron(rk, surface=surf),
                }
            )
        assert not any(
            ph.is_kanji_char(text[p]) and not covered[p] for p in range(len(text))
        ), f"未カバー漢字検出: {text}"

        topup_clean.append(
            {
                "id": r["id"],
                "split": "train",
                "source": "qwen3_synth",
                "license": "Apache-2.0",
                "group_id": f"topup_{i:06d}",
                "text": text,
                "spans": clean_spans,
            }
        )
    val.close()
    print(f"Verified topup records to insert: {len(topup_clean)}", flush=True)

    # 2. 既存 train.jsonl 読み込み（冪等チェック：既に qwen3_pt_ があれば異常として停止）
    train_path = PROD_DIR / "train.jsonl"
    orig_train = load_jsonl(train_path)
    assert len(orig_train) == 100000, f"train.jsonl の行数が 100,000 ではありません: {len(orig_train)}"
    existing_pt = sum(1 for r in orig_train if str(r.get("id", "")).startswith("qwen3_pt_"))
    if existing_pt > 0:
        raise RuntimeError(f"train.jsonl には既に {existing_pt} 件の qwen3_pt_ が含まれています。二重適用を防止するため停止します。")

    # 3. カタログ同形異音語 (>=2候補) を1つも含まない青空文庫文のインデックスを末尾から抽出
    non_cat_aozora_idxs: List[int] = []
    for idx, r in enumerate(orig_train):
        if r.get("source") != "aozora":
            continue
        ext_sps = extractor.extract_spans(r["text"])
        has_cat = any(sp["key"] in catalog_keys and len(sp["candidates"]) >= 2 for sp in ext_sps)
        if not has_cat:
            non_cat_aozora_idxs.append(idx)

    assert len(non_cat_aozora_idxs) >= len(topup_clean), (
        f"差し替え可能な青空文庫文不足: {len(non_cat_aozora_idxs)} < {len(topup_clean)}"
    )
    drop_set = set(non_cat_aozora_idxs[-len(topup_clean) :])
    kept_rows = [r for idx, r in enumerate(orig_train) if idx not in drop_set]
    new_train = kept_rows + topup_clean
    random.Random(args.seed).shuffle(new_train)
    assert len(new_train) == 100000, f"差し替え後の行数が 100,000 ではありません: {len(new_train)}"
    print(f"Replaced {len(drop_set)} non-catalog aozora rows (pool={len(non_cat_aozora_idxs)}). New train={len(new_train)}", flush=True)

    # 4. 厳格非重複・0リーク検証（sealed-test.jsonl は一切開かない！）
    print("Running zero-leakage & non-overlap LSH assertions...", flush=True)
    # 4a. topup vs 母集団 (qwen3_prod_verified + aozora_prod_clean)
    topup_texts = [r["text"] for r in topup_clean]
    pop_texts = [
        r["text"]
        for p in (RAW_DIR / "qwen3_prod_verified.jsonl", RAW_DIR / "aozora_prod_clean.jsonl")
        for r in load_jsonl(p)
    ]
    pop_checker = FastLSHChecker(pop_texts)
    ex_pop, nr_pop = pop_checker.check_overlap(topup_texts)
    assert ex_pop == 0 and nr_pop == 0, f"topup vs prod raw population overlap: exact={ex_pop}, near={nr_pop}"
    print(f"  topup vs raw_population (aozora_prod + qwen3_prod, n={len(pop_texts)}): exact=0, near_lsh080=0 [PASS]", flush=True)

    # 4b. new_train vs dev & pilot splits
    train_texts = [r["text"] for r in new_train]
    train_checker = FastLSHChecker(train_texts)
    leak_report = {"topup_vs_raw_population": {"exact_norm": ex_pop, "near_dup_lsh_080": nr_pop}}
    for label, path in [
        ("prod_dev", PROD_DIR / "dev.jsonl"),
        ("pilot_homograph_eval", PILOT_DIR / "homograph-eval.jsonl"),
        ("pilot_train", PILOT_DIR / "train.jsonl"),
        ("pilot_dev", PILOT_DIR / "dev.jsonl"),
    ]:
        other_texts = [r["text"] for r in load_jsonl(path)]
        ex_ov, nr_ov = train_checker.check_overlap(other_texts)
        leak_report[f"new_train_vs_{label}"] = {"exact_norm": ex_ov, "near_dup_lsh_080": nr_ov}
        assert ex_ov == 0 and nr_ov == 0, f"リーク検出 (new_train vs {label}): exact={ex_ov}, near={nr_ov}"
        print(f"  new_train vs {label:20s}: exact=0, near_lsh080=0 [PASS]", flush=True)

    # 5. train.jsonl 原子的上書き保存
    tmp_path = PROD_DIR / "train.jsonl.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in new_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_path, train_path)

    # 6. new_train 内の 135 カタログ語・277 有効ターゲットおよびコア 12 ターゲット集計
    valid_targets = []
    for item in hc.HOMOGRAPH_CATALOG:
        vr = [ri for ri in item["readings"] if (item["surface"], ri["reading"]) not in sq._SKIP_SENSES]
        if len(vr) >= 2:
            for ri in vr:
                valid_targets.append((item["key"], ri["reading"]))

    target_cnt: Counter = Counter()
    with_cat = 0
    with_extra = 0
    total_spans = 0
    works: Set[str] = set()
    by_source: Counter = Counter()
    lengths: List[int] = []

    for r in new_train:
        by_source[r["source"]] += 1
        if r.get("work_id"):
            works.add(r["work_id"])
        lengths.append(len(r["text"]))
        total_spans += len(r["spans"])

        ext_sps = extractor.extract_spans(r["text"])
        has_c = False
        has_e = False
        for sp in ext_sps:
            if len(sp["candidates"]) < 2:
                continue
            if sp["key"] in catalog_keys:
                has_c = True
                got = er.extract_subspan_reading(r["spans"], sp["start"], sp["end"])
                if got is not None:
                    rk = ph.normalize_kana(got[0])
                    if rk in sp["candidates"]:
                        target_cnt[(sp["key"], rk)] += 1
            else:
                has_e = True
        if has_c:
            with_cat += 1
        if has_e:
            with_extra += 1

    lengths.sort()
    zero_targets = [(k, rd) for (k, rd) in valid_targets if target_cnt[(k, rd)] == 0]
    core_12_counts = {f"{k}[{rd}]": target_cnt[(k, rd)] for k, rd in CORE_12}
    print("\n=== Core 12 targets counts in new train (100k) ===", flush=True)
    for name, c in core_12_counts.items():
        print(f"  {name:16s}: {c}", flush=True)
    print(
        f"Total valid catalog targets (277): zero_in_train={len(zero_targets)}, "
        f"zero_list={zero_targets}",
        flush=True,
    )

    # 7. build-stats.json と DATA-DESIGN-prod.md 更新
    stats_path = PROD_DIR / "build-stats.json"
    with open(stats_path, encoding="utf-8") as f:
        bstats = json.load(f)

    bstats["splits"]["train"] = {
        "total_rows": len(new_train),
        "by_source": dict(by_source),
        "total_spans": total_spans,
        "with_catalog_homograph_rows": with_cat,
        "with_extra_multi_rows": with_extra,
        "unique_works": len(works),
        "char_len": {
            "min": lengths[0],
            "median": lengths[len(lengths) // 2],
            "mean": round(sum(lengths) / len(lengths), 2),
            "max": lengths[-1],
        },
    }
    bstats["topup_applied"] = {
        "applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topup_qwen3_added": len(topup_clean),
        "non_catalog_aozora_replaced": len(drop_set),
        "non_overlap_and_leak_checks": leak_report,
        "valid_catalog_targets_total": len(valid_targets),
        "valid_catalog_targets_zero_in_train": len(zero_targets),
        "core_12_targets_in_train": core_12_counts,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(bstats, f, ensure_ascii=False, indent=2)

    design_path = ROOT / "DATA-DESIGN-prod.md"
    if design_path.exists():
        cur_md = design_path.read_text(encoding="utf-8")
        section_hdr = "## 4. 欠落ターゲット Qwen3 補完（v2 差分適用）"
        if section_hdr not in cur_md:
            note = (
                f"\n\n{section_hdr}\n\n"
                f"- **補完スクリプト**：`code/topup_qwen3_prod.py` / "
                f"`code/apply_prod_topup.py`\n"
                f"- **適用内容**：既存 `qwen3_prod_verified.jsonl` で 6 件未満だった欠落・極少カタログターゲットに対し、`qwen3:14b` で新規合成・独立逆検証・全スパン辞書検証を通過した **`{len(topup_clean)}` 件** を、同数の「カタログ同形異音語を含まない青空文庫文」と `data/prod/train.jsonl` 内で安全に差し替え（合計 `100,000` 件厳守：`aozora {by_source['aozora']:,}` 件 / `qwen3_synth {by_source['qwen3_synth']:,}` 件）。\n"
                f"- **完全非重複保証**：`prod/sealed-test.jsonl` および `prod/dev.jsonl` は一切変更・開封せず、補完分と本番母集団全体（`aozora_prod_clean` + `qwen3_prod_verified`）・`prod/dev`・パイロット各分割の完全一致・5-gram MinHash LSH (`>= 0.80`) 重複 **`0`** を再検証済み。\n"
                f"- **有効ターゲット網羅状況**：全 `{len(valid_targets)}` ターゲット中 `train` 内 0 件は `{len(zero_targets)}` 件に改善。\n"
            )
            design_path.write_text(cur_md + note, encoding="utf-8")

    print(f"DONE apply_prod_topup in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
