#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 ROHAN4600 (CC0 1.0) クリーン実文収集スクリプト。

data/raw/rohan4600/Rohan4600_transcript_utf8.txt から漢字スパン付き文を抽出し、
人名（漢字）・英数字を除外した上、すべてのスパンの読みが統合辞書 (UniDic ∪ SudachiDict) で
検証できた文のみ data/raw/rohan_clean.jsonl へ保存する。
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import fugashi
from sudachipy import dictionary, tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import homograph_catalog as hc  # noqa: E402
from lexicon_util import LexiconValidator  # noqa: E402
import phonology as ph  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROHAN_TXT = ROOT / "data" / "raw" / "rohan4600" / "Rohan4600_transcript_utf8.txt"
MASTER_DIR = ROOT / "data" / "master"
OUT_JSONL = ROOT / "data" / "raw" / "rohan_clean.jsonl"
OUT_STATS = ROOT / "data" / "raw" / "rohan_stats.json"

_RUBY_TOKEN_RE = re.compile(r"([^\s\(\)（）《》]+?)\(([^()\n]+)\)")


def parse_rohan_rubied(rubied: str) -> Tuple[str, List[dict]]:
    """'流(なが)し斬(ぎ)り...' 形式から (原文テキスト, 生ルビスパン一覧) を抽出する。"""
    text_parts: List[str] = []
    spans: List[dict] = []
    pos = 0
    cur_len = 0
    # 括弧に直結した漢字塊を厳密マッチ（例: 流(なが) -> surface="流", ruby="なが"）
    # ※もし非漢字+漢字が連結される場合を避けるため、漢字連続のみを抽出する
    pat = re.compile(r"([一-龥々ヶヵ〆〇]+)\(([^()\n]+)\)")
    for m in pat.finditer(rubied):
        if m.start() > pos:
            chunk = rubied[pos : m.start()]
            text_parts.append(chunk)
            cur_len += len(chunk)
        surf = m.group(1)
        ruby = m.group(2)
        st = cur_len
        text_parts.append(surf)
        cur_len += len(surf)
        spans.append({"start": st, "end": cur_len, "surface": surf, "ruby": ruby})
        pos = m.end()
    if pos < len(rubied):
        tail = rubied[pos:]
        text_parts.append(tail)
    return "".join(text_parts), spans


def contains_kanji_person_name(
    text: str, sudachi_tok, unidic_tagger, protected: Set[str]
) -> bool:
    """漢字を含む人名（固有名詞-人名）が文中に含まれるか判定する（カタカナ外来人名は対象外なので許容）。"""
    for m in sudachi_tok.tokenize(text, tokenizer.Tokenizer.SplitMode.C):
        s = m.surface()
        if not ph.has_kanji(s):
            continue
        pos = m.part_of_speech()
        if pos[1] == "固有名詞" and pos[2] == "人名" and s not in protected:
            return True
    for w in unidic_tagger(text):
        s = w.surface
        if not ph.has_kanji(s):
            continue
        if w.feature.pos2 == "固有名詞" and w.feature.pos3 == "人名" and s not in protected:
            return True
    return False


def main() -> None:
    t0 = time.time()
    val = LexiconValidator(MASTER_DIR)
    sudachi_tok = dictionary.Dictionary(dict="core").tokenizer()
    unidic_tagger = fugashi.Tagger()

    protected: Set[str] = {item["surface"] for item in hc.HOMOGRAPH_CATALOG} | {
        item["span_surface"] for item in hc.HOMOGRAPH_CATALOG
    }
    catalog_surfaces: Set[str] = {item["span_surface"] for item in hc.HOMOGRAPH_CATALOG} | {
        item["surface"] for item in hc.HOMOGRAPH_CATALOG
    }

    counts = Counter()
    records: List[dict] = []
    unverified_examples: List[Tuple[str, str, str]] = []

    with open(ROHAN_TXT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            counts["total_lines"] += 1
            uid, rest = line.split(":", 1)
            if "," not in rest:
                counts["bad_format"] += 1
                continue
            rubied, _full_kana = rest.rsplit(",", 1)
            # 漢字以外の文字に対する括弧注記が残っていないか確認
            plain_text, raw_spans = parse_rohan_rubied(rubied)
            if "(" in plain_text or ")" in plain_text:
                counts["unparsed_paren"] += 1
                continue

            if not (10 <= len(plain_text) <= 140):
                counts["bad_length"] += 1
                continue
            if not raw_spans:
                counts["no_spans"] += 1
                continue
            if ph.PERSON_EXCLUDE_RE.search(plain_text):
                counts["has_alnum"] += 1
                continue

            # ルビ漏れ漢字（スパン外の文字に漢字が残っていないか）チェック
            covered = [False] * len(plain_text)
            for sp in raw_spans:
                for p in range(sp["start"], sp["end"]):
                    covered[p] = True
            if any(ph.is_kanji_char(plain_text[p]) and not covered[p] for p in range(len(plain_text))):
                counts["uncovered_kanji"] += 1
                continue

            if contains_kanji_person_name(plain_text, sudachi_tok, unidic_tagger, protected):
                counts["has_person_name"] += 1
                continue

            # 全スパンを辞書検証 & 出力整形
            verified_spans: List[dict] = []
            all_ok = True
            has_homograph = False
            matched_homographs: List[str] = []

            for sp in raw_spans:
                st, ed, surf, ruby_raw = sp["start"], sp["end"], sp["surface"], sp["ruby"]
                rk = ph.normalize_kana(ruby_raw)
                lead = plain_text[max(0, st - 2) : st]
                trail = plain_text[ed : ed + 4]
                if not val.verify_span_reading(surf, rk, trailing_text=trail, leading_text=lead):
                    all_ok = False
                    if len(unverified_examples) < 30:
                        unverified_examples.append((surf, rk, plain_text))
                    break
                pron = val.to_pron(rk, surface=surf)
                verified_spans.append(
                    {
                        "start": st,
                        "end": ed,
                        "surface": surf,
                        "reading": ph.kata_to_hira(rk),
                        "reading_kata": rk,
                        "pron": pron,
                    }
                )
                if surf in catalog_surfaces:
                    has_homograph = True
                    matched_homographs.append(surf)

            if not all_ok:
                counts["dict_unverified"] += 1
                continue

            counts["accepted"] += 1
            if has_homograph:
                counts["accepted_with_homograph"] += 1
            records.append(
                {
                    "id": uid,
                    "source": "rohan4600",
                    "license": "CC0-1.0",
                    "group_id": f"rohan:{uid}",
                    "text": plain_text,
                    "spans": verified_spans,
                    "has_explicit_homograph": has_homograph,
                    "homograph_surfaces": sorted(set(matched_homographs)),
                }
            )

    val.close()
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "counts": dict(counts),
        "elapsed_sec": round(time.time() - t0, 2),
        "sample_unverified": unverified_examples[:15],
    }
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("ROHAN collection complete:", json.dumps(dict(counts), ensure_ascii=False))


if __name__ == "__main__":
    main()
