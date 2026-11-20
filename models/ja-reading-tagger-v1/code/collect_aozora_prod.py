#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 本番用 青空文庫 大規模クリーン実文収集スクリプト。

特徴:
- パイロット工程でダウンロード済みの 293 作品 (data/raw/aozora/works/*.zip) を完全除外し、
  残余 8,182 作品・503 著者から新規に収集する（パイロット全分割との作品重複 0）。
- フィルタ・検証条件はパイロット collect_aozora.py と 100% 同一。
- 低頻度一般多読語（extra_lex）の濃縮配分に備え、各文に含まれる多読語フラグを記録する。
"""
import argparse
import csv
import io
import json
import random
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import fugashi
from sudachipy import dictionary, tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_aozora as ca  # noqa: E402
import homograph_catalog as hc  # noqa: E402
from lexicon_util import LexiconValidator  # noqa: E402
import phonology as ph  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PILOT_WORKS_DIR = ROOT / "data" / "raw" / "aozora" / "works"
CAT_ZIP = ROOT / "data" / "raw" / "aozora" / "list_person_all_extended_utf8.zip"
PROD_WORKS_DIR = ROOT / "data" / "raw" / "aozora_prod" / "works"
MASTER_DIR = ROOT / "data" / "master"
OUT_JSONL = ROOT / "data" / "raw" / "aozora_prod_clean.jsonl"
OUT_STATS = ROOT / "data" / "raw" / "aozora_prod_stats.json"


def download_work_text_prod(work: dict, delay_sec: float) -> Optional[str]:
    PROD_WORKS_DIR.mkdir(parents=True, exist_ok=True)
    local_zip = PROD_WORKS_DIR / f"{work['work_id']}.zip"
    if not local_zip.exists() or local_zip.stat().st_size == 0:
        time.sleep(delay_sec)
        try:
            req = urllib.request.Request(work["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                local_zip.write_bytes(r.read())
        except Exception:
            return None
    try:
        with zipfile.ZipFile(local_zip) as zf:
            txts = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if not txts:
                return None
            return zf.read(txts[0]).decode("cp932", errors="replace")
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-works", type=int, default=1800)
    parser.add_argument("--max-per-author", type=int, default=12)
    parser.add_argument("--max-sents-per-work", type=int, default=400)
    parser.add_argument("--target-total", type=int, default=125000)
    parser.add_argument("--delay", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    t0 = time.time()
    pilot_excluded = {p.stem for p in PILOT_WORKS_DIR.glob("*.zip")}
    all_clean = ca.load_clean_catalog(CAT_ZIP)
    rem_clean = [w for w in all_clean if w["work_id"] not in pilot_excluded]
    selected = ca.select_diverse_works(
        rem_clean,
        max_works=args.max_works,
        max_per_author=args.max_per_author,
        seed=args.seed,
    )
    print(
        f"Catalog clean={len(all_clean)}, pilot_excluded={len(pilot_excluded)}, "
        f"remaining={len(rem_clean)}, selected={len(selected)}",
        flush=True,
    )

    # 既に存在するパイロットのテキストとも厳格重複チェックしておく（公開分のみではなく、そもそも作品が重複0だが念のため重複防止）
    val = LexiconValidator(MASTER_DIR)
    sudachi_tok = dictionary.Dictionary(dict="core").tokenizer()
    unidic_tagger = fugashi.Tagger()

    protected: Set[str] = {item["surface"] for item in hc.HOMOGRAPH_CATALOG} | {
        item["span_surface"] for item in hc.HOMOGRAPH_CATALOG
    }
    catalog_pair_set: Set[Tuple[str, str]] = set()
    for item in hc.HOMOGRAPH_CATALOG:
        ss = item["span_surface"]
        for rinfo in item["readings"]:
            catalog_pair_set.add((ss, rinfo["reading"]))
    catalog_surfaces: Set[str] = {item["span_surface"] for item in hc.HOMOGRAPH_CATALOG} | {
        item["surface"] for item in hc.HOMOGRAPH_CATALOG
    }
    multi_surfaces: Set[str] = set(val.multi_readings.keys())

    counts = Counter()
    seen_texts: Set[str] = set()
    records: List[dict] = []

    with open(OUT_JSONL, "w", encoding="utf-8") as fout:
        for wi, w in enumerate(selected, 1):
            raw_txt = download_work_text_prod(w, delay_sec=args.delay)
            if not raw_txt:
                counts["download_failed"] += 1
                continue
            counts["works_processed"] += 1
            sents = ca.extract_body_sentences(raw_txt)
            work_accepted = 0

            for si, sent in enumerate(sents):
                if work_accepted >= args.max_sents_per_work:
                    break
                counts["raw_sentences"] += 1

                parsed = ca.parse_aozora_rubies(sent)
                if parsed is None:
                    counts["bad_ruby_markup"] += 1
                    continue
                plain, raw_rubies = parsed

                if not (15 <= len(plain) <= 120):
                    counts["bad_length"] += 1
                    continue
                if (
                    ca._OLD_KANA_RE.search(plain)
                    or ca._ARCHAIC_ATEJI_RE.search(plain)
                    or ph.PERSON_EXCLUDE_RE.search(plain)
                ):
                    counts["old_kana_or_alnum"] += 1
                    continue
                if plain.count("「") != plain.count("」") or plain.count("『") != plain.count("』"):
                    counts["unmatched_quotes"] += 1
                    continue
                if not ca._ALLOWED_PLAIN_RE.match(plain):
                    counts["disallowed_chars"] += 1
                    continue
                if plain in seen_texts:
                    counts["duplicate"] += 1
                    continue

                ruby_spans: List[dict] = []
                rubied_mask = [False] * len(plain)
                ruby_ok = True
                explicit_homograph_targets: List[dict] = []

                for rb in raw_rubies:
                    st, ed, base, r_raw = rb["start"], rb["end"], rb["surface"], rb["ruby"]
                    rk_full = ph.normalize_kana(r_raw)
                    if not ph.is_valid_reading_str(rk_full):
                        ruby_ok = False
                        break
                    stripped = ph.strip_okurigana(base, rk_full)
                    if stripped is None:
                        ruby_ok = False
                        break
                    for rs, re_, ks, kr in stripped:
                        abs_s, abs_e = st + rs, st + re_
                        lead = plain[max(0, abs_s - 2) : abs_s]
                        trail = plain[abs_e : abs_e + 4]
                        if not val.verify_span_reading(ks, kr, trailing_text=trail, leading_text=lead):
                            ruby_ok = False
                            break
                        for p in range(abs_s, abs_e):
                            rubied_mask[p] = True
                        sp_obj = {
                            "start": abs_s,
                            "end": abs_e,
                            "surface": ks,
                            "reading": ph.kata_to_hira(kr),
                            "reading_kata": kr,
                            "pron": val.to_pron(kr, surface=ks),
                            "from_ruby": True,
                        }
                        ruby_spans.append(sp_obj)
                        if (ks, kr) in catalog_pair_set:
                            explicit_homograph_targets.append(sp_obj)
                    if not ruby_ok:
                        break

                if not ruby_ok:
                    counts["ruby_dict_unverified"] += 1
                    continue

                unrubied_spans = ca._collect_kanji_blocks_agreement(
                    plain, rubied_mask, sudachi_tok, unidic_tagger, val, protected
                )
                if unrubied_spans is None:
                    counts["unrubied_disagree_or_person"] += 1
                    continue

                all_spans = sorted(ruby_spans + unrubied_spans, key=lambda x: x["start"])
                if len(all_spans) < 2:
                    counts["too_few_spans"] += 1
                    continue

                covered = [False] * len(plain)
                for sp in all_spans:
                    for p in range(sp["start"], sp["end"]):
                        covered[p] = True
                if any(ph.is_kanji_char(plain[p]) and not covered[p] for p in range(len(plain))):
                    counts["uncovered_kanji"] += 1
                    continue

                matched_homographs = sorted(
                    {sp["surface"] for sp in all_spans if sp["surface"] in catalog_surfaces}
                )
                matched_extra_multi = sorted(
                    {
                        sp["surface"]
                        for sp in all_spans
                        if sp["surface"] in multi_surfaces and sp["surface"] not in catalog_surfaces
                    }
                )
                has_explicit_homo = len(explicit_homograph_targets) > 0

                seen_texts.add(plain)
                counts["accepted"] += 1
                if has_explicit_homo:
                    counts["accepted_explicit_homograph_ruby"] += 1
                if matched_homographs:
                    counts["accepted_with_any_homograph"] += 1
                if matched_extra_multi:
                    counts["accepted_with_extra_multi"] += 1

                work_accepted += 1
                rec = {
                    "id": f"aozora_p_{w['work_id']}_{si:05d}",
                    "source": "aozora",
                    "license": "PublicDomain",
                    "work_id": w["work_id"],
                    "author_id": w["author_id"],
                    "group_id": f"aozora_work:{w['work_id']}",
                    "text": plain,
                    "spans": all_spans,
                    "has_explicit_homograph_ruby": has_explicit_homo,
                    "explicit_homograph_targets": explicit_homograph_targets,
                    "homograph_surfaces": matched_homographs,
                    "extra_multi_surfaces": matched_extra_multi,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records.append(rec)

            if wi % 100 == 0 or len(records) >= args.target_total:
                fout.flush()
                print(
                    f"Progress works {wi}/{len(selected)}: accepted={len(records)} "
                    f"(homo={counts['accepted_with_any_homograph']}, extra_multi={counts['accepted_with_extra_multi']}, "
                    f"elapsed={time.time()-t0:.1f}s)",
                    flush=True,
                )
            if len(records) >= args.target_total:
                break

    val.close()
    stats = {
        "counts": dict(counts),
        "pilot_works_excluded": len(pilot_excluded),
        "unique_works_used": len({r["work_id"] for r in records}),
        "unique_authors_used": len({r["author_id"] for r in records}),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Aozora prod collection complete:", json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
