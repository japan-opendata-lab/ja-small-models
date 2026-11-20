#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 青空文庫 クリーン実文収集スクリプト。

SPEC-v1.md 準拠の厳格フィルタ:
- 公式カタログ (list_person_all_extended_utf8.zip) で:
  作品著作権フラグ == "なし"
  全関連人物で 人物著作権フラグ in ("なし", "")
  全関連人物の 役割フラグ に "翻訳者" を含まない
  原題 == ""
  文字遣い種別 == "新字新仮名"
  テキストファイルURL に "_ruby_" を含む .zip
- 文レベル厳格検証:
  1) 文字数 15〜120、漢字スパン >= 2、英数字・特殊記号・外字欠落・旧仮名(ゐゑゝゞ)なし
  2) 漢字の人名 (固有名詞-人名) を含まない
  3) 明示ルビはすべて LexiconValidator (UniDic ∪ SudachiDict) に実在する読みと完全一致
     （義訓・当て字が1つでもあれば文ごと棄却）
  4) 非ルビ漢字部分は Sudachi Mode C と UniDic の読みが100%一致し、辞書検証通過
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
import homograph_catalog as hc  # noqa: E402
from lexicon_util import LexiconValidator  # noqa: E402
import phonology as ph  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AOZORA_DIR = ROOT / "data" / "raw" / "aozora"
WORKS_DIR = AOZORA_DIR / "works"
MASTER_DIR = ROOT / "data" / "master"
OUT_JSONL = ROOT / "data" / "raw" / "aozora_clean.jsonl"
OUT_STATS = ROOT / "data" / "raw" / "aozora_stats.json"

# 青空注記・特殊記号除去
_NOTE_RE = re.compile(r"［＃[^］\n]*］")
_RUBY_PIPE_RE = re.compile(r"｜([^｜《》\n]{1,20})《([^《》\n]{1,30})》")
_RUBY_AUTO_RE = re.compile(r"([一-龥々ヶヵ〆〇]+)《([^《》\n]{1,30})》")
_ALLOWED_PLAIN_RE = re.compile(
    r"^[ぁ-んァ-ヶー一-龥々ヶヵ〆〇、。「」『』（）！？…・　 ]+$"
)
_OLD_KANA_RE = re.compile(r"[ゐゑヰヱゝゞヽヾ／＼※]")
_ARCHAIC_ATEJI_RE = re.compile(r"(?:何う|斯う|斯く|其れ|其の|此れ|此の|彼処|何処|些と)")


def load_clean_catalog(zip_path: Path) -> List[dict]:
    """カタログから著作権切れ・非翻訳・新字新仮名・ルビ付き作品リストを抽出する。"""
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        rows = list(csv.DictReader(io.TextIOWrapper(zf.open(csv_name), encoding="utf-8-sig")))

    by_work: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_work[r["作品ID"]].append(r)

    clean: List[dict] = []
    for wid, wr in by_work.items():
        r0 = wr[0]
        roles = {x.get("役割フラグ", "") for x in wr}
        url = r0.get("テキストファイルURL", "")
        if (
            r0.get("作品著作権フラグ") == "なし"
            and all(x.get("人物著作権フラグ", "なし") in ("なし", "") for x in wr)
            and "翻訳者" not in roles
            and not r0.get("原題", "").strip()
            and r0.get("文字遣い種別") == "新字新仮名"
            and "_ruby_" in url
            and url.endswith(".zip")
        ):
            clean.append(
                {
                    "work_id": wid,
                    "title": r0.get("作品名", ""),
                    "author_id": r0.get("人物ID", ""),
                    "author": (r0.get("姓", "") + r0.get("名", "")),
                    "url": url,
                }
            )
    return clean


def select_diverse_works(works: List[dict], max_works: int, max_per_author: int, seed: int) -> List[dict]:
    """著者多様性を保ちつつ作品をサンプリングする。"""
    rng = random.Random(seed)
    shuffled = list(works)
    rng.shuffle(shuffled)
    author_counts: Counter = Counter()
    picked: List[dict] = []
    for w in shuffled:
        aid = w["author_id"]
        if author_counts[aid] >= max_per_author:
            continue
        author_counts[aid] += 1
        picked.append(w)
        if len(picked) >= max_works:
            break
    return picked


def download_work_text(work: dict, delay_sec: float) -> Optional[str]:
    """作品 zip をローカルキャッシュまたは青空文庫から取得し本文文字列を返す。"""
    WORKS_DIR.mkdir(parents=True, exist_ok=True)
    local_zip = WORKS_DIR / f"{work['work_id']}.zip"
    if not local_zip.exists() or local_zip.stat().st_size == 0:
        time.sleep(delay_sec)
        req = urllib.request.Request(work["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            local_zip.write_bytes(r.read())
    try:
        with zipfile.ZipFile(local_zip) as zf:
            txts = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if not txts:
                return None
            raw = zf.read(txts[0])
            return raw.decode("cp932", errors="replace")
    except Exception:
        return None


def extract_body_sentences(raw_text: str) -> List[str]:
    """青空文庫ヘッダ/フッタ・注記を除去し、文単位リストへ分割する。"""
    parts = re.split(r"-{20,}", raw_text)
    body = parts[2] if len(parts) >= 3 else raw_text
    body = body.split("底本：")[0]
    body = body.replace("\r\n", "\n").replace("\r", "\n")

    sents: List[str] = []
    for para in body.split("\n"):
        para = para.strip()
        if not para:
            continue
        # 注記を含む段落は外字注記の危険があるため注記削除前に外字記号「※」がある行をスキップ
        if "※" in para or "［＃" in para:
            # 傍点や字下げ等も混ざるが、安全のため外字系は除外し注記除去
            if "※［＃" in para:
                continue
            para = _NOTE_RE.sub("", para).strip()
        if not para:
            continue
        # 句点「。」で分割（ただし「。」直後が閉じ括弧なら含める）
        cur = ""
        for ch in para:
            cur += ch
            if ch == "。":
                s = cur.strip()
                if s:
                    sents.append(s)
                cur = ""
        if cur.strip() and cur.strip().endswith(("」", "』", "！", "？")):
            sents.append(cur.strip())
    return sents


def parse_aozora_rubies(sent: str) -> Optional[Tuple[str, List[dict]]]:
    """青空ルビ記号（｜基底《ルビ》・漢字《ルビ》）をパースし (プレーン文, 明示ルビ一覧) を返す。"""
    if "《" not in sent and "》" not in sent and "｜" not in sent:
        return sent, []

    # まず ｜基底《ルビ》 と 漢字《ルビ》 の両方に一致する統合正規表現で左からスキャン
    # パターン: (?:｜([^｜《》\n]{1,20})《([^《》\n]{1,30})》)|([一-龥々ヶヵ〆〇]+)《([^《》\n]{1,30})》)
    pat = re.compile(
        r"｜([^｜《》\n]{1,20})《([^《》\n]{1,30})》|([一-龥々ヶヵ〆〇]+)《([^《》\n]{1,30})》"
    )
    parts: List[str] = []
    rubies: List[dict] = []
    pos = 0
    cur_len = 0
    for m in pat.finditer(sent):
        if m.start() > pos:
            chunk = sent[pos : m.start()]
            parts.append(chunk)
            cur_len += len(chunk)
        if m.group(1) is not None:
            base, ruby = m.group(1), m.group(2)
        else:
            base, ruby = m.group(3), m.group(4)

        # ルビ基底に漢字が含まれない場合（かなに外来語ルビ等）は対象外なので棄却
        if not ph.has_kanji(base):
            return None
        # 基底内にひらがなや記号が混ざる「｜あした《明日》」や「｜漢字とかな《...》」は厳格除外（漢字塊または送り仮名剥離可能なもののみ）
        st = cur_len
        parts.append(base)
        cur_len += len(base)
        rubies.append({"start": st, "end": cur_len, "surface": base, "ruby": ruby})
        pos = m.end()

    if pos < len(sent):
        parts.append(sent[pos:])
    plain = "".join(parts)
    if "《" in plain or "》" in plain or "｜" in plain:
        return None
    return plain, rubies


def _collect_kanji_blocks_agreement(
    plain: str,
    rubied_mask: List[bool],
    sudachi_tok,
    unidic_tagger,
    val: LexiconValidator,
    protected: Set[str],
) -> Optional[List[dict]]:
    """非ルビ漢字部分について Sudachi C と UniDic の一致・非人名・辞書検証を行い、スパンを返す。"""
    # 1. Sudachi C 解析
    su_spans: List[Tuple[int, int, str, str]] = []
    for m in sudachi_tok.tokenize(plain, tokenizer.Tokenizer.SplitMode.C):
        surf = m.surface()
        if not ph.has_kanji(surf):
            continue
        pos = m.part_of_speech()
        if pos[1] == "固有名詞" and pos[2] == "人名" and surf not in protected:
            return None
        rf = m.reading_form()
        if not rf or rf == "*":
            return None
        rk = ph.normalize_kana(rf)
        if not ph.is_valid_reading_str(rk):
            return None
        stripped = ph.strip_okurigana(surf, rk)
        if stripped is None:
            return None
        mst = m.begin()
        for rs, re_, ks, kr in stripped:
            abs_s, abs_e = mst + rs, mst + re_
            su_spans.append((abs_s, abs_e, ks, kr))

    # 2. UniDic 解析
    ud_spans: List[Tuple[int, int, str, str]] = []
    cur_p = 0
    for w in unidic_tagger(plain):
        surf = w.surface
        idx = plain.find(surf, cur_p)
        if idx < 0:
            return None
        cur_p = idx + len(surf)
        if not ph.has_kanji(surf):
            continue
        if w.feature.pos2 == "固有名詞" and w.feature.pos3 == "人名" and surf not in protected:
            return None
        if not w.feature.kana:
            return None
        rk = ph.normalize_kana(w.feature.kana)
        if not ph.is_valid_reading_str(rk):
            return None
        stripped = ph.strip_okurigana(surf, rk)
        if stripped is None:
            return None
        for rs, re_, ks, kr in stripped:
            abs_s, abs_e = idx + rs, idx + re_
            ud_spans.append((abs_s, abs_e, ks, kr))

    # 3. 非ルビ領域のみで両者のスパン列を連続漢字ブロック単位に照合
    su_unrubied = []
    for s, e, ks, kr in su_spans:
        has_r = any(rubied_mask[p] for p in range(s, e))
        all_r = all(rubied_mask[p] for p in range(s, e))
        if has_r and not all_r:
            return None  # ルビ境界が形態素途中で切れている安全除外
        if not has_r:
            su_unrubied.append((s, e, ks, kr))

    ud_unrubied = []
    for s, e, ks, kr in ud_spans:
        has_r = any(rubied_mask[p] for p in range(s, e))
        all_r = all(rubied_mask[p] for p in range(s, e))
        if has_r and not all_r:
            return None
        if not has_r:
            ud_unrubied.append((s, e, ks, kr))

    def to_blocks(items: List[Tuple[int, int, str, str]]) -> List[Tuple[int, int, str, str]]:
        if not items:
            return []
        blocks: List[Tuple[int, int, str, str]] = []
        bs, be, bk, br = items[0]
        for s, e, ks, kr in items[1:]:
            if s == be:
                be = e
                bk += ks
                br += kr
            else:
                blocks.append((bs, be, bk, br))
                bs, be, bk, br = s, e, ks, kr
        blocks.append((bs, be, bk, br))
        return blocks

    su_blocks = to_blocks(su_unrubied)
    ud_blocks = to_blocks(ud_unrubied)
    if len(su_blocks) != len(ud_blocks):
        return None
    for (s1, e1, k1, r1), (s2, e2, k2, r2) in zip(su_blocks, ud_blocks):
        if s1 != s2 or e1 != e2 or k1 != k2 or r1 != r2:
            return None

    # Sudachi 粒度の非ルビスパン（すでに両者でブロック読み完全一致検証済み）を返す
    out: List[dict] = []
    for s, e, ks, kr in su_unrubied:
        lead = plain[max(0, s - 2) : s]
        trail = plain[e : e + 4]
        if not val.verify_span_reading(ks, kr, trailing_text=trail, leading_text=lead):
            return None
        out.append(
            {
                "start": s,
                "end": e,
                "surface": ks,
                "reading": ph.kata_to_hira(kr),
                "reading_kata": kr,
                "pron": val.to_pron(kr, surface=ks),
                "from_ruby": False,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-works", type=int, default=320)
    parser.add_argument("--max-per-author", type=int, default=6)
    parser.add_argument("--max-sents-per-work", type=int, default=220)
    parser.add_argument("--target-total", type=int, default=30000)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    cat_zip = AOZORA_DIR / "list_person_all_extended_utf8.zip"
    clean_works = load_clean_catalog(cat_zip)
    selected = select_diverse_works(
        clean_works, max_works=args.max_works, max_per_author=args.max_per_author, seed=args.seed
    )
    print(f"Clean catalog works: {len(clean_works)}, selected: {len(selected)}")

    val = LexiconValidator(MASTER_DIR)
    sudachi_tok = dictionary.Dictionary(dict="core").tokenizer()
    unidic_tagger = fugashi.Tagger()

    protected: Set[str] = {item["surface"] for item in hc.HOMOGRAPH_CATALOG} | {
        item["span_surface"] for item in hc.HOMOGRAPH_CATALOG
    }
    # カタログの (span_surface, reading_kata) ペア集合（明確な同形異音語ルビ判定用）
    catalog_pair_set: Set[Tuple[str, str]] = set()
    for item in hc.HOMOGRAPH_CATALOG:
        ss = item["span_surface"]
        for rinfo in item["readings"]:
            catalog_pair_set.add((ss, rinfo["reading"]))
    catalog_surfaces: Set[str] = {item["span_surface"] for item in hc.HOMOGRAPH_CATALOG} | {
        item["surface"] for item in hc.HOMOGRAPH_CATALOG
    }

    counts = Counter()
    seen_texts: Set[str] = set()
    records: List[dict] = []

    for wi, w in enumerate(selected, 1):
        raw_txt = download_work_text(w, delay_sec=args.delay)
        if not raw_txt:
            counts["download_failed"] += 1
            continue
        counts["works_processed"] += 1
        sents = extract_body_sentences(raw_txt)
        work_accepted = 0

        for si, sent in enumerate(sents):
            if work_accepted >= args.max_sents_per_work:
                break
            counts["raw_sentences"] += 1

            parsed = parse_aozora_rubies(sent)
            if parsed is None:
                counts["bad_ruby_markup"] += 1
                continue
            plain, raw_rubies = parsed

            if not (15 <= len(plain) <= 120):
                counts["bad_length"] += 1
                continue
            if _OLD_KANA_RE.search(plain) or _ARCHAIC_ATEJI_RE.search(plain) or ph.PERSON_EXCLUDE_RE.search(plain):
                counts["old_kana_or_alnum"] += 1
                continue
            if plain.count("「") != plain.count("」") or plain.count("『") != plain.count("』"):
                counts["unmatched_quotes"] += 1
                continue
            if not _ALLOWED_PLAIN_RE.match(plain):
                counts["disallowed_chars"] += 1
                continue
            if plain in seen_texts:
                counts["duplicate"] += 1
                continue

            # 明示ルビのスパン処理と辞書検証
            ruby_spans: List[dict] = []
            rubied_mask = [False] * len(plain)
            ruby_ok = True
            explicit_homograph_targets: List[dict] = []

            for rb in raw_rubies:
                st, ed, base, r_raw = rb["start"], rb["end"], rb["surface"], rb["ruby"]
                # もし明示ルビ基底に送り仮名・ひらがなが含まれていたら剥離する
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

            # 非ルビ漢字の検証（Sudachi == UniDic かつ辞書検証）
            unrubied_spans = _collect_kanji_blocks_agreement(
                plain, rubied_mask, sudachi_tok, unidic_tagger, val, protected
            )
            if unrubied_spans is None:
                counts["unrubied_disagree_or_person"] += 1
                continue

            all_spans = sorted(ruby_spans + unrubied_spans, key=lambda x: x["start"])
            if len(all_spans) < 2:
                counts["too_few_spans"] += 1
                continue

            # 原文の全漢字がスパンでカバーされていることを検証
            covered = [False] * len(plain)
            for sp in all_spans:
                for p in range(sp["start"], sp["end"]):
                    covered[p] = True
            if any(ph.is_kanji_char(plain[p]) and not covered[p] for p in range(len(plain))):
                counts["uncovered_kanji"] += 1
                continue

            # カタログ同形異音語が含まれるか
            matched_homographs = sorted(
                {sp["surface"] for sp in all_spans if sp["surface"] in catalog_surfaces}
            )
            has_explicit_homo = len(explicit_homograph_targets) > 0

            seen_texts.add(plain)
            counts["accepted"] += 1
            if has_explicit_homo:
                counts["accepted_explicit_homograph_ruby"] += 1
            if matched_homographs:
                counts["accepted_with_any_homograph"] += 1

            work_accepted += 1
            records.append(
                {
                    "id": f"aozora_{w['work_id']}_{si:05d}",
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
                }
            )

        if wi % 40 == 0 or len(records) >= args.target_total:
            print(
                f"Progress works {wi}/{len(selected)}: accepted={len(records)} "
                f"(explicit_homo_ruby={counts['accepted_explicit_homograph_ruby']})"
            )
        if len(records) >= args.target_total:
            break

    val.close()
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "counts": dict(counts),
        "unique_works_used": len({r["work_id"] for r in records}),
        "unique_authors_used": len({r["author_id"] for r in records}),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Aozora collection complete:", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
