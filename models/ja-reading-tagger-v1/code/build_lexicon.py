#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 統合クリーン辞書マスター構築スクリプト。

使用データ（すべて商用利用可能・クリーンライセンス）:
- SudachiDict Raw CSV (small_lex.zip / core_lex.zip v20261015, Apache License 2.0)
- UniDic (unidic-lite 2.1.2, 修正 BSD License)

生成物:
- data/master/lexicon.sqlite      : 全漢字見出し語の辞書検証用インデックス
- data/master/reading_assets.json : 軽量オンメモリ辞書（単漢字・多読語・発音変換・同形異音語カタログ）
- data/master/lexicon-stats.json  : 辞書構築統計・カタログ検証結果
"""
import csv
import io
import json
import sqlite3
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import fugashi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import homograph_catalog as hc  # noqa: E402
import phonology as ph  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_SUDACHI_DIR = ROOT / "data" / "raw" / "sudachidict"
MASTER_DIR = ROOT / "data" / "master"

# 語頭連濁ペア判定用（先頭清音 -> 有声音）
_VOICING_PAIR: Dict[str, str] = {
    "カ": "ガ", "キ": "ギ", "ク": "グ", "ケ": "ゲ", "コ": "ゴ",
    "サ": "ザ", "シ": "ジ", "ス": "ズ", "セ": "ゼ", "ソ": "ゾ",
    "タ": "ダ", "チ": "ヂ", "ツ": "ヅ", "テ": "デ", "ト": "ド",
    "ハ": "バ", "ヒ": "ビ", "フ": "ブ", "ヘ": "ベ", "ホ": "ボ",
    "パ": "バ", "ピ": "ビ", "プ": "ブ", "ペ": "ベ", "ポ": "ボ",
}
_RENDAKU_MAP: Dict[str, Tuple[str, ...]] = {
    "カ": ("ガ",), "キ": ("ギ",), "ク": ("グ",), "ケ": ("ゲ",), "コ": ("ゴ",),
    "サ": ("ザ",), "シ": ("ジ",), "ス": ("ズ",), "セ": ("ゼ",), "ソ": ("ゾ",),
    "タ": ("ダ",), "チ": ("ヂ", "ジ"), "ツ": ("ヅ", "ズ"), "テ": ("デ",), "ト": ("ド",),
    "ハ": ("バ", "パ"), "ヒ": ("ビ", "ピ"), "フ": ("ブ", "プ"), "ヘ": ("ベ", "ペ"), "ホ": ("ボ", "ポ"),
}
_SOKUON_TAILS = frozenset("ツチクキ")


def is_simple_rendaku_variant(cand: str, existing_seion_readings: Set[str]) -> bool:
    """同じ語に対して先頭1モーラのみ清音→濁音化した連濁派生形かを判定する。

    注意: 1〜2文字漢語の漢音/呉音対立（コウ/ギョウ、カ/ゲ、セイ/ショウ等）は除外しない。
    """
    if len(cand) < 2:
        return False
    for r in existing_seion_readings:
        if len(r) == len(cand) and r[1:] == cand[1:]:
            if _VOICING_PAIR.get(r[0]) == cand[0]:
                return True
    return False


def main() -> None:
    t0 = time.time()
    MASTER_DIR.mkdir(parents=True, exist_ok=True)

    # 1. SudachiDict CSV (small_lex + core_lex) の読み込み
    # surface -> {reading: (min_cost, pos1, is_indep)}
    entries: Dict[str, Dict[str, Tuple[int, str, bool]]] = defaultdict(dict)
    small_surfaces: Set[str] = set()
    single_kanji: Dict[str, Set[str]] = defaultdict(set)

    stats = Counter()
    for fname in ("small_lex.zip", "core_lex.zip"):
        zpath = RAW_SUDACHI_DIR / fname
        if not zpath.exists():
            raise FileNotFoundError(f"SudachiDict zip がありません: {zpath}")
        is_small = fname.startswith("small")
        with zipfile.ZipFile(zpath) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                for row in reader:
                    stats[f"{fname}_rows"] += 1
                    if len(row) < 13:
                        continue
                    surf = row[0]
                    left_id = row[1]
                    try:
                        cost = int(row[3])
                    except ValueError:
                        cost = 20000
                    pos1, pos2, pos3 = row[5], row[6], row[7]
                    read_raw = row[11]

                    # 人名・記号・空白・無漢字・無読みは除外
                    if pos2 == "固有名詞" and pos3 == "人名":
                        stats["excluded_person"] += 1
                        continue
                    if pos1 in ("記号", "補助記号", "空白"):
                        continue
                    if not ph.has_kanji(surf):
                        continue
                    if not read_raw or read_raw == "*":
                        continue
                    rk = ph.normalize_kana(read_raw)
                    if not ph.is_valid_reading_str(rk):
                        continue

                    # 独立語（left_id != "-1" かつ 接尾辞/接頭辞・固有名詞・数詞でない）
                    is_indep = (
                        (left_id != "-1")
                        and (pos1 not in ("接尾辞", "接頭辞"))
                        and (pos2 not in ("固有名詞", "数詞"))
                    )
                    prev = entries[surf].get(rk)
                    if prev is None:
                        entries[surf][rk] = (cost, pos1, is_indep)
                    else:
                        best_cost = min(prev[0], cost) if is_indep == prev[2] else (cost if is_indep else prev[0])
                        entries[surf][rk] = (best_cost, prev[1], prev[2] or is_indep)

                    if is_small:
                        small_surfaces.add(surf)
                    if len(surf) == 1 and ph.is_kanji_char(surf) and left_id != "-1":
                        single_kanji[surf].add(rk)
                    stats["accepted_kanji_rows"] += 1

    print(f"Sudachi loaded in {time.time()-t0:.1f}s: unique_kanji_surfaces={len(entries)}")

    # 2. UniDic (unidic-lite) による補完と pron_map 構築
    # small_lex の標準語（および全1〜2文字語、カタログ語）約10万語を UniDic に通す
    t1 = time.time()
    tagger = fugashi.Tagger()
    pron_surface_map: Dict[str, str] = {}
    pron_kana_candidates: Dict[str, Set[str]] = defaultdict(set)

    probe_surfaces = set(hc.HOMOGRAPH_CATALOG[i]["surface"] for i in range(len(hc.HOMOGRAPH_CATALOG)))
    probe_surfaces |= {s for s in entries if len(s) <= 2 or s in small_surfaces}

    unidic_added = 0
    for surf in probe_surfaces:
        for path in tagger.nbestToNodeList(surf, 15):
            if len(path) != 1 or path[0].surface != surf:
                continue
            w = path[0]
            feat = w.feature
            if feat.pos2 == "固有名詞" and feat.pos3 == "人名":
                continue
            if feat.pos1 in ("記号", "補助記号", "空白"):
                continue
            if not feat.kana:
                continue
            rk = ph.normalize_kana(feat.kana)
            if not ph.is_valid_reading_str(rk):
                continue
            if rk not in entries[surf]:
                is_indep = (feat.pos1 not in ("接尾辞", "接頭辞")) and (
                    feat.pos2 not in ("固有名詞", "数詞")
                )
                entries[surf][rk] = (9500, feat.pos1, is_indep)
                unidic_added += 1
            if len(surf) == 1 and ph.is_kanji_char(surf):
                single_kanji[surf].add(rk)
            if feat.pron:
                pr = ph.normalize_kana(feat.pron)
                if pr != ph.rule_kana_to_pron(rk):
                    pron_surface_map[f"{surf}\t{rk}"] = pr
                pron_kana_candidates[rk].add(pr)

    # 一意に決まる kana -> pron かつ rule_kana_to_pron と異なるものを登録
    pron_map: Dict[str, str] = dict(pron_surface_map)
    for rk, pr_set in pron_kana_candidates.items():
        if len(pr_set) == 1:
            pr = next(iter(pr_set))
            if pr != ph.rule_kana_to_pron(rk):
                pron_map[rk] = pr

    print(f"UniDic probe done in {time.time()-t1:.1f}s: added={unidic_added}, pron_exceptions={len(pron_map)}")

    # 3. 複合語検証関数（カタログチェック用）
    single_sets = {k: set(v) for k, v in single_kanji.items()}

    def verify_compound(surf: str, r_kata: str) -> bool:
        if r_kata in entries.get(surf, {}):
            return True
        if "々" in surf and len(surf) >= 2:
            exp = "".join(surf[i - 1] if (ch == "々" and i > 0) else ch for i, ch in enumerate(surf))
            if exp != surf and verify_compound(exp, r_kata):
                return True
        n, m = len(surf), len(r_kata)
        if n == 1:
            return r_kata in single_sets.get(surf, set())
        if n > 16 or m > 36:
            return False
        reach: List[Set[int]] = [set() for _ in range(n + 1)]
        reach[0].add(0)
        for i in range(n):
            if not reach[i]:
                continue
            for j in range(i + 1, min(n, i + 6) + 1):
                if i == 0 and j == n:
                    continue
                piece = surf[i:j]
                if not ph.has_kanji(piece):
                    cands = {ph.normalize_kana(piece)}
                else:
                    base = set(entries.get(piece, {}).keys())
                    if len(piece) == 1:
                        base |= single_sets.get(piece, set())
                    if piece in ("ヶ", "ヵ"):
                        base |= {"カ", "ガ", "ケ", "ゲ"}
                    cands = set(base)
                    if i > 0:
                        for r in base:
                            if r and r[0] in _RENDAKU_MAP:
                                for v in _RENDAKU_MAP[r[0]]:
                                    cands.add(v + r[1:])
                    for r in list(cands):
                        if len(r) >= 2 and r[-1] in _SOKUON_TAILS:
                            cands.add(r[:-1] + "ッ")
                for pos in reach[i]:
                    for c in cands:
                        if c and r_kata.startswith(c, pos):
                            reach[j].add(pos + len(c))
        return m in reach[n]

    # 4. 同形異音語カタログの辞書検証
    cat_verified = 0
    cat_failed = []
    for item in hc.HOMOGRAPH_CATALOG:
        surf = item["surface"]
        for rinfo in item["readings"]:
            full_r = rinfo["full_reading"]
            # 活用語過去形（「行った」「開いた」等）は基本形や語幹+活用形として検証する
            ok = verify_compound(surf, full_r)
            if not ok and item["category"] == "predicate":
                # 過去形・音便形の語幹対応検証（末尾の「った」「いた」「めた」を基本形や送り仮名剥離で確認）
                span_s = item["span_surface"]
                span_r = rinfo["reading"]
                # 例: 行(イ)+った -> 行く(イク) 等が辞書にあれば語幹イが辞書検証済み
                # surface の漢字1文字に対して辞書エントリのいずれかの先頭一致で検証
                dict_cands = entries.get(surf, {})
                if not dict_cands:
                    # 同じ漢字で始まる small_lex の動詞・形容詞を探す
                    for ksurf, kreads in entries.items():
                        if ksurf.startswith(span_s) and len(ksurf) <= len(span_s) + 3:
                            for kr in kreads:
                                stripped = ph.strip_okurigana(ksurf, kr)
                                if stripped and stripped[0][2] == span_s and stripped[0][3] == span_r:
                                    ok = True
                                    break
                        if ok:
                            break
            if ok:
                cat_verified += 1
            else:
                cat_failed.append((surf, full_r))

    print(f"Catalog verification: verified={cat_verified}, failed={len(cat_failed)}")
    if cat_failed:
        raise RuntimeError(f"辞書未検証のカタログ項目があります: {cat_failed}")

    # 5. multi_readings (span_extractor 用の多読候補マップ) 構築
    # カタログ定義語を第1優先登録 + small_lex の純漢字1〜3字多読語（ノイズ連濁除く）を補完
    multi_readings: Dict[str, dict] = {}
    for item in hc.HOMOGRAPH_CATALOG:
        key = item["key"]
        rs = [r["reading"] for r in item["readings"]]
        multi_readings[key] = {
            "surface": item["surface"],
            "span_surface": item["span_surface"],
            "category": item["category"],
            "candidates": rs,
            "default": rs[0],
            "from_catalog": True,
        }

    # Sudachi small_lex 内の高信頼・多読純漢字語（長さ1〜3文字、非カタロク語）を追加
    extra_multi_count = 0
    for surf in small_surfaces:
        if surf in multi_readings:
            continue
        if not (1 <= len(surf) <= 3 and ph.is_all_kanji(surf)):
            continue
        rmap = entries.get(surf, {})
        # 独立語（名詞・動詞・形容詞・形状詞・副詞）でコスト11000以下の候補に絞る
        indep = [
            (r, cost)
            for r, (cost, pos1, is_indep) in rmap.items()
            if is_indep
            and cost <= 11000
            and not r.endswith("ッ")
            and pos1 in ("名詞", "動詞", "形容詞", "形状詞", "副詞")
        ]
        if len(indep) < 2:
            continue
        indep.sort(key=lambda x: x[1])
        # 連濁ノイズ（先頭のみ清濁差）と語中濁音変種を除外
        kept: List[str] = []
        kept_set: Set[str] = set()
        for r, cost in indep:
            if is_simple_rendaku_variant(r, kept_set):
                continue
            # 後半要素の連濁（例: ヒロカワ vs ヒロガワ、オオテ vs オオデ）で1文字のみ清濁差かつ同長も除外
            is_inner_rendaku = False
            for kr in kept_set:
                if len(kr) == len(r) and len(r) >= 3:
                    diffs = [idx for idx in range(len(r)) if kr[idx] != r[idx]]
                    if len(diffs) == 1:
                        di = diffs[0]
                        if _VOICING_PAIR.get(kr[di]) == r[di]:
                            is_inner_rendaku = True
                            break
            if is_inner_rendaku:
                continue
            if r not in kept_set:
                kept.append(r)
                kept_set.add(r)
        if len(kept) >= 2 and len(kept) <= 5:
            multi_readings[surf] = {
                "surface": surf,
                "span_surface": surf,
                "category": "lexicon_multi",
                "candidates": kept,
                "default": kept[0],
                "from_catalog": False,
            }
            extra_multi_count += 1

    print(f"multi_readings built: catalog={len(hc.HOMOGRAPH_CATALOG)}, extra_lex={extra_multi_count}, total={len(multi_readings)}")

    # 6. SQLite lexicon.sqlite 書き出し
    db_path = MASTER_DIR / "lexicon.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE entries (surface TEXT PRIMARY KEY, readings_all TEXT NOT NULL, default_reading TEXT NOT NULL)"
    )
    rows_to_insert = []
    for surf, rmap in entries.items():
        # コスト順にソート（独立語優先）
        sorted_r = sorted(rmap.items(), key=lambda x: (0 if x[1][2] else 1, x[1][0]))
        all_r = [r for r, _ in sorted_r]
        rows_to_insert.append((surf, json.dumps(all_r, ensure_ascii=False), all_r[0]))
    conn.executemany("INSERT INTO entries VALUES (?, ?, ?)", rows_to_insert)
    conn.commit()
    conn.close()

    # 7. reading_assets.json と lexicon-stats.json 書き出し
    assets = {
        "single_kanji": {k: sorted(v) for k, v in single_kanji.items()},
        "multi_readings": multi_readings,
        "pron_map": pron_map,
        "homograph_catalog": hc.HOMOGRAPH_CATALOG,
    }
    with open(MASTER_DIR / "reading_assets.json", "w", encoding="utf-8") as f:
        json.dump(assets, f, ensure_ascii=False)

    lex_stats = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": {
            "sudachidict": "v20261015 (small_lex.csv + core_lex.csv, Apache-2.0)",
            "unidic": "unidic-lite 2.1.2 (Modified BSD)",
        },
        "total_sudachi_rows_scanned": stats["small_lex.zip_rows"] + stats["core_lex.zip_rows"],
        "excluded_person_entries": stats["excluded_person"],
        "unique_kanji_surfaces": len(entries),
        "single_kanji_chars": len(single_kanji),
        "pron_exceptions_count": len(pron_map),
        "catalog_words": len(hc.HOMOGRAPH_CATALOG),
        "catalog_target_readings_verified": cat_verified,
        "catalog_failed": len(cat_failed),
        "multi_readings_total": len(multi_readings),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(MASTER_DIR / "lexicon-stats.json", "w", encoding="utf-8") as f:
        json.dump(lex_stats, f, ensure_ascii=False, indent=2)

    print("Saved master lexicon:", json.dumps(lex_stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
