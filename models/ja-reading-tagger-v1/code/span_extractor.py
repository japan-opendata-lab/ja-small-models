#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 決定的スパン抽出・辞書候補生成・辞書ベースラインモジュール。

特徴:
- 入力は text のみ（ゴールド情報は一切参照しない）
- SudachiPy SplitMode.C をベースに漢字スパンを抽出し、活用語は送り仮名を剥離して
  送り仮名文脈付きキー（例: "行+った", "開+く"）を生成する
- 一部複合語（一日・十分・一分・大人・大地震 等）での Sudachi 境界ズレを補正する
- 同一テキストに対する Sudachi Mode C / Mode A / UniDic の純粋辞書方式予測関数も提供する
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import fugashi
from sudachipy import dictionary, tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phonology as ph  # noqa: E402

_NUM_KANJI = frozenset("一二三四五六七八九十百千万億兆〇零")
_NUM_SENSITIVE_COMPOUNDS = frozenset({"一日", "一分", "十分", "三月", "五月"})
_ADJ_CATALOG_KEYS = frozenset({"辛+い", "良+い", "強+い"})


class SpanExtractor:
    """推論・学習共通の漢字スパン抽出＆辞書検証済み候補生成器。"""

    def __init__(self, master_dir: Path, exclude_person_names: bool = True) -> None:
        self.master_dir = Path(master_dir)
        self.exclude_person_names = exclude_person_names
        with open(self.master_dir / "reading_assets.json", encoding="utf-8") as f:
            assets = json.load(f)

        self.multi_readings: Dict[str, dict] = assets["multi_readings"]
        self.pron_map: Dict[str, str] = assets["pron_map"]
        self.catalog_by_key: Dict[str, dict] = {
            item["key"]: item for item in assets["homograph_catalog"]
        }

        # 複合語結合補正用の純漢字カタログ語（長さ2〜4文字）
        self.compound_catalog: Set[str] = {
            item["surface"]
            for item in assets["homograph_catalog"]
            if item["category"] == "compound" and ph.is_all_kanji(item["surface"])
        }
        # カタログ上の一般語を Sudachi が人名誤爆して落とさないための保護セット
        self.protected_surfaces: Set[str] = {
            item["surface"] for item in assets["homograph_catalog"]
        } | {
            item["span_surface"] for item in assets["homograph_catalog"]
        }

        self.sudachi_dict = dictionary.Dictionary(dict="core")
        self.tok = self.sudachi_dict.tokenizer()

    def to_pron(self, reading_kata: str, surface: str = "") -> str:
        """表記読みカタカナを発音読みカタカナ（UniDic準拠）へ変換する。"""
        return ph.kana_to_pron(reading_kata, surface=surface, pron_map=self.pron_map)

    def _tokenize_with_compound_fix(self, text: str) -> List[Tuple[int, int, str, str, Tuple[str, ...]]]:
        """Sudachi Mode C でトークナイズし、カタログ複合語の境界ズレ（例: 一+日中 -> 一日+中）を補正する。"""
        ms = list(self.tok.tokenize(text, tokenizer.Tokenizer.SplitMode.C))
        raw_tokens: List[Tuple[int, int, str, str, Tuple[str, ...]]] = [
            (m.begin(), m.end(), m.surface(), m.reading_form(), tuple(m.part_of_speech()[:3]))
            for m in ms
        ]
        fixed: List[Tuple[int, int, str, str, Tuple[str, ...]]] = []
        i = 0
        n_tok = len(raw_tokens)
        while i < n_tok:
            st, ed, surf, read_f, pos = raw_tokens[i]
            # 複合語カタログ (長さ3, 2) への先行マッチ確認（現在トークンが複合語より短い場合のみ補正）
            matched_comp: Optional[str] = None
            for L in (3, 2):
                if st + L <= len(text) and len(surf) < L:
                    cand = text[st : st + L]
                    if cand in self.compound_catalog:
                        # 漢数字直後の誤結合ガード（例: 二十一日, 十二月）
                        if cand in _NUM_SENSITIVE_COMPOUNDS and st > 0 and text[st - 1] in _NUM_KANJI:
                            continue
                        matched_comp = cand
                        break

            if matched_comp is not None:
                comp_end = st + len(matched_comp)
                # comp_end を完全に超えない、またはちょうど一致するトークン群を消費
                j = i
                while j < n_tok and raw_tokens[j][1] <= comp_end:
                    j += 1
                info = self.multi_readings[matched_comp]
                fixed.append(
                    (st, comp_end, matched_comp, info["default"], ("名詞", "普通名詞", "一般"))
                )
                # もし途中のトークンをまたいで comp_end が終わった場合、
                # 次の句読点・空白・文末までの残余テキストを再解析して raw_tokens の該当区間と差し替える
                if j < n_tok and raw_tokens[j][0] < comp_end < raw_tokens[j][1]:
                    chunk_end = len(text)
                    for p in range(comp_end, len(text)):
                        if text[p] in "、。！？,.!?\n\r \u3000":
                            chunk_end = p + 1
                            break
                    k = j
                    while k < n_tok and raw_tokens[k][1] <= chunk_end:
                        k += 1
                    chunk_end = max(chunk_end, raw_tokens[k - 1][1]) if k > j else raw_tokens[j][1]
                    rem_text = text[comp_end:chunk_end]
                    re_tokens: List[Tuple[int, int, str, str, Tuple[str, ...]]] = [
                        (
                            comp_end + rm.begin(),
                            comp_end + rm.end(),
                            rm.surface(),
                            rm.reading_form(),
                            tuple(rm.part_of_speech()[:3]),
                        )
                        for rm in self.tok.tokenize(rem_text, tokenizer.Tokenizer.SplitMode.C)
                    ]
                    raw_tokens = raw_tokens[:j] + re_tokens + raw_tokens[k:]
                    n_tok = len(raw_tokens)
                i = j
                continue

            fixed.append((st, ed, surf, read_f, pos))
            i += 1
        return fixed

    def _resolve_key_and_candidates(
        self,
        text: str,
        abs_s: int,
        abs_e: int,
        k_surf: str,
        k_read: str,
        morph_is_all_kanji: bool,
        pos: Tuple[str, ...],
    ) -> Tuple[str, List[str]]:
        """漢字ブロックに対する識別キーと辞書候補リストを決定する。"""
        # 0. 接頭辞・接尾辞・数詞は複合要素・助数詞として文脈依存読みが Sudachi で確定するため
        #    独立名詞の単漢字多読候補を適用せず単一確定とする（例: 地球上, 衆道家, 二家, 一本, 一時間）
        if pos[0] in ("接頭辞", "接尾辞") or pos[1] == "数詞":
            return k_surf, ([k_read] if k_read else [])

        # 1. 後続文字を先読みして活用語カタログキー（例: 行+った, 開+く）を確認（品詞整合チェック付き）
        for tlen in (2, 1):
            if abs_e + tlen <= len(text):
                trail = text[abs_e : abs_e + tlen]
                # 送り仮名はひらがなのみ
                if all("\u3041" <= c <= "\u3096" for c in trail):
                    ck = f"{k_surf}+{trail}"
                    if ck in self.multi_readings:
                        if ck in _ADJ_CATALOG_KEYS and pos[0] != "形容詞":
                            continue
                        if ck not in _ADJ_CATALOG_KEYS and pos[0] != "動詞":
                            continue
                        cands = list(self.multi_readings[ck]["candidates"])
                        if k_read and k_read not in cands:
                            cands = [k_read] + cands
                        return ck, cands

        # 2. 活用語・送り仮名剥離済みパーツの場合、名詞の単漢字候補（例: 行=ギョウ）は混ぜない
        is_predicate = pos[0] in ("動詞", "形容詞") or (not morph_is_all_kanji)
        if not is_predicate and k_surf in self.multi_readings:
            cands = list(self.multi_readings[k_surf]["candidates"])
            if k_read and k_read not in cands:
                cands = [k_read] + cands
            return k_surf, cands

        # 3. 単一候補（曖昧性なし）
        return k_surf, ([k_read] if k_read else [])

    def extract_spans(self, text: str) -> List[dict]:
        """テキストから対象漢字スパン一覧と候補リストを抽出する（入力は text のみ）。"""
        if not text:
            return []
        tokens = self._tokenize_with_compound_fix(text)
        spans: List[dict] = []

        for st, ed, surf, read_f, pos in tokens:
            if not ph.has_kanji(surf):
                continue
            if self.exclude_person_names and pos[1] == "固有名詞" and pos[2] == "人名":
                if surf not in self.protected_surfaces:
                    continue

            rk = ph.normalize_kana(read_f) if (read_f and read_f != "*") else ""
            if not rk or not ph.is_valid_reading_str(rk):
                # 未知読みの場合は辞書フォールバック（複数候補があれば先頭、無ければスキップ）
                if surf in self.multi_readings:
                    rk = self.multi_readings[surf]["default"]
                else:
                    continue

            morph_all_kanji = ph.is_all_kanji(surf)
            stripped = ph.strip_okurigana(surf, rk)
            if stripped is None:
                # 送り仮名剥離できない特殊語（全体が漢字を含む1塊）はそのまま扱う
                stripped = [(0, len(surf), surf, rk)]

            for rel_s, rel_e, k_surf, k_read in stripped:
                abs_s = st + rel_s
                abs_e = st + rel_e
                assert text[abs_s:abs_e] == k_surf, f"スパン不一致: {text[abs_s:abs_e]!r} != {k_surf!r}"

                key, cands = self._resolve_key_and_candidates(
                    text, abs_s, abs_e, k_surf, k_read, morph_all_kanji, pos
                )
                if not cands:
                    continue
                default_r = k_read if k_read in cands else cands[0]
                spans.append(
                    {
                        "start": abs_s,
                        "end": abs_e,
                        "surface": k_surf,
                        "key": key,
                        "candidates": cands,
                        "default_reading": default_r,
                    }
                )
        return spans


class DictionaryBaselines:
    """比較用・純粋辞書方式（Sudachi Mode C / Mode A / UniDic）の統一予測器。"""

    def __init__(self, master_dir: Path) -> None:
        self.master_dir = Path(master_dir)
        with open(self.master_dir / "reading_assets.json", encoding="utf-8") as f:
            assets = json.load(f)
        self.pron_map: Dict[str, str] = assets["pron_map"]
        self.sudachi_tok = dictionary.Dictionary(dict="core").tokenizer()
        self.unidic_tagger = fugashi.Tagger()

    def predict_sudachi(self, text: str, mode: str = "C") -> List[dict]:
        """純粋な SudachiPy (Mode A or C) の読み予測結果を返す。"""
        split_mode = (
            tokenizer.Tokenizer.SplitMode.A
            if mode.upper() == "A"
            else tokenizer.Tokenizer.SplitMode.C
        )
        out: List[dict] = []
        for m in self.sudachi_tok.tokenize(text, split_mode):
            surf = m.surface()
            if not ph.has_kanji(surf):
                continue
            rf = m.reading_form()
            if not rf or rf == "*":
                continue
            rk = ph.normalize_kana(rf)
            if not ph.is_valid_reading_str(rk):
                continue
            stripped = ph.strip_okurigana(surf, rk)
            if stripped is None:
                stripped = [(0, len(surf), surf, rk)]
            st = m.begin()
            for rel_s, rel_e, k_surf, k_read in stripped:
                abs_s = st + rel_s
                abs_e = st + rel_e
                out.append(
                    {
                        "start": abs_s,
                        "end": abs_e,
                        "surface": k_surf,
                        "reading": ph.kata_to_hira(k_read),
                        "reading_kata": k_read,
                        "pron": ph.kana_to_pron(k_read, surface=k_surf, pron_map=self.pron_map),
                    }
                )
        return out

    def predict_unidic(self, text: str) -> List[dict]:
        """純粋な UniDic (fugashi + unidic-lite) の読み予測結果を返す。"""
        out: List[dict] = []
        pos = 0
        for w in self.unidic_tagger(text):
            surf = w.surface
            idx = text.find(surf, pos)
            if idx < 0:
                idx = pos
            pos = idx + len(surf)
            if not ph.has_kanji(surf):
                continue
            rk = ph.normalize_kana(w.feature.kana) if w.feature.kana else ""
            if not rk or not ph.is_valid_reading_str(rk):
                continue
            pr_full = ph.normalize_kana(w.feature.pron) if w.feature.pron else ""
            stripped = ph.strip_okurigana(surf, rk)
            if stripped is None:
                stripped = [(0, len(surf), surf, rk)]
            for rel_s, rel_e, k_surf, k_read in stripped:
                abs_s = idx + rel_s
                abs_e = idx + rel_e
                # 単体漢字ブロックそのままなら UniDic の pron を活用（送り仮名除去ありなら kana_to_pron で整合）
                if len(stripped) == 1 and k_surf == surf and pr_full:
                    pron = pr_full
                else:
                    pron = ph.kana_to_pron(k_read, surface=k_surf, pron_map=self.pron_map)
                out.append(
                    {
                        "start": abs_s,
                        "end": abs_e,
                        "surface": k_surf,
                        "reading": ph.kata_to_hira(k_read),
                        "reading_kata": k_read,
                        "pron": pron,
                    }
                )
        return out
