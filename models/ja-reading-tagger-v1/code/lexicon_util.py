#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 統合辞書検索・検証ユーティリティ。

build_lexicon.py が生成した data/master/lexicon.sqlite と reading_assets.json を用い、
任意の (surface, reading_kata) が UniDic ∪ SudachiDict に実在する正規の読みであるか検証する。
"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import phonology as ph

# 日本語の正規連濁マッピング（後続要素頭子音の有声化）
_RENDAKU_MAP: Dict[str, Tuple[str, ...]] = {
    "カ": ("ガ",), "キ": ("ギ",), "ク": ("グ",), "ケ": ("ゲ",), "コ": ("ゴ",),
    "サ": ("ザ",), "シ": ("ジ",), "ス": ("ズ",), "セ": ("ゼ",), "ソ": ("ゾ",),
    "タ": ("ダ",), "チ": ("ヂ", "ジ"), "ツ": ("ヅ", "ズ"), "テ": ("デ",), "ト": ("ド",),
    "ハ": ("バ", "パ"), "ヒ": ("ビ", "ピ"), "フ": ("ブ", "プ"), "ヘ": ("ベ", "ペ"), "ホ": ("ボ", "ポ"),
}

# 漢字複合語での促音便末尾変化
_SOKUON_TAILS = frozenset("ツチクキ")


class LexiconValidator:
    """UniDic ∪ SudachiDict に基づく決定的な読み検証器。"""

    def __init__(self, master_dir: Path) -> None:
        self.master_dir = Path(master_dir)
        self.db_path = self.master_dir / "lexicon.sqlite"
        self.assets_path = self.master_dir / "reading_assets.json"
        if not self.db_path.exists() or not self.assets_path.exists():
            raise FileNotFoundError(
                f"辞書マスターが未構築です: {self.master_dir}。先に build_lexicon.py を実行してください。"
            )
        self.conn = sqlite3.connect(str(self.db_path))
        with open(self.assets_path, encoding="utf-8") as f:
            assets = json.load(f)
        self.single_kanji: Dict[str, List[str]] = assets["single_kanji"]
        self.single_kanji_sets: Dict[str, Set[str]] = {
            k: set(v) for k, v in self.single_kanji.items()
        }
        self.multi_readings: Dict[str, dict] = assets["multi_readings"]
        self.pron_map: Dict[str, str] = assets["pron_map"]
        self.catalog_by_key: Dict[str, dict] = {
            item["key"]: item for item in assets["homograph_catalog"]
        }
        self.catalog_surfaces: Set[str] = {
            item["surface"] for item in assets["homograph_catalog"]
        }
        self._cache: Dict[str, Set[str]] = {}

    def close(self) -> None:
        self.conn.close()

    def lookup_exact(self, surface: str) -> Set[str]:
        """見出し語そのものに対する辞書登録済み読み集合（人名除外）を返す。"""
        if surface in self._cache:
            return self._cache[surface]
        cur = self.conn.execute(
            "SELECT readings_all FROM entries WHERE surface = ?", (surface,)
        )
        row = cur.fetchone()
        res: Set[str] = set(json.loads(row[0])) if row else set()
        if len(self._cache) < 300000:
            self._cache[surface] = res
        return res

    def lookup_default(self, surface: str) -> Optional[str]:
        """見出し語に対する辞書上のデフォルト（最小コスト）読みを返す。"""
        cur = self.conn.execute(
            "SELECT default_reading FROM entries WHERE surface = ?", (surface,)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def _possible_piece_readings(self, piece: str, allow_rendaku: bool) -> Set[str]:
        """部分漢字列またはかな列の取り得る読み集合（連濁・促音変種含む）。"""
        if not piece:
            return {""}
        if not ph.has_kanji(piece):
            return {ph.normalize_kana(piece)}

        base: Set[str] = set()
        if len(piece) == 1:
            base |= self.single_kanji_sets.get(piece, set())
        base |= self.lookup_exact(piece)

        if not base:
            return set()

        variants: Set[str] = set(base)
        # 連濁（2番目以降の要素）
        if allow_rendaku:
            for r in base:
                if r and r[0] in _RENDAKU_MAP:
                    for voiced in _RENDAKU_MAP[r[0]]:
                        variants.add(voiced + r[1:])
        # 促音化（後ろに別要素が続く場合の「ツ/チ/ク/キ -> ッ」や「イチ -> イッ」等）
        for r in list(variants):
            if len(r) >= 2 and r[-1] in _SOKUON_TAILS:
                variants.add(r[:-1] + "ッ")
        return variants

    def verify_reading(self, surface: str, reading_kata: str) -> bool:
        """(surface, reading_kata) が UniDic ∪ SudachiDict（または正規複合構成）に実在するか検証する。"""
        r_kata = ph.normalize_kana(reading_kata)
        if not surface or not r_kata or not ph.is_valid_reading_str(r_kata):
            return False

        # 1. 直接一致（最も厳密かつ高速）
        if r_kata in self.lookup_exact(surface):
            return True

        # 2. 単漢字かつ「々」展開（例: 人々 -> 人+人）
        if "々" in surface and len(surface) >= 2:
            expanded = []
            for idx, ch in enumerate(surface):
                if ch == "々" and idx > 0:
                    expanded.append(expanded[-1])
                else:
                    expanded.append(ch)
            exp_surf = "".join(expanded)
            if exp_surf != surface and self.verify_reading(exp_surf, r_kata):
                return True

        # 3. 送り仮名剥離できる場合は、語幹・活用形一致も試す
        #    例: 動いた (ウゴイタ) -> 語幹「動」(ウゴ) が「動く(ウゴク)」等の語幹と一致するか
        #    （ただし一般には完全形態が Sudachi/UniDic にあるため、下記複合分割で十分処理できる）

        # 4. 複合語分割による構成要素の辞書読み連結一致
        #    例: 一日=イチニチ (一[イチ]+日[ニチ]), 十分=ジュップン (十[ジュッ]+分[プン]),
        #        大地震=ダイジシン (大[ダイ]+地震[ジシン]), 流し斬り=ナガシギリ (流し+斬り)
        n = len(surface)
        if n == 1:
            return r_kata in self.single_kanji_sets.get(surface, set())

        # DPで surface[0:i] が r_kata[0:j] を構成できるか探索（要素最大長=min(6, n)）
        m = len(r_kata)
        # 単語長が極端に長い場合は直接一致のみ
        if n > 16 or m > 36:
            return False

        reachable: List[Set[int]] = [set() for _ in range(n + 1)]
        reachable[0].add(0)

        for i in range(n):
            if not reachable[i]:
                continue
            max_step = min(n, i + 6)
            for j in range(i + 1, max_step + 1):
                piece = surface[i:j]
                # もし全体が単一ブロックなら既に 1. で確認済みなのでスキップ
                if i == 0 and j == n:
                    continue
                cands = self._possible_piece_readings(piece, allow_rendaku=(i > 0))
                # 「ヶ」「ヵ」が単独で助数詞「カ」「ガ」「ケ」として使われる場合
                if piece in ("ヶ", "ヵ"):
                    cands |= {"カ", "ガ", "ケ", "ゲ"}
                if not cands:
                    continue
                for pos in reachable[i]:
                    for cand in cands:
                        if cand and r_kata.startswith(cand, pos):
                            reachable[j].add(pos + len(cand))

        return m in reachable[n]

    def verify_span_reading(
        self,
        surface: str,
        reading_kata: str,
        trailing_text: str = "",
        leading_text: str = "",
    ) -> bool:
        """文脈（直前の接頭辞かな・直後の送り仮名）を含めて辞書実在を検証する。"""
        r_kata = ph.normalize_kana(reading_kata)
        if not surface or not r_kata or not ph.is_valid_reading_str(r_kata):
            return False

        # 1. 表層そのままの検証
        if self.verify_reading(surface, r_kata):
            return True

        # 2. 直後の送り仮名（ひらがな 1〜3 文字）を結合した活用形での検証
        #    例: 流(なが)+し -> 流し(ナガシ), 入(はい)+れ -> 入れ(ハイレ), 行(おこな)+った -> 行っ(オコナッ)
        trail_hira = ""
        for ch in trailing_text[:4]:
            if "\u3041" <= ch <= "\u3096":
                trail_hira += ch
            else:
                break
        for k in range(1, len(trail_hira) + 1):
            t_sub = trail_hira[:k]
            comb_surf = surface + t_sub
            comb_read = r_kata + ph.hira_to_kata(t_sub)
            if comb_read in self.lookup_exact(comb_surf):
                return True
            if k >= 2 and self.verify_reading(comb_surf, comb_read):
                return True

        # 3. 直前接頭辞「お/ご」+ 漢字 + 後続接尾辞（例: お父さん -> 父[トウ], ご利益 -> 利益）
        if leading_text and leading_text[-1] in ("お", "ご", "御"):
            lead_ch = leading_text[-1]
            # leading + surface + trailing 1..2文字
            for k in range(0, min(3, len(trail_hira)) + 1):
                t_sub = trail_hira[:k]
                comb_surf = lead_ch + surface + t_sub
                comb_read = ph.hira_to_kata(lead_ch) + r_kata + ph.hira_to_kata(t_sub)
                if comb_read in self.lookup_exact(comb_surf):
                    return True

        # 4. 直前漢字との連濁（「流し斬り」の「斬」など、前後に漢字がある語中連濁）
        #    例: 先頭文字の濁音・半濁音を清音に戻して再検証（ただし「人」「日」等の熟語後部や複合活用語のみ）
        if leading_text and (ph.is_kanji_char(leading_text[-1]) or ("\u3041" <= leading_text[-1] <= "\u3096")):
            # 逆連濁マップで清音に戻して確認
            _unvoice = {
                "ガ": "カ", "ギ": "キ", "グ": "ク", "ゲ": "ケ", "ゴ": "コ",
                "ザ": "サ", "ジ": "シ", "ズ": "ス", "ゼ": "セ", "ゾ": "ソ",
                "ダ": "タ", "ヂ": "チ", "ヅ": "ツ", "デ": "テ", "ド": "ト",
                "バ": "ハ", "ビ": "ヒ", "ブ": "フ", "ベ": "ヘ", "ボ": "ホ",
                "パ": "ハ", "ピ": "ヒ", "プ": "フ", "ペ": "ヘ", "ポ": "ホ",
            }
            if r_kata[0] in _unvoice:
                unv_r = _unvoice[r_kata[0]] + r_kata[1:]
                if self.verify_reading(surface, unv_r):
                    return True
                for k in range(1, len(trail_hira) + 1):
                    t_sub = trail_hira[:k]
                    if (unv_r + ph.hira_to_kata(t_sub)) in self.lookup_exact(surface + t_sub):
                        return True
        return False

    def to_pron(self, reading_kata: str, surface: str = "") -> str:
        """表記読みカタカナを発音読みカタカナ（UniDic準拠）に変換する。"""
        return ph.kana_to_pron(reading_kata, surface=surface, pron_map=self.pron_map)
