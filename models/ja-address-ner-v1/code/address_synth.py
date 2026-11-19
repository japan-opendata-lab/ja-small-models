#!/usr/bin/env python3
"""第7号 ja-address-ner-v1 用 架空住所合成器。

公的マスター（municipalities.json, town_index.json, old_municipalities.json）の
地名（事実情報）と完全架空の番地・建物名を組み合わせ、崩れた表記・旧市町村・
同名地名・判定不能ケースを含む多様な合成住所データを生成する。
実在の個人住所は一切使用しない。
"""
import collections
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_norm as gn  # noqa: E402
from linker import AddressLinker  # noqa: E402

DEFAULT_MASTER_DIR = Path(__file__).resolve().parent.parent / "data" / "master"

# 分割間リーク遮断用の建物名語幹（train / dev / sealed-test で完全非重複）
BUILDING_PREFIXES: Dict[str, List[str]] = {
    "train": [
        "サンライズ", "グリーン", "パーク", "リバーサイド", "メゾン", "グランド",
        "スカイ", "アーバン", "ロイヤル", "エクセル", "ベル", "プレイス",
        "クレスト", "フォレスト", "シーサイド", "サニー", "フローラル", "コスモ",
        "ヴィラ", "プラザ", "エスポワール", "コンフォート", "シャトー", "レジデンス",
        "セジュール", "ディア", "ルミエール", "パル", "ラフォーレ", "オーロラ",
        "サンシャイン", "フェニックス", "カトレア", "セントラル", "ニュー", "フェア",
    ],
    "dev": [
        "ブルー", "ブリーズ", "モーニング", "ガーデニア", "ソレイユ", "マリン",
        "ヒルサイド", "プライム", "ミライ", "オリオン", "アーク", "コースト",
    ],
    "sealed-test": [
        "シルバー", "ウィング", "ステラ", "キャッスル", "オーシャン", "ハーモニー",
        "エメラルド", "クラウン", "パレス", "シトラス", "ヴェール", "ルネ",
    ],
}

BUILDING_SUFFIXES: Dict[str, List[str]] = {
    "train": ["ハイツ", "マンション", "コーポ", "コート", "タワー", "ビル", "テラス", "ハイム", "荘"],
    "dev": ["ハイツ", "マンション", "ガーデン", "ハウス", "スクエア", "レジデンス", "ビル"],
    "sealed-test": ["コーポ", "コート", "タワー", "アパート", "プラザ", "テラス", "館"],
}

# 分割別 非重複の架空地名語彙（実在都道府県・市区町村・町域との非重複を初期化時に検証）
FAKE_PREFS: Dict[str, List[str]] = {
    "train": ["架空県", "空想県", "新星県", "東雲県", "瑞穂県", "彩雲県", "翠緑県", "星雲県"],
    "dev": ["白樺県", "蒼穹県", "曙光県", "萌木県"],
    "sealed-test": ["琥珀県", "翡翠県", "常磐県", "朝霧県"],
}

FAKE_CITIES: Dict[str, List[str]] = {
    "train": ["架空市", "夢見区", "星見町", "月見村", "雲海市", "虹ヶ丘市", "風早町", "汐見区", "希望市", "光輝町"],
    "dev": ["天翔市", "若葉野町", "澄川区", "涼風市", "霧ヶ峰村"],
    "sealed-test": ["蒼海市", "銀嶺町", "朝凪区", "花明市", "星屑村"],
}

FAKE_TOWNS: Dict[str, List[str]] = {
    "train": ["架空町", "不存在町", "夢の台", "幻ヶ丘", "新天地町", "空色町", "幻影町", "星影町"],
    "dev": ["仮想町", "空想ヶ丘", "未知町", "蜃気楼町", "淡雪町"],
    "sealed-test": ["虚構町", "幻想台", "未知野", "薄明町", "風花町"],
}

# 正字 -> 旧字・異体字（variant 用）
TO_OLD_VARIANT: List[Tuple[str, str]] = [
    ("沢", "澤"), ("桜", "櫻"), ("広", "廣"), ("浜", "濱"), ("辺", "邊"),
    ("斉", "齋"), ("桧", "檜"), ("竜", "龍"), ("高", "髙"), ("崎", "﨑"),
    ("条", "條"), ("国", "國"), ("学", "學"), ("会", "會"), ("恵", "惠"),
    ("関", "關"), ("島", "嶋"), ("真", "眞"), ("滝", "瀧"), ("栄", "榮"),
    ("豊", "豐"), ("瀬", "瀨"), ("緑", "綠"), ("黒", "黑"), ("稲", "稻"),
]

_ARABIC_TO_FULL = str.maketrans("0123456789", "０１２３４５６７８９")
_KANJI_DIGITS = {
    1: "一", 2: "二", 3: "三", 4: "四", 5: "五",
    6: "六", 7: "七", 8: "八", 9: "九", 10: "十",
    11: "十一", 12: "十二", 13: "十三", 14: "十四", 15: "十五",
    16: "十六", 17: "十七", 18: "十八", 19: "十九", 20: "二十",
}


def _katakana_to_hiragana(s: str) -> str:
    out = []
    for ch in s:
        c = ord(ch)
        if 0x30A1 <= c <= 0x30F6:
            out.append(chr(c - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _gen_block(rng: random.Random, style: Optional[str] = None) -> str:
    """架空の番地等表記を生成する。"""
    a = rng.randint(1, 9)
    b = rng.randint(1, 35)
    c = rng.randint(1, 45)
    if style is None:
        style = rng.choices(
            ["hyphen", "chome_ban_go", "ban_go", "banchi", "no"],
            weights=[0.42, 0.25, 0.18, 0.08, 0.07],
        )[0]
    if style == "hyphen":
        return f"{a}-{b}-{c}" if rng.random() < 0.7 else f"{b}-{c}"
    if style == "chome_ban_go":
        return f"{a}丁目{b}番{c}号"
    if style == "ban_go":
        return f"{b}番{c}号"
    if style == "banchi":
        return f"{b}番地{c}" if rng.random() < 0.6 else f"{b}番地"
    if style == "no":
        return f"{a}の{b}の{c}"
    if style == "fullwidth":
        sep = rng.choice(["－", "ー", "−"])
        return f"{a}{sep}{b}{sep}{c}".translate(_ARABIC_TO_FULL)
    if style == "kanji":
        ka = _KANJI_DIGITS.get(a, str(a))
        kb = _KANJI_DIGITS.get(b % 20 or 1, str(b))
        kc = _KANJI_DIGITS.get(c % 20 or 1, str(c))
        return f"{ka}丁目{kb}番{kc}号"
    return f"{a}-{b}-{c}"


def _gen_building(rng: random.Random, split: str) -> str:
    """分割専用語彙から架空の建物名（＋号室/階数）を生成する。"""
    pfx = rng.choice(BUILDING_PREFIXES[split])
    sfx = rng.choice(BUILDING_SUFFIXES[split])
    num_tag = ""
    r = rng.random()
    if r < 0.55:
        room = rng.randint(1, 9) * 100 + rng.randint(1, 18)
        unit = rng.choice(["号室", "号", ""])
        num_tag = f"{room}{unit}"
    elif r < 0.80:
        fl = rng.randint(1, 12)
        num_tag = f"{fl}F" if rng.random() < 0.5 else f"{fl}階"
    elif r < 0.90:
        num_tag = f" {rng.randint(101, 809)}号室"
    if rng.random() < 0.15 and sfx in ("ハイツ", "コーポ", "マンション"):
        return f"第{rng.randint(2, 5)}{pfx}{sfx}{num_tag}"
    return f"{pfx}{sfx}{num_tag}"


class AddressSynthesizer:
    """マスター辞書とリンカーを用いて整合性保証付きの合成住所を生成する。"""

    def __init__(self, master_dir: Optional[Path] = None) -> None:
        mdir = Path(master_dir) if master_dir else DEFAULT_MASTER_DIR
        self.linker = AddressLinker(mdir)
        self.munis: Dict[str, dict] = self.linker.munis
        self.town_data: Dict[str, dict] = json.loads(
            (mdir / "town_index.json").read_text(encoding="utf-8")
        )
        self.old_records: List[dict] = json.loads(
            (mdir / "old_municipalities.json").read_text(encoding="utf-8")
        )

        self.codes_with_town: List[str] = sorted(
            c for c, td in self.town_data.items() if td["clean_towns"]
        )

        # 県省略しても全国一意な自治体コード（特別区かつ政令市と同名区は除外）
        base_city_counts = collections.Counter()
        for m in self.munis.values():
            base_city_counts[gn.normalize_geo(m["base_city"])] += 1
        ward_names = {gn.normalize_geo(m["ward"]) for m in self.munis.values() if m["ward"]}
        self.unique_city_codes: List[str] = [
            c
            for c in self.codes_with_town
            if base_city_counts[gn.normalize_geo(self.munis[c]["base_city"])] == 1
            and gn.normalize_geo(self.munis[c]["base_city"]) not in ward_names
        ]

        # 同名市区町村・同名区グループ（same_name_resolved & unresolvable_ambiguous 用）
        short_groups: Dict[str, List[str]] = collections.defaultdict(list)
        for c, m in self.munis.items():
            if not self.town_data[c]["clean_towns"]:
                continue
            if m["ward"]:
                short_groups[m["ward"]].append(c)
            elif m["pref"] == "東京都" and m["city"].endswith("区"):
                short_groups[m["city"]].append(c)
            else:
                short_groups[m["base_city"]].append(c)

        self.same_name_resolved_pool: List[Tuple[str, str, str, str, str]] = []
        # (short_city, code, town, postal, kana)
        self.same_name_ambig_town_pool: List[Tuple[str, str]] = []
        # (short_city, shared_town)
        self.same_name_short_cities: List[str] = []

        for short_name, codes in sorted(short_groups.items()):
            if len(codes) < 2:
                continue
            self.same_name_short_cities.append(short_name)
            # 各町域正規化キーがグループ内の何コードに出現するか
            key_to_codes: Dict[str, Set[str]] = collections.defaultdict(set)
            for c in codes:
                for nk in self.linker.town_norm_by_code[c]:
                    key_to_codes[nk].add(c)
            for c in codes:
                for t_surf, postal, kana in self.town_data[c]["clean_towns"]:
                    t_keys = gn.extract_town_variants(t_surf)
                    # 1自治体固有の町域（他候補に1キーも被らない）
                    if t_keys and all(key_to_codes[k] == {c} for k in t_keys):
                        self.same_name_resolved_pool.append(
                            (short_name, c, t_surf, postal, kana)
                        )
                    # 複数候補に共通実在する町域
                    elif any(len(key_to_codes[k]) >= 2 for k in t_keys):
                        self.same_name_ambig_town_pool.append((short_name, t_surf))

        # 共通町域の重複除去
        self.same_name_ambig_town_pool = sorted(set(self.same_name_ambig_town_pool))

        # 全国複数自治体に存在する単独町域（例: 本町, 中央 等）
        clean_town_codes: Dict[str, Set[str]] = collections.defaultdict(set)
        for c in self.codes_with_town:
            for t_surf, _, _ in self.town_data[c]["clean_towns"]:
                if 2 <= len(t_surf) <= 5:
                    clean_town_codes[t_surf].add(c)
        self.shared_standalone_towns: List[str] = sorted(
            t for t, cs in clean_town_codes.items() if len(cs) >= 4
        )

        # 政令市名単独リスト
        self.seirei_city_list: List[Tuple[str, str]] = sorted(
            {(m["pref"], m["seirei_city"]) for m in self.munis.values() if m["seirei_city"]}
        )

        # 旧市町村の事前整理（1対1 / 分割型で固有町域ペア構築）
        self.old_1to1: List[dict] = [r for r in self.old_records if len(r["codes"]) == 1]
        self.old_split_towns: List[Tuple[str, str, str, str, str]] = []
        # (old_name, pref, target_code, town, postal)
        self.old_split_names: List[Tuple[str, str]] = []
        for r in self.old_records:
            if len(r["codes"]) <= 1:
                continue
            self.old_split_names.append((r["old_name"], r["pref"]))
            key_c: Dict[str, Set[str]] = collections.defaultdict(set)
            for c in r["codes"]:
                for nk in self.linker.town_norm_by_code[c]:
                    key_c[nk].add(c)
            for c in r["codes"]:
                for t_surf, postal, _ in self.town_data[c]["clean_towns"]:
                    tk = gn.extract_town_variants(t_surf)
                    if tk and all(key_c[k] == {c} for k in tk):
                        self.old_split_towns.append(
                            (r["old_name"], r["pref"], c, t_surf, postal)
                        )

        # variant 用候補事前インデックス（異体字変換可能な地名・ヶ/ケ/が含有町名）
        old_chars = {a for a, _ in TO_OLD_VARIANT}
        self.variant_char_codes: List[str] = [
            c
            for c in self.codes_with_town
            if (set(self.munis[c]["pref"]) & old_chars)
            or (set(self.munis[c]["city"]) & old_chars)
        ]
        self.ke_ga_pool: List[Tuple[str, str, str]] = []  # (code, town, postal)
        for c in self.codes_with_town:
            for t_surf, postal, _ in self.town_data[c]["clean_towns"]:
                if any(ch in t_surf for ch in ("ヶ", "ケ", "が", "の", "ノ", "ツ", "ッ")):
                    self.ke_ga_pool.append((c, t_surf, postal))
        self.county_codes: List[str] = [
            c for c in self.codes_with_town if self.munis[c]["county"]
        ]

        # 架空語彙が実在の都道府県・市区町村・町域と一切一致しないことを検証
        for sp, prefs in FAKE_PREFS.items():
            for fp in prefs:
                assert gn.normalize_pref(fp) is None, f"架空県が実在と衝突 ({sp}): {fp}"
        for sp, cities in FAKE_CITIES.items():
            for fc in cities:
                res = self.linker.link(city=fc)
                assert res["reason"] == "unknown_city", f"架空市が実在と衝突 ({sp}): {fc}"
        for sp, towns in FAKE_TOWNS.items():
            for ft in towns:
                res = self.linker.link(town=ft)
                assert res["reason"] == "unknown_town", f"架空町が実在と衝突 ({sp}): {ft}"

    def _pick_town(self, rng: random.Random, code: str) -> Tuple[str, str, str]:
        return rng.choice(self.town_data[code]["clean_towns"])

    def _verify_and_build(
        self,
        category: str,
        segments: List[Tuple[Optional[str], str]],
        postal: Optional[str],
        expected_status: str,
        expected_code: Optional[str],
        subtype: str = "",
    ) -> Optional[dict]:
        """生成したセグメント列に対して AddressLinker の判定結果が期待通りか検証する。"""
        comp_map: Dict[str, str] = {}
        for lb, txt in segments:
            if lb is not None:
                comp_map[lb] = txt
        res = self.linker.link_components(comp_map)
        if expected_status == "RESOLVED":
            if res["status"] != "RESOLVED" or res["municipality_code"] != expected_code:
                return None
        else:
            if res["status"] != "UNRESOLVABLE":
                return None
        return {
            "category": category,
            "subtype": subtype or category,
            "segments": segments,
            "address_text": "".join(t for _, t in segments),
            "postal": postal if expected_status == "RESOLVED" else None,
            "expected_status": expected_status,
            "expected_code": expected_code,
            "linker_reason": res["reason"],
        }

    def _gen_standard(self, rng: random.Random, split: str) -> Optional[dict]:
        code = rng.choice(self.codes_with_town)
        m = self.munis[code]
        town, postal, _ = self._pick_town(rng, code)
        block = _gen_block(rng)
        segs: List[Tuple[Optional[str], str]] = [
            ("PREF", m["pref"]),
            ("CITY", m["city"]),
            ("TOWN", town),
            ("BLOCK", block),
        ]
        if rng.random() < 0.45:
            segs.append((None, " "))
            segs.append(("BUILDING", _gen_building(rng, split)))
        return self._verify_and_build("standard", segs, postal, "RESOLVED", code)

    def _gen_omit_pref(self, rng: random.Random, split: str) -> Optional[dict]:
        code = rng.choice(self.unique_city_codes)
        m = self.munis[code]
        town, postal, _ = self._pick_town(rng, code)
        city_surf = m["base_city"] if (m["county"] and rng.random() < 0.4) else m["city"]
        segs: List[Tuple[Optional[str], str]] = [
            ("CITY", city_surf),
            ("TOWN", town),
            ("BLOCK", _gen_block(rng)),
        ]
        if rng.random() < 0.40:
            segs.append((None, " "))
            segs.append(("BUILDING", _gen_building(rng, split)))
        return self._verify_and_build("omit_pref", segs, postal, "RESOLVED", code)

    def _gen_same_name_resolved(self, rng: random.Random, split: str) -> Optional[dict]:
        short_city, code, town, postal, _ = rng.choice(self.same_name_resolved_pool)
        segs: List[Tuple[Optional[str], str]] = [
            ("CITY", short_city),
            ("TOWN", town),
            ("BLOCK", _gen_block(rng)),
        ]
        if rng.random() < 0.35:
            segs.append((None, " "))
            segs.append(("BUILDING", _gen_building(rng, split)))
        return self._verify_and_build(
            "same_name_resolved", segs, postal, "RESOLVED", code
        )

    def _gen_old_muni(self, rng: random.Random, split: str) -> Optional[dict]:
        if rng.random() < 0.72:
            rec = rng.choice(self.old_1to1)
            code = rec["codes"][0]
            pref = rec["pref"]
            old_name = rec["old_name"]
            city_surf = f"旧{old_name}" if rng.random() < 0.25 else old_name
            town, postal, _ = self._pick_town(rng, code)
            segs: List[Tuple[Optional[str], str]] = []
            if rng.random() < 0.65:
                segs.append(("PREF", pref))
            segs.append(("CITY", city_surf))
            if rng.random() < 0.85:
                segs.append(("TOWN", town))
            segs.append(("BLOCK", _gen_block(rng)))
            if rng.random() < 0.30:
                segs.append((None, " "))
                segs.append(("BUILDING", _gen_building(rng, split)))
            return self._verify_and_build(
                "old_muni", segs, postal, "RESOLVED", code, subtype="old_1to1"
            )
        old_name, pref, code, town, postal = rng.choice(self.old_split_towns)
        city_surf = f"旧{old_name}" if rng.random() < 0.20 else old_name
        segs = []
        if rng.random() < 0.65:
            segs.append(("PREF", pref))
        segs.extend([("CITY", city_surf), ("TOWN", town), ("BLOCK", _gen_block(rng))])
        if rng.random() < 0.30:
            segs.append((None, " "))
            segs.append(("BUILDING", _gen_building(rng, split)))
        return self._verify_and_build(
            "old_muni", segs, postal, "RESOLVED", code, subtype="old_split"
        )

    def _apply_old_kanji(self, rng: random.Random, s: str) -> str:
        cands = [(a, b) for a, b in TO_OLD_VARIANT if a in s]
        if not cands:
            return s
        a, b = rng.choice(cands)
        return s.replace(a, b, 1)

    def _gen_variant(self, rng: random.Random, split: str) -> Optional[dict]:
        mode = rng.choices(
            ["old_kanji", "ke_ga_no", "pref_short", "omit_county", "kana", "num_style"],
            weights=[0.22, 0.22, 0.16, 0.15, 0.10, 0.15],
        )[0]
        if mode == "ke_ga_no" and self.ke_ga_pool:
            code, town, postal = rng.choice(self.ke_ga_pool)
            m = self.munis[code]
            pref_s, city_s = m["pref"], m["city"]
            swaps = [
                ("ヶ", ["ケ", "が"]), ("ケ", ["ヶ", "が"]), ("が", ["ヶ", "ケ"]),
                ("の", ["ノ"]), ("ノ", ["の"]), ("ツ", ["ッ"]), ("ッ", ["ツ"]),
            ]
            town_v = town
            for src, tgts in swaps:
                if src in town_v:
                    town_v = town_v.replace(src, rng.choice(tgts), 1)
                    break
            block = _gen_block(rng, rng.choice(["fullwidth", "kanji", "hyphen"]))
            segs: List[Tuple[Optional[str], str]] = [
                ("PREF", pref_s), ("CITY", city_s), ("TOWN", town_v), ("BLOCK", block)
            ]
        elif mode == "old_kanji" and self.variant_char_codes:
            code = rng.choice(self.variant_char_codes)
            m = self.munis[code]
            town, postal, _ = self._pick_town(rng, code)
            pref_s = self._apply_old_kanji(rng, m["pref"])
            city_s = self._apply_old_kanji(rng, m["city"])
            town_v = self._apply_old_kanji(rng, town)
            if pref_s == m["pref"] and city_s == m["city"] and town_v == town:
                return None
            segs = [
                ("PREF", pref_s), ("CITY", city_s), ("TOWN", town_v), ("BLOCK", _gen_block(rng))
            ]
        elif mode == "pref_short":
            code = rng.choice(self.codes_with_town)
            m = self.munis[code]
            if m["pref"] == "北海道":
                return None
            town, postal, _ = self._pick_town(rng, code)
            pref_s = m["pref"][:-1]
            sep = " " if rng.random() < 0.5 else ""
            segs = [("PREF", pref_s)]
            if sep:
                segs.append((None, sep))
            segs.extend([("CITY", m["city"]), ("TOWN", town), ("BLOCK", _gen_block(rng))])
        elif mode == "omit_county":
            code = rng.choice(self.county_codes)
            m = self.munis[code]
            town, postal, _ = self._pick_town(rng, code)
            segs = [
                ("PREF", m["pref"]),
                ("CITY", m["base_city"]),
                ("TOWN", town),
                ("BLOCK", _gen_block(rng)),
            ]
        elif mode == "kana":
            code = rng.choice(self.codes_with_town)
            m = self.munis[code]
            town, postal, kana = self._pick_town(rng, code)
            if not kana:
                return None
            town_k = _katakana_to_hiragana(kana) if rng.random() < 0.75 else kana
            segs = [
                ("PREF", m["pref"]),
                ("CITY", m["city"]),
                ("TOWN", town_k),
                ("BLOCK", _gen_block(rng)),
            ]
        else:  # num_style
            code = rng.choice(self.codes_with_town)
            m = self.munis[code]
            town, postal, _ = self._pick_town(rng, code)
            block = _gen_block(rng, rng.choice(["fullwidth", "kanji", "no"]))
            segs = [
                ("PREF", m["pref"]),
                ("CITY", m["city"]),
                ("TOWN", town),
                ("BLOCK", block),
            ]

        if rng.random() < 0.30:
            segs.append((None, rng.choice([" ", "　"])))
            segs.append(("BUILDING", _gen_building(rng, split)))
        return self._verify_and_build(
            "variant", segs, postal, "RESOLVED", code, subtype=f"variant_{mode}"
        )

    def _gen_omit_block(self, rng: random.Random, split: str) -> Optional[dict]:
        _ = split
        if rng.random() < 0.70:
            code = rng.choice(self.codes_with_town)
            m = self.munis[code]
            town, postal, _ = self._pick_town(rng, code)
            segs: List[Tuple[Optional[str], str]] = [
                ("PREF", m["pref"]),
                ("CITY", m["city"]),
                ("TOWN", town),
            ]
        elif rng.random() < 0.65:
            code = rng.choice(self.unique_city_codes)
            m = self.munis[code]
            town, postal, _ = self._pick_town(rng, code)
            segs = [("CITY", m["city"]), ("TOWN", town)]
        else:
            code = rng.choice(self.codes_with_town)
            m = self.munis[code]
            postal = m["postals"][0] if m["postals"] else None
            segs = [("PREF", m["pref"]), ("CITY", m["city"])]
        return self._verify_and_build("omit_block", segs, postal, "RESOLVED", code)

    def _gen_reordered(self, rng: random.Random, split: str) -> Optional[dict]:
        code = rng.choice(self.codes_with_town)
        m = self.munis[code]
        town, postal, _ = self._pick_town(rng, code)
        block = _gen_block(rng)
        bld = _gen_building(rng, split)
        sp = rng.choice([" ", "　"])
        pattern = rng.choice(["bld_first", "reverse_tail", "bld_middle"])
        if pattern == "bld_first":
            segs = [
                ("BUILDING", bld),
                (None, sp),
                ("PREF", m["pref"]),
                ("CITY", m["city"]),
                ("TOWN", town),
                ("BLOCK", block),
            ]
        elif pattern == "reverse_tail":
            segs = [
                ("TOWN", town),
                ("BLOCK", block),
                (None, sp),
                ("PREF", m["pref"]),
                ("CITY", m["city"]),
            ]
        else:
            segs = [
                ("PREF", m["pref"]),
                ("CITY", m["city"]),
                (None, sp),
                ("BUILDING", bld),
                (None, sp),
                ("TOWN", town),
                ("BLOCK", block),
            ]
        return self._verify_and_build("reordered", segs, postal, "RESOLVED", code)

    def _gen_unresolvable_ambiguous(self, rng: random.Random, split: str) -> Optional[dict]:
        sub = rng.choices(
            ["ambig_city_no_town", "ambig_city_shared_town", "ambig_seirei_or_old_split", "ambig_town_only"],
            weights=[0.30, 0.25, 0.25, 0.20],
        )[0]
        if sub == "ambig_city_no_town":
            short_c = rng.choice(self.same_name_short_cities)
            segs: List[Tuple[Optional[str], str]] = [("CITY", short_c), ("BLOCK", _gen_block(rng))]
        elif sub == "ambig_city_shared_town":
            short_c, town = rng.choice(self.same_name_ambig_town_pool)
            segs = [("CITY", short_c), ("TOWN", town), ("BLOCK", _gen_block(rng))]
        elif sub == "ambig_seirei_or_old_split":
            if rng.random() < 0.70:
                pref, seirei = rng.choice(self.seirei_city_list)
                segs = []
                if rng.random() < 0.65:
                    segs.append(("PREF", pref))
                segs.extend([("CITY", seirei), ("BLOCK", _gen_block(rng))])
            else:
                old_name, pref = rng.choice(self.old_split_names)
                city_s = f"旧{old_name}" if rng.random() < 0.3 else old_name
                segs = []
                if rng.random() < 0.65:
                    segs.append(("PREF", pref))
                segs.extend([("CITY", city_s), ("BLOCK", _gen_block(rng))])
        else:  # ambig_town_only
            town = rng.choice(self.shared_standalone_towns)
            segs = [("TOWN", town), ("BLOCK", _gen_block(rng))]

        if rng.random() < 0.25:
            segs.append((None, " "))
            segs.append(("BUILDING", _gen_building(rng, split)))
        return self._verify_and_build(
            "unresolvable_ambiguous", segs, None, "UNRESOLVABLE", None, subtype=sub
        )

    def _gen_unresolvable_nonexistent(self, rng: random.Random, split: str) -> Optional[dict]:
        sub = rng.choices(
            ["fake_all", "real_pref_fake_city", "pref_city_mismatch", "real_city_fake_town"],
            weights=[0.25, 0.25, 0.25, 0.25],
        )[0]
        if sub == "fake_all":
            segs: List[Tuple[Optional[str], str]] = [
                ("PREF", rng.choice(FAKE_PREFS[split])),
                ("CITY", rng.choice(FAKE_CITIES[split])),
                ("TOWN", rng.choice(FAKE_TOWNS[split])),
                ("BLOCK", _gen_block(rng)),
            ]
        elif sub == "real_pref_fake_city":
            code = rng.choice(self.codes_with_town)
            segs = [
                ("PREF", self.munis[code]["pref"]),
                ("CITY", rng.choice(FAKE_CITIES[split])),
                ("TOWN", rng.choice(FAKE_TOWNS[split])),
                ("BLOCK", _gen_block(rng)),
            ]
        elif sub == "pref_city_mismatch":
            c1 = rng.choice(self.codes_with_town)
            c2 = rng.choice(self.codes_with_town)
            if self.munis[c1]["pref"] == self.munis[c2]["pref"]:
                return None
            town2, _, _ = self._pick_town(rng, c2)
            segs = [
                ("PREF", self.munis[c1]["pref"]),
                ("CITY", self.munis[c2]["city"]),
                ("TOWN", town2),
                ("BLOCK", _gen_block(rng)),
            ]
        else:  # real_city_fake_town
            code = rng.choice(self.codes_with_town)
            m = self.munis[code]
            segs = [
                ("PREF", m["pref"]),
                ("CITY", m["city"]),
                ("TOWN", rng.choice(FAKE_TOWNS[split])),
                ("BLOCK", _gen_block(rng)),
            ]

        if rng.random() < 0.25:
            segs.append((None, " "))
            segs.append(("BUILDING", _gen_building(rng, split)))
        return self._verify_and_build(
            "unresolvable_nonexistent", segs, None, "UNRESOLVABLE", None, subtype=sub
        )

    def generate_one(
        self,
        rng: random.Random,
        category: str,
        split: str,
        max_tries: int = 40,
    ) -> dict:
        """指定カテゴリ・分割の合成住所を1件生成する（リンカー整合性検証済み）。"""
        fn_map = {
            "standard": self._gen_standard,
            "omit_pref": self._gen_omit_pref,
            "same_name_resolved": self._gen_same_name_resolved,
            "old_muni": self._gen_old_muni,
            "variant": self._gen_variant,
            "omit_block": self._gen_omit_block,
            "reordered": self._gen_reordered,
            "unresolvable_ambiguous": self._gen_unresolvable_ambiguous,
            "unresolvable_nonexistent": self._gen_unresolvable_nonexistent,
        }
        fn = fn_map[category]
        for _ in range(max_tries):
            out = fn(rng, split)
            if out is not None:
                return out
        raise RuntimeError(f"住所生成に失敗しました: category={category}, split={split}")
