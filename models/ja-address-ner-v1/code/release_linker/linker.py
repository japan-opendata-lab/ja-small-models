#!/usr/bin/env python3
"""第7号 ja-address-ner-v1 第2段階 決定的辞書リンカー（差し替え可能な参照実装）。

設計方針:
  - 第1段階（トークン分類モデル）は住所抽出と5要素分解（PREF/CITY/TOWN/BLOCK/BUILDING）
    のみを行い、市区町村コードを直接出力しない。
  - 第2段階である本モジュールが、外部辞書（municipalities.json, town_index.json,
    old_municipalities.json）と照合して全国地方公共団体コード（5桁）を付与する。
  - 同名地名で一意に決まらない場合や矛盾・該当なしの場合は無理に1件を断言せず、
    status="UNRESOLVABLE"（理由コード付き）を返す。

辞書データの入手・組み立て方法:
  本リポジトリには辞書データ本体を同梱していません（行政区再編への追随や独自辞書への
  差し替えを容易にするため）。日本郵便「郵便番号データ (utf_ken_all.csv)」および
  デジタル庁「アドレス・ベース・レジストリ (ABR) 町字マスター」から、同梱の
  build_dictionary.py を用いて辞書ディレクトリを生成してください。
  詳細は同ディレクトリの DICTIONARY-SETUP.md を参照してください。
"""
import collections
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
import geo_norm as gn  # noqa: E402


class AddressLinker:
    """構成要素（PREF/CITY/TOWN）から 5桁市区町村コードを決定論的に付与する辞書リンカー。"""

    def __init__(self, master_dir: Optional[Union[str, Path]] = None) -> None:
        if master_dir is None:
            env_dir = os.environ.get("JA_ADDRESS_MASTER_DIR")
            if not env_dir:
                raise ValueError(
                    "master_dir（または環境変数 JA_ADDRESS_MASTER_DIR）を指定してください。"
                    "辞書は同梱の build_dictionary.py で公的データから生成できます。"
                )
            mdir = Path(env_dir)
        else:
            mdir = Path(master_dir)

        for req in ("municipalities.json", "town_index.json", "old_municipalities.json"):
            if not (mdir / req).exists():
                raise FileNotFoundError(
                    f"必須辞書ファイルが見つかりません: {mdir / req}"
                )

        self.munis: Dict[str, dict] = json.loads(
            (mdir / "municipalities.json").read_text(encoding="utf-8")
        )
        town_data: Dict[str, dict] = json.loads(
            (mdir / "town_index.json").read_text(encoding="utf-8")
        )
        old_list: List[dict] = json.loads(
            (mdir / "old_municipalities.json").read_text(encoding="utf-8")
        )

        # 現行市区町村インデックス
        self.exact_city_to_codes: Dict[str, Set[str]] = collections.defaultdict(set)
        self.seirei_city_to_codes: Dict[str, Set[str]] = collections.defaultdict(set)
        self.ward_only_to_codes: Dict[str, Set[str]] = collections.defaultdict(set)
        self.city_kana_to_codes: Dict[str, Set[str]] = collections.defaultdict(set)

        for code, m in self.munis.items():
            for alias in m["aliases"]:
                for k in gn.expand_geo_keys(alias):
                    self.exact_city_to_codes[k].add(code)
            if m.get("seirei_city"):
                for k in gn.expand_geo_keys(m["seirei_city"]):
                    self.seirei_city_to_codes[k].add(code)
            if m.get("ward"):
                for k in gn.expand_geo_keys(m["ward"]):
                    self.ward_only_to_codes[k].add(code)
            if m.get("city_kana"):
                self.city_kana_to_codes[m["city_kana"]].add(code)

        # 旧市町村・旧行政区インデックス
        self.old_muni_to_entries: Dict[str, List[dict]] = collections.defaultdict(list)
        for entry in old_list:
            for k in gn.expand_geo_keys(entry["old_name"]):
                self.old_muni_to_entries[k].append(entry)

        # 町域インデックス（コード別セット & 逆引き）
        self.town_norm_by_code: Dict[str, Set[str]] = {}
        self.town_kana_by_code: Dict[str, Set[str]] = {}
        self.town_norm_to_codes: Dict[str, Set[str]] = collections.defaultdict(set)
        self.town_kana_to_codes: Dict[str, Set[str]] = collections.defaultdict(set)

        for code, td in town_data.items():
            nset = set(td["norm_keys"])
            kset = set(td["kana_keys"])
            self.town_norm_by_code[code] = nset
            self.town_kana_by_code[code] = kset
            for k in nset:
                self.town_norm_to_codes[k].add(code)
            for kk in kset:
                self.town_kana_to_codes[kk].add(code)

    def _code_has_town(self, code: str, town: str) -> bool:
        t_keys = gn.extract_town_variants(town)
        if t_keys & self.town_norm_by_code.get(code, set()):
            return True
        if gn.is_all_kana(town):
            tk = gn.to_katakana(town)
            kset = self.town_kana_by_code.get(code, set())
            if tk in kset:
                return True
            for pfx in ("オオアザ", "アザ"):
                if tk.startswith(pfx) and tk[len(pfx) :] in kset:
                    return True
        return False

    def _lookup_town_codes(self, town: str) -> Set[str]:
        codes: Set[str] = set()
        for k in gn.extract_town_variants(town):
            codes.update(self.town_norm_to_codes.get(k, set()))
        if not codes and gn.is_all_kana(town):
            tk = gn.to_katakana(town)
            codes.update(self.town_kana_to_codes.get(tk, set()))
            for pfx in ("オオアザ", "アザ"):
                if tk.startswith(pfx):
                    codes.update(self.town_kana_to_codes.get(tk[len(pfx) :], set()))
        return codes

    def _resolved(self, code: str, reason: str, candidates: List[str]) -> dict:
        m = self.munis[code]
        return {
            "status": "RESOLVED",
            "municipality_code": code,
            "lg_code": m.get("lg_code"),
            "canonical_pref": m["pref"],
            "canonical_city": m["city"],
            "reason": reason,
            "candidate_codes": candidates,
        }

    @staticmethod
    def _unresolvable(reason: str, candidates: Optional[List[str]] = None) -> dict:
        return {
            "status": "UNRESOLVABLE",
            "municipality_code": None,
            "lg_code": None,
            "canonical_pref": None,
            "canonical_city": None,
            "reason": reason,
            "candidate_codes": candidates or [],
        }

    def link(
        self,
        pref: Optional[str] = None,
        city: Optional[str] = None,
        town: Optional[str] = None,
        block: Optional[str] = None,
        building: Optional[str] = None,
    ) -> dict:
        """分解済み構成要素から 5桁市区町村コードを決定論的にリンクする。"""
        _ = block, building
        pref_s = (pref or "").strip()
        city_s = (city or "").strip()
        town_s = (town or "").strip()

        # Step 1: PREF 検証
        pref_canon: Optional[str] = None
        if pref_s:
            pref_canon = gn.normalize_pref(pref_s)
            if pref_canon is None:
                return self._unresolvable("unknown_pref")

        # Step 2: CITY 照合
        city_cands: Optional[Set[str]] = None
        is_old_muni = False
        if city_s:
            raw_cands: Set[str] = set()
            c_keys = gn.expand_geo_keys(city_s)
            for k in c_keys:
                raw_cands.update(self.exact_city_to_codes.get(k, set()))
                raw_cands.update(self.seirei_city_to_codes.get(k, set()))
                raw_cands.update(self.ward_only_to_codes.get(k, set()))
            if not raw_cands and gn.is_all_kana(city_s):
                ck = gn.to_katakana(city_s)
                raw_cands.update(self.city_kana_to_codes.get(ck, set()))

            if not raw_cands:
                old_query = city_s[1:].strip() if city_s.startswith("旧") else city_s
                for k in gn.expand_geo_keys(old_query):
                    for entry in self.old_muni_to_entries.get(k, []):
                        raw_cands.update(entry["codes"])
                        is_old_muni = True

            if not raw_cands:
                return self._unresolvable("unknown_city")

            if pref_canon:
                filtered = {c for c in raw_cands if self.munis[c]["pref"] == pref_canon}
                if not filtered:
                    return self._unresolvable("pref_city_mismatch", sorted(raw_cands))
                city_cands = filtered
            else:
                city_cands = raw_cands

        # Step 3: TOWN 絞り込みおよび整合性検証
        if city_cands is not None:
            sorted_cands = sorted(city_cands)
            if len(sorted_cands) == 1:
                code = sorted_cands[0]
                if town_s and not self._code_has_town(code, town_s):
                    return self._unresolvable("town_not_in_municipality", sorted_cands)
                if is_old_muni:
                    reason = "old_muni_unique"
                elif pref_canon:
                    reason = "exact_pref_city"
                else:
                    reason = "unique_city"
                return self._resolved(code, reason, sorted_cands)

            if not town_s:
                return self._unresolvable("ambiguous_city_no_town", sorted_cands)

            matched = [c for c in sorted_cands if self._code_has_town(c, town_s)]
            if len(matched) == 1:
                reason = "old_muni_by_town" if is_old_muni else "disambiguated_by_town"
                return self._resolved(matched[0], reason, matched)
            if len(matched) > 1:
                return self._unresolvable("ambiguous_city_and_town", matched)
            return self._unresolvable("town_not_in_candidates", sorted_cands)

        # CITY が空の場合
        if town_s:
            t_cands = self._lookup_town_codes(town_s)
            if pref_canon:
                t_cands = {c for c in t_cands if self.munis[c]["pref"] == pref_canon}
            sorted_t = sorted(t_cands)
            if len(sorted_t) == 1:
                return self._resolved(sorted_t[0], "unique_town", sorted_t)
            if len(sorted_t) > 1:
                return self._unresolvable("ambiguous_town_only", sorted_t[:20])
            return self._unresolvable("unknown_town")

        return self._unresolvable("no_geographic_cue")

    def link_components(self, components: Dict[str, Optional[str]]) -> dict:
        """辞書形式の構成要素マップを受け取りリンクする。"""
        return self.link(
            pref=components.get("PREF"),
            city=components.get("CITY"),
            town=components.get("TOWN"),
            block=components.get("BLOCK"),
            building=components.get("BUILDING"),
        )
