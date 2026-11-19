#!/usr/bin/env python3
"""第2段階 決定的辞書リンカー（差し替え可能な参照実装）。

第1段階モデルが抽出・分解した5構成要素（PREF, CITY, TOWN, BLOCK, BUILDING）を
市区町村マスター・町域インデックス・旧市町村マスターと決定論的に照合し、
一意に特定できる場合のみ全国地方公共団体コード（5桁）を返す。
曖昧・矛盾・該当なしの場合は無理に決めず UNRESOLVABLE を返す。
"""
import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_norm as gn  # noqa: E402

DEFAULT_MASTER_DIR = Path(__file__).resolve().parent.parent / "data" / "master"


class AddressLinker:
    """構成要素（PREF/CITY/TOWN）から 5桁市区町村コードを決定論的に付与する辞書リンカー。"""

    def __init__(self, master_dir: Optional[Path] = None) -> None:
        mdir = Path(master_dir) if master_dir else DEFAULT_MASTER_DIR
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
            if m["seirei_city"]:
                for k in gn.expand_geo_keys(m["seirei_city"]):
                    self.seirei_city_to_codes[k].add(code)
            if m["ward"]:
                for k in gn.expand_geo_keys(m["ward"]):
                    self.ward_only_to_codes[k].add(code)
            if m["city_kana"]:
                self.city_kana_to_codes[m["city_kana"]].add(code)
                # 郡抜き・政令市名なしの簡易カナも補助登録
                if m["county"]:
                    county_k = gn.to_katakana(m["county"])
                    if m["city_kana"].startswith(county_k):
                        pass  # 漢字側と独立に利用
                if m["base_city"] and not m["ward"]:
                    # base_city のカナは utf_ken_all の city_kana と同一または郡抜き
                    pass

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
            "lg_code": m["lg_code"],
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
        _ = block, building  # リンカーは行政界（PREF/CITY/TOWN）のみで判断する
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
                # 旧市町村・旧行政区照合（「旧」接頭辞の除去も許容）
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

            # 複数候補あり（同名市/同名区/同名町村/政令市単独/複数区分割旧市）
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


def run_self_test() -> int:
    linker = AddressLinker()
    cases = [
        # (desc, kwargs, expected_status, expected_code_or_reason)
        ("標準フル住所", dict(pref="東京都", city="渋谷区", town="神南", block="1-2-3"), "RESOLVED", "13113"),
        ("県省略（一意市）", dict(city="横浜市西区", town="みなとみらい"), "RESOLVED", "14103"),
        ("同名市+固有町域（東京府中）", dict(city="府中市", town="宮西町"), "RESOLVED", "13206"),
        ("同名市+固有町域（広島府中）", dict(city="府中市", town="元町"), "RESOLVED", "34208"),
        ("同名区+固有町域（札幌中央区）", dict(city="中央区", town="北一条西"), "RESOLVED", "01101"),
        ("旧市町村1対1（浜松市中区）", dict(pref="静岡県", city="浜松市中区", town="元城町"), "RESOLVED", "22138"),
        ("旧市町村1対1（旧与野市）", dict(city="旧与野市", town="下落合"), "RESOLVED", "11105"),
        ("旧市町村分割型+固有町域（浦和市）", dict(city="浦和市", town="常盤"), "RESOLVED", "11107"),
        ("表記ゆれ（異体字・ヶ/が・県末字省略）", dict(pref="東京", city="渋谷区", town="千駄が谷"), "RESOLVED", "13113"),
        ("表記ゆれ（郡省略・異体字須惠）", dict(pref="福岡県", city="須惠町", town="旅石"), "RESOLVED", "40344"),
        ("番地欠落（町域まで）", dict(pref="大阪府", city="大阪市北区", town="梅田"), "RESOLVED", "27127"),
        ("同名市+町域なし（曖昧棄却）", dict(city="府中市", block="1-2-3"), "UNRESOLVABLE", "ambiguous_city_no_town"),
        ("同名区+共通町域（本町）", dict(city="中央区", town="本町"), "UNRESOLVABLE", "ambiguous_city_and_town"),
        ("政令市単独+番地のみ", dict(city="札幌市", block="1-2-3"), "UNRESOLVABLE", "ambiguous_city_no_town"),
        ("県と市の矛盾", dict(pref="東京都", city="横浜市中区", town="日本大通"), "UNRESOLVABLE", "pref_city_mismatch"),
        ("架空都道府県", dict(pref="架空県", city="渋谷区", town="神南"), "UNRESOLVABLE", "unknown_pref"),
        ("架空市区町村", dict(pref="東京都", city="新橋市", town="本町"), "UNRESOLVABLE", "unknown_city"),
        ("実在市+架空町域", dict(pref="東京都", city="渋谷区", town="存在しない架空町"), "UNRESOLVABLE", "town_not_in_municipality"),
        ("複数自治体共通の町域のみ", dict(town="本町", block="2-3-4"), "UNRESOLVABLE", "ambiguous_town_only"),
        ("県のみ", dict(pref="東京都"), "UNRESOLVABLE", "no_geographic_cue"),
    ]
    failed = 0
    for name, kw, exp_status, exp_val in cases:
        res = linker.link(**kw)
        ok = res["status"] == exp_status and (
            (res["municipality_code"] == exp_val)
            if exp_status == "RESOLVED"
            else (res["reason"] == exp_val)
        )
        if not ok:
            print(f"[FAIL] {name}: got={res} expected=({exp_status}, {exp_val})")
            failed += 1
        else:
            print(f"[PASS] {name} -> {res['status']} ({res['municipality_code'] or res['reason']})")
    assert failed == 0, f"{failed} self-test cases failed"
    print(f"All {len(cases)} self-test cases passed!")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return run_self_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
