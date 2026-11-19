#!/usr/bin/env python3
"""公的オープンデータ（日本郵便 utf_ken_all + デジタル庁 ABR 町字マスター）から
第2段階リンカーおよび合成データ用の住所マスターを構築する。

出力（data/master/）:
  - municipalities.json : 現行 1,892 市区町村マスター
  - town_index.json     : コード別 合成用クリーン町域 + 照合用正規化/カナキー
  - old_municipalities.json : 旧市町村・旧行政区マスター（現行名非衝突を全件検証）
  - master-stats.json   : 構築統計
"""
import collections
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_norm as gn  # noqa: E402

KEN_ALL_PATH = Path(os.environ.get("KEN_ALL_PATH", "data/raw_parts/utf_ken_all.csv"))  # Japan Post postal-code CSV
ABR_PARQUET_PATH = Path(os.environ.get("ABR_PARQUET_PATH", "data/japan_address_registry_town_master.parquet"))  # town master built from the Address Base Registry
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "master"

# 旧市町村・旧行政区マスター（総務省 市町村合併・政令市行政区再編の事実情報）
# codes が 1 件のものは 1 対 1 確定型、複数件のものは町域によって一意に絞り込む分割型。
RAW_OLD_MUNICIPALITIES: List[Tuple[str, str, List[str], str]] = [
    # 政令市 行政区再編（浜松市 2024年再編）
    ("浜松市中区", "静岡県", ["22138"], "2024年再編->中央区"),
    ("浜松市東区", "静岡県", ["22138"], "2024年再編->中央区"),
    ("浜松市西区", "静岡県", ["22138"], "2024年再編->中央区"),
    ("浜松市南区", "静岡県", ["22138"], "2024年再編->中央区"),
    ("浜松市浜北区", "静岡県", ["22139"], "2024年再編->浜名区"),
    ("浜松市北区", "静岡県", ["22138", "22139"], "2024年再編->中央区/浜名区に分割"),
    # 埼玉県（さいたま市合併・ふじみ野・川口・久喜・熊谷・春日部・深谷）
    ("与野市", "埼玉県", ["11105"], "さいたま市中央区へ"),
    ("岩槻市", "埼玉県", ["11110"], "さいたま市岩槻区へ"),
    ("浦和市", "埼玉県", ["11106", "11107", "11108", "11109"], "さいたま市桜/浦和/南/緑区へ分割"),
    ("大宮市", "埼玉県", ["11101", "11102", "11103", "11104"], "さいたま市西/北/大宮/見沼区へ分割"),
    ("上福岡市", "埼玉県", ["11245"], "ふじみ野市"),
    ("入間郡大井町", "埼玉県", ["11245"], "ふじみ野市"),
    ("鳩ヶ谷市", "埼玉県", ["11203"], "川口市へ編入"),
    ("北葛飾郡栗橋町", "埼玉県", ["11232"], "久喜市"),
    ("北葛飾郡鷲宮町", "埼玉県", ["11232"], "久喜市"),
    ("大里郡妻沼町", "埼玉県", ["11202"], "熊谷市"),
    ("北葛飾郡庄和町", "埼玉県", ["11214"], "春日部市"),
    ("大里郡花園町", "埼玉県", ["11218"], "深谷市"),
    # 北海道
    ("亀田郡戸井町", "北海道", ["01202"], "函館市へ編入"),
    ("上磯郡上磯町", "北海道", ["01236"], "北斗市"),
    ("上磯郡大野町", "北海道", ["01236"], "北斗市"),
    ("阿寒郡阿寒町", "北海道", ["01206"], "釧路市"),
    ("白糠郡音別町", "北海道", ["01206"], "釧路市"),
    ("静内郡静内町", "北海道", ["01610"], "新ひだか町"),
    ("三石郡三石町", "北海道", ["01610"], "新ひだか町"),
    # 東北
    ("南津軽郡浪岡町", "青森県", ["02201"], "青森市"),
    ("南津軽郡尾上町", "青森県", ["02210"], "平川市"),
    ("南津軽郡平賀町", "青森県", ["02210"], "平川市"),
    ("西津軽郡木造町", "青森県", ["02209"], "つがる市"),
    ("古川市", "宮城県", ["04215"], "大崎市"),
    ("栗原郡築館町", "宮城県", ["04213"], "栗原市"),
    ("登米郡迫町", "宮城県", ["04212"], "登米市"),
    ("桃生郡矢本町", "宮城県", ["04214"], "東松島市"),
    ("原町市", "福島県", ["07212"], "南相馬市"),
    ("伊達郡保原町", "福島県", ["07213"], "伊達市"),
    ("伊達郡梁川町", "福島県", ["07213"], "伊達市"),
    ("田村郡船引町", "福島県", ["07211"], "田村市"),
    # 北関東
    ("水海道市", "茨城県", ["08211"], "常総市"),
    ("岩井市", "茨城県", ["08228"], "坂東市"),
    ("下館市", "茨城県", ["08227"], "筑西市"),
    ("勝田市", "茨城県", ["08221"], "ひたちなか市"),
    ("那珂湊市", "茨城県", ["08221"], "ひたちなか市"),
    ("鹿島郡波崎町", "茨城県", ["08232"], "神栖市"),
    ("筑波郡谷田部町", "茨城県", ["08220"], "つくば市"),
    ("新治郡桜村", "茨城県", ["08220"], "つくば市"),
    ("今市市", "栃木県", ["09206"], "日光市"),
    ("黒磯市", "栃木県", ["09213"], "那須塩原市"),
    ("塩谷郡氏家町", "栃木県", ["09214"], "さくら市"),
    ("新田郡新田町", "群馬県", ["10205"], "太田市"),
    ("新田郡尾島町", "群馬県", ["10205"], "太田市"),
    ("佐波郡境町", "群馬県", ["10204"], "伊勢崎市"),
    ("佐波郡赤堀町", "群馬県", ["10204"], "伊勢崎市"),
    ("群馬郡榛名町", "群馬県", ["10202"], "高崎市"),
    ("勢多郡大胡町", "群馬県", ["10201"], "前橋市"),
    # 南関東
    ("東葛飾郡関宿町", "千葉県", ["12208"], "野田市"),
    ("八日市場市", "千葉県", ["12235"], "匝瑳市"),
    ("佐原市", "千葉県", ["12236"], "香取市"),
    ("夷隅郡大原町", "千葉県", ["12238"], "いすみ市"),
    ("田無市", "東京都", ["13229"], "西東京市"),
    ("保谷市", "東京都", ["13229"], "西東京市"),
    ("秋川市", "東京都", ["13228"], "あきる野市"),
    ("西多摩郡五日市町", "東京都", ["13228"], "あきる野市"),
    ("津久井郡城山町", "神奈川県", ["14151"], "相模原市緑区"),
    ("津久井郡津久井町", "神奈川県", ["14151"], "相模原市緑区"),
    ("津久井郡相模湖町", "神奈川県", ["14151"], "相模原市緑区"),
    ("津久井郡藤野町", "神奈川県", ["14151"], "相模原市緑区"),
    # 甲信越・北陸
    ("新津市", "新潟県", ["15105"], "新潟市秋葉区"),
    ("中蒲原郡小須戸町", "新潟県", ["15105"], "新潟市秋葉区"),
    ("白根市", "新潟県", ["15106"], "新潟市南区"),
    ("中蒲原郡亀田町", "新潟県", ["15104"], "新潟市江南区"),
    ("豊栄市", "新潟県", ["15101"], "新潟市北区"),
    ("西蒲原郡巻町", "新潟県", ["15108"], "新潟市西蒲区"),
    ("松任市", "石川県", ["17210"], "白山市"),
    ("石川郡鶴来町", "石川県", ["17210"], "白山市"),
    ("河北郡宇ノ気町", "石川県", ["17209"], "かほく市"),
    ("更埴市", "長野県", ["20218"], "千曲市"),
    ("南安曇郡豊科町", "長野県", ["20220"], "安曇野市"),
    ("南安曇郡穂高町", "長野県", ["20220"], "安曇野市"),
    ("小県郡東部町", "長野県", ["20219"], "東御市"),
    ("小県郡丸子町", "長野県", ["20203"], "上田市"),
    # 東海
    ("清水市", "静岡県", ["22103"], "静岡市清水区"),
    ("庵原郡蒲原町", "静岡県", ["22103"], "静岡市清水区"),
    ("浜北市", "静岡県", ["22139"], "浜松市浜名区"),
    ("天竜市", "静岡県", ["22140"], "浜松市天竜区"),
    ("田方郡韮山町", "静岡県", ["22225"], "伊豆の国市"),
    ("田方郡大仁町", "静岡県", ["22225"], "伊豆の国市"),
    ("田方郡修善寺町", "静岡県", ["22222"], "伊豆市"),
    ("小笠郡大須賀町", "静岡県", ["22213"], "掛川市"),
    ("尾西市", "愛知県", ["23203"], "一宮市"),
    ("葉栗郡木曽川町", "愛知県", ["23203"], "一宮市"),
    ("西春日井郡西枇杷島町", "愛知県", ["23233"], "清須市"),
    ("西春日井郡清洲町", "愛知県", ["23233"], "清須市"),
    ("海部郡七宝町", "愛知県", ["23237"], "あま市"),
    ("海部郡甚目寺町", "愛知県", ["23237"], "あま市"),
    ("幡豆郡一色町", "愛知県", ["23213"], "西尾市"),
    ("幡豆郡吉良町", "愛知県", ["23213"], "西尾市"),
    ("西春日井郡師勝町", "愛知県", ["23234"], "北名古屋市"),
    ("西春日井郡西春町", "愛知県", ["23234"], "北名古屋市"),
    ("上野市", "三重県", ["24216"], "伊賀市"),
    ("久居市", "三重県", ["24201"], "津市"),
    ("志摩郡阿児町", "三重県", ["24215"], "志摩市"),
    # 近畿
    ("八日市市", "滋賀県", ["25213"], "東近江市"),
    ("甲賀郡水口町", "滋賀県", ["25209"], "甲賀市"),
    ("高島郡今津町", "滋賀県", ["25212"], "高島市"),
    ("船井郡園部町", "京都府", ["26213"], "南丹市"),
    ("竹野郡網野町", "京都府", ["26212"], "京丹後市"),
    ("相楽郡木津町", "京都府", ["26214"], "木津川市"),
    ("南河内郡美原町", "大阪府", ["27147"], "堺市美原区"),
    ("龍野市", "兵庫県", ["28229"], "たつの市"),
    ("篠山市", "兵庫県", ["28221"], "丹波篠山市"),
    ("氷上郡柏原町", "兵庫県", ["28223"], "丹波市"),
    ("三原郡南淡町", "兵庫県", ["28224"], "南あわじ市"),
    ("津名郡津名町", "兵庫県", ["28226"], "淡路市"),
    ("宍粟郡山崎町", "兵庫県", ["28227"], "宍粟市"),
    ("加東郡社町", "兵庫県", ["28228"], "加東市"),
    # 中国・四国
    ("児島市", "岡山県", ["33202"], "倉敷市"),
    ("玉島市", "岡山県", ["33202"], "倉敷市"),
    ("西大寺市", "岡山県", ["33103"], "岡山市東区"),
    ("因島市", "広島県", ["34205"], "尾道市"),
    ("芦品郡新市町", "広島県", ["34207"], "福山市"),
    ("安芸郡音戸町", "広島県", ["34202"], "呉市"),
    ("徳山市", "山口県", ["35215"], "周南市"),
    ("新南陽市", "山口県", ["35215"], "周南市"),
    ("吉敷郡小郡町", "山口県", ["35203"], "山口市"),
    ("麻植郡鴨島町", "徳島県", ["36205"], "吉野川市"),
    ("美馬郡脇町", "徳島県", ["36207"], "美馬市"),
    ("大川郡志度町", "香川県", ["37206"], "さぬき市"),
    ("大川郡白鳥町", "香川県", ["37207"], "東かがわ市"),
    ("川之江市", "愛媛県", ["38213"], "四国中央市"),
    ("伊予三島市", "愛媛県", ["38213"], "四国中央市"),
    ("東予市", "愛媛県", ["38206"], "西条市"),
    ("北条市", "愛媛県", ["38201"], "松山市"),
    # 九州・沖縄
    ("前原市", "福岡県", ["40230"], "糸島市"),
    ("糸島郡志摩町", "福岡県", ["40230"], "糸島市"),
    ("甘木市", "福岡県", ["40228"], "朝倉市"),
    ("山田市", "福岡県", ["40227"], "嘉麻市"),
    ("本渡市", "熊本県", ["43215"], "天草市"),
    ("牛深市", "熊本県", ["43215"], "天草市"),
    ("宇土郡松橋町", "熊本県", ["43213"], "宇城市"),
    ("具志川市", "沖縄県", ["47213"], "うるま市"),
    ("石川市", "沖縄県", ["47213"], "うるま市"),
    ("平良市", "沖縄県", ["47214"], "宮古島市"),
    ("島尻郡佐敷町", "沖縄県", ["47215"], "南城市"),
]

_BAD_TOWN_WORDS = ("以下に掲載がない場合", "の次に番地がくる場合", "一円", "甲、乙")


def clean_ken_all_town(raw_town: str) -> str:
    """utf_ken_all.csv の町域文字列から注記や括弧・高層ビル階別行を除去してクリーンな町域名を返す。"""
    t = raw_town.strip()
    if not t or any(w in t for w in _BAD_TOWN_WORDS):
        return ""
    if "（" in t:
        _, paren = t.split("（", 1)
        # 超高層ビル階数別郵便番号行を除外
        if "階" in paren or "ビル" in paren:
            return ""
        t = t.split("（", 1)[0].strip()
    if not t or any(w in t for w in _BAD_TOWN_WORDS):
        return ""
    if re.fullmatch(r"[0-9０-９一二三四五六七八九十]+丁目?", t):
        return ""
    if "、" in t or "〜" in t:
        return ""
    return t


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 日本郵便 utf_ken_all.csv 読込
    ken_munis: Dict[str, dict] = {}
    ken_clean_towns: Dict[str, Dict[str, Tuple[str, str]]] = collections.defaultdict(dict)
    ken_raw_towns: Dict[str, Set[str]] = collections.defaultdict(set)
    ken_kana_towns: Dict[str, Set[str]] = collections.defaultdict(set)
    ken_postals: Dict[str, List[str]] = collections.defaultdict(list)

    with open(KEN_ALL_PATH, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 9:
                continue
            code = row[0].strip().zfill(5)
            postal = row[2].strip().zfill(7)
            pref_kana = row[3].strip()
            city_kana = row[4].strip()
            town_kana_raw = row[5].strip()
            pref = row[6].strip()
            city = row[7].strip()
            town_raw = row[8].strip()

            if code not in ken_munis:
                ken_munis[code] = {
                    "pref": pref,
                    "ken_city": city,
                    "pref_kana": pref_kana,
                    "city_kana": city_kana,
                }
            if postal and len(ken_postals[code]) < 20 and postal not in ken_postals[code]:
                ken_postals[code].append(postal)

            cleaned = clean_ken_all_town(town_raw)
            if cleaned:
                ken_raw_towns[code].add(cleaned)
                tk = town_kana_raw.split("(", 1)[0].split("（", 1)[0].strip()
                if tk and tk != "イカニケイサイガナイバアイ":
                    ken_kana_towns[code].add(gn.to_katakana(tk))
                if cleaned not in ken_clean_towns[code]:
                    ken_clean_towns[code][cleaned] = (postal, gn.to_katakana(tk) if tk else "")

    # 2. デジタル庁 ABR 町字マスター読込
    table = pq.read_table(
        ABR_PARQUET_PATH,
        columns=[
            "municipality_code",
            "lg_code",
            "prefecture",
            "county",
            "city",
            "ward",
            "oaza_cho",
            "oaza_cho_kana",
            "koaza",
            "koaza_kana",
            "status_flg",
            "postal_code",
        ],
    )
    abr_dict = table.to_pydict()
    abr_munis: Dict[str, dict] = {}
    abr_towns: Dict[str, Set[str]] = collections.defaultdict(set)
    abr_kana: Dict[str, Set[str]] = collections.defaultdict(set)
    abr_clean_fallback: Dict[str, Dict[str, Tuple[str, str]]] = collections.defaultdict(dict)

    n_abr = len(abr_dict["municipality_code"])
    for i in range(n_abr):
        code = str(abr_dict["municipality_code"][i]).strip().zfill(5)
        lg_code = str(abr_dict["lg_code"][i]).strip()
        pref = abr_dict["prefecture"][i] or ""
        county = abr_dict["county"][i] or ""
        city = abr_dict["city"][i] or ""
        ward = abr_dict["ward"][i] or ""
        oaza = abr_dict["oaza_cho"][i] or ""
        oaza_kana = abr_dict["oaza_cho_kana"][i] or ""
        koaza = abr_dict["koaza"][i] or ""
        koaza_kana = abr_dict["koaza_kana"][i] or ""
        sf = abr_dict["status_flg"][i]
        pcode = str(abr_dict["postal_code"][i] or "").strip()

        if code not in abr_munis:
            abr_munis[code] = {
                "lg_code": lg_code,
                "pref": pref,
                "county": county,
                "abr_city": city,
                "ward": ward,
            }

        if sf in (0, 1):
            t_surf = oaza if oaza else koaza
            t_kana = oaza_kana if oaza else koaza_kana
            if t_surf:
                abr_towns[code].add(t_surf)
                if oaza and koaza and oaza != koaza:
                    abr_towns[code].add(oaza + koaza)
                if t_kana:
                    abr_kana[code].add(gn.to_katakana(t_kana))
                if t_surf not in abr_clean_fallback[code] and not any(
                    w in t_surf for w in _BAD_TOWN_WORDS
                ):
                    abr_clean_fallback[code][t_surf] = (
                        pcode if pcode else (ken_postals[code][0] if ken_postals[code] else ""),
                        gn.to_katakana(t_kana) if t_kana else "",
                    )

    # 5桁コードの完全一致を検証
    assert set(ken_munis.keys()) == set(abr_munis.keys()), "ken_all と ABR で 5桁コード不一致"
    assert len(ken_munis) == 1892, f"期待件数 1892 に対し {len(ken_munis)}"

    # 3. 統合 municipalities.json 構築
    municipalities: Dict[str, dict] = {}
    existing_full_names: Set[str] = set()
    existing_short_names: Set[str] = set()

    for code in sorted(ken_munis.keys()):
        km = ken_munis[code]
        am = abr_munis[code]
        assert km["pref"] == am["pref"], f"都道府県不一致: {code}"

        pref = am["pref"]
        county = am["county"]
        base_city = am["abr_city"]
        ken_city = km["ken_city"]
        ward = am["ward"]
        # 政令市の場合 base_city は "札幌市中央区" だが、政令市名単独 ("札幌市") も記録
        seirei_city = base_city[: -len(ward)] if (ward and base_city.endswith(ward)) else ""

        aliases = sorted({ken_city, base_city, (county + base_city) if county else base_city} - {""})
        existing_full_names.add(gn.normalize_geo(ken_city))
        existing_short_names.add(gn.normalize_geo(base_city))

        municipalities[code] = {
            "code": code,
            "lg_code": am["lg_code"],
            "pref": pref,
            "county": county,
            "city": ken_city,
            "base_city": base_city,
            "ward": ward,
            "seirei_city": seirei_city,
            "pref_kana": gn.to_katakana(km["pref_kana"]),
            "city_kana": gn.to_katakana(km["city_kana"]),
            "postals": ken_postals[code],
            "aliases": aliases,
        }

    # 4. town_index.json 構築
    town_index: Dict[str, dict] = {}
    total_clean_towns = 0
    total_norm_keys = 0
    empty_clean_codes = []

    for code in sorted(ken_munis.keys()):
        clean_map = ken_clean_towns.get(code) or abr_clean_fallback.get(code) or {}
        clean_list = [
            [t, p, k] for t, (p, k) in sorted(clean_map.items(), key=lambda x: x[0])
        ]
        if not clean_list:
            empty_clean_codes.append(code)

        norm_keys: Set[str] = set()
        for t in ken_raw_towns.get(code, set()) | abr_towns.get(code, set()):
            norm_keys.update(gn.extract_town_variants(t))

        kana_keys = sorted(
            {k for k in (ken_kana_towns.get(code, set()) | abr_kana.get(code, set())) if k}
        )

        total_clean_towns += len(clean_list)
        total_norm_keys += len(norm_keys)
        town_index[code] = {
            "clean_towns": clean_list,
            "norm_keys": sorted(norm_keys),
            "kana_keys": kana_keys,
        }

    # 5. old_municipalities.json 構築と厳格検証
    old_records = []
    seen_old_names: Set[str] = set()
    for old_name, pref, codes, note in RAW_OLD_MUNICIPALITIES:
        assert pref in gn.PREFECTURES, f"不正な都道府県: {pref}"
        norm_old = gn.normalize_geo(old_name)
        assert norm_old not in seen_old_names, f"旧市町村名重複: {old_name}"
        seen_old_names.add(norm_old)

        # 現行の市区町村名（完全名および郡抜き名）と衝突しないことを確認
        assert norm_old not in existing_full_names, f"現行 city と衝突: {old_name}"
        assert norm_old not in existing_short_names, f"現行 base_city と衝突: {old_name}"

        for c in codes:
            assert c in municipalities, f"存在しないコード: {c} ({old_name})"
            assert municipalities[c]["pref"] == pref, (
                f"都道府県不整合: {old_name} {pref} != {municipalities[c]['pref']}"
            )

        # 複数区分割型は、各区に固有な町域がちゃんと存在するか確認
        if len(codes) > 1:
            key_counts = collections.Counter()
            for c in codes:
                for t_entry in town_index[c]["clean_towns"]:
                    for k in gn.extract_town_variants(t_entry[0]):
                        key_counts[k] += 1
            for c in codes:
                unique_t = [
                    t_entry[0]
                    for t_entry in town_index[c]["clean_towns"]
                    if all(key_counts[k] == 1 for k in gn.extract_town_variants(t_entry[0]))
                ]
                assert unique_t, f"分割型旧市 {old_name} の {c} に固有町域がない"

        old_records.append(
            {
                "old_name": old_name,
                "pref": pref,
                "codes": codes,
                "note": note,
            }
        )

    # 出力書込
    (OUT_DIR / "municipalities.json").write_text(
        json.dumps(municipalities, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "town_index.json").write_text(
        json.dumps(town_index, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "old_municipalities.json").write_text(
        json.dumps(old_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats = {
        "municipality_count": len(municipalities),
        "prefecture_count": len({m["pref"] for m in municipalities.values()}),
        "total_clean_towns": total_clean_towns,
        "total_norm_keys": total_norm_keys,
        "no_town_municipality_count": len(empty_clean_codes),
        "old_municipalities_count": len(old_records),
        "old_1to1_count": sum(1 for r in old_records if len(r["codes"]) == 1),
        "old_split_count": sum(1 for r in old_records if len(r["codes"]) > 1),
    }
    (OUT_DIR / "master-stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
