#!/usr/bin/env python3
"""公的オープンデータから第2段階辞書リンカー用の辞書ディレクトリを組み立てる公開用スクリプト。

【出典データと入手先・ライセンス】
1. 日本郵便「郵便番号データ（住所の郵便番号・UTF-8・1レコード1行形式 utf_ken_all.csv）」【必須】
   - 入手先: https://www.post.japanpost.jp/zipcode/dl/utf-zip.html
   - ライセンス: 日本郵便は「郵便番号データに限っては著作権を主張しません」と明記（商用利用可）
2. デジタル庁「アドレス・ベース・レジストリ（ABR）町字マスター」【任意・推奨】
   - 入手先: https://catalog.registries.digital.go.jp/ （町字マスター CSV / Parquet）
   - ライセンス: 公共データ利用規約（第1.0版）PDL1.0（商用利用可・出典記載）
   ※ --abr を省略した場合も utf_ken_all.csv のみから全 1,892 市区町村辞書を構築できます。

【実行例】
  python build_dictionary.py --ken-all utf_ken_all.csv --out ./dict_master
  python build_dictionary.py --ken-all utf_ken_all.csv --abr abr_town_master.parquet --out ./dict_master
"""
import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
import geo_norm as gn  # noqa: E402

# 総務省 市町村合併・政令市行政区再編に基づく旧市町村・旧行政区マスター（事実情報）
RAW_OLD_MUNICIPALITIES: List[Tuple[str, str, List[str], str]] = [
    ("浜松市中区", "静岡県", ["22138"], "2024年再編->中央区"),
    ("浜松市東区", "静岡県", ["22138"], "2024年再編->中央区"),
    ("浜松市西区", "静岡県", ["22138"], "2024年再編->中央区"),
    ("浜松市南区", "静岡県", ["22138"], "2024年再編->中央区"),
    ("浜松市浜北区", "静岡県", ["22139"], "2024年再編->浜名区"),
    ("浜松市北区", "静岡県", ["22138", "22139"], "2024年再編->中央区/浜名区に分割"),
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
    ("亀田郡戸井町", "北海道", ["01202"], "函館市へ編入"),
    ("上磯郡上磯町", "北海道", ["01236"], "北斗市"),
    ("上磯郡大野町", "北海道", ["01236"], "北斗市"),
    ("阿寒郡阿寒町", "北海道", ["01206"], "釧路市"),
    ("白糠郡音別町", "北海道", ["01206"], "釧路市"),
    ("静内郡静内町", "北海道", ["01610"], "新ひだか町"),
    ("三石郡三石町", "北海道", ["01610"], "新ひだか町"),
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
_COUNTY_RE = re.compile(r"^(.+?郡)([^市]+(?:町|村))$")
_WARD_RE = re.compile(r"^(.+?市)(.+?区)$")


def clean_ken_all_town(raw_town: str) -> str:
    """utf_ken_all.csv の町域文字列から注記や括弧・高層ビル階別行を除去する。"""
    t = raw_town.strip()
    if not t or any(w in t for w in _BAD_TOWN_WORDS):
        return ""
    if "（" in t:
        _, paren = t.split("（", 1)
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


def _split_city_fallback(ken_city: str) -> Tuple[str, str, str]:
    """ABR 非指定時に utf_ken_all.csv の市区町村名から (county, base_city, ward) を推定する。"""
    m_c = _COUNTY_RE.match(ken_city)
    if m_c:
        return m_c.group(1), m_c.group(2), ""
    m_w = _WARD_RE.match(ken_city)
    if m_w:
        return "", ken_city, m_w.group(2)
    return "", ken_city, ""


def _read_abr_table(abr_path: Path) -> Tuple[Dict[str, dict], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """ABR 町字マスター（Parquet または CSV）を読み込む。"""
    abr_munis: Dict[str, dict] = {}
    abr_towns: Dict[str, Set[str]] = collections.defaultdict(set)
    abr_kana: Dict[str, Set[str]] = collections.defaultdict(set)

    def _ingest_row(code5, lg_code, pref, county, city, ward, oaza, oaza_kana, koaza, koaza_kana, sf):
        code = str(code5 or "").strip()[:5].zfill(5)
        if not code or code == "00000":
            return
        if code not in abr_munis:
            abr_munis[code] = {
                "lg_code": str(lg_code or "").strip(),
                "pref": (pref or "").strip(),
                "county": (county or "").strip(),
                "abr_city": (city or "").strip(),
                "ward": (ward or "").strip(),
            }
        try:
            sf_int = int(sf) if sf is not None and str(sf).strip() != "" else 1
        except ValueError:
            sf_int = 1
        if sf_int in (0, 1):
            oaza_s = (oaza or "").strip()
            koaza_s = (koaza or "").strip()
            t_surf = oaza_s if oaza_s else koaza_s
            t_kana = (oaza_kana or "").strip() if oaza_s else (koaza_kana or "").strip()
            if t_surf:
                abr_towns[code].add(t_surf)
                if oaza_s and koaza_s and oaza_s != koaza_s:
                    abr_towns[code].add(oaza_s + koaza_s)
                if t_kana:
                    abr_kana[code].add(gn.to_katakana(t_kana))

    if abr_path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        cols = [
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
        ]
        d = pq.read_table(abr_path, columns=cols).to_pydict()
        for i in range(len(d["municipality_code"])):
            _ingest_row(
                d["municipality_code"][i],
                d["lg_code"][i],
                d["prefecture"][i],
                d["county"][i],
                d["city"][i],
                d["ward"][i],
                d["oaza_cho"][i],
                d["oaza_cho_kana"][i],
                d["koaza"][i],
                d["koaza_kana"][i],
                d["status_flg"][i],
            )
    else:
        with open(abr_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                lg = r.get("lg_code", "")
                mc = r.get("municipality_code") or (str(lg)[:5] if lg else "")
                _ingest_row(
                    mc,
                    lg,
                    r.get("prefecture") or r.get("pref", ""),
                    r.get("county", ""),
                    r.get("city", ""),
                    r.get("ward", ""),
                    r.get("oaza_cho", ""),
                    r.get("oaza_cho_kana", ""),
                    r.get("koaza", ""),
                    r.get("koaza_kana", ""),
                    r.get("status_flg", 1),
                )
    return abr_munis, abr_towns, abr_kana


def build_dictionary(ken_all_path: Path, out_dir: Path, abr_path: Optional[Path] = None) -> dict:
    """日本郵便データ（＋任意で ABR 町字マスター）からリンカー用辞書を構築する。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    ken_munis: Dict[str, dict] = {}
    ken_clean_towns: Dict[str, Dict[str, Tuple[str, str]]] = collections.defaultdict(dict)
    ken_raw_towns: Dict[str, Set[str]] = collections.defaultdict(set)
    ken_kana_towns: Dict[str, Set[str]] = collections.defaultdict(set)
    ken_postals: Dict[str, List[str]] = collections.defaultdict(list)

    with open(ken_all_path, encoding="utf-8-sig") as f:
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

    abr_munis: Dict[str, dict] = {}
    abr_towns: Dict[str, Set[str]] = collections.defaultdict(set)
    abr_kana: Dict[str, Set[str]] = collections.defaultdict(set)
    if abr_path is not None:
        abr_munis, abr_towns, abr_kana = _read_abr_table(abr_path)

    municipalities: Dict[str, dict] = {}
    existing_full_names: Set[str] = set()
    existing_short_names: Set[str] = set()

    for code in sorted(ken_munis.keys()):
        km = ken_munis[code]
        am = abr_munis.get(code)
        pref = am["pref"] if am and am["pref"] else km["pref"]
        ken_city = km["ken_city"]

        if am and am["abr_city"]:
            county = am["county"]
            base_city = am["abr_city"]
            ward = am["ward"]
            lg_code = am["lg_code"]
        else:
            county, base_city, ward = _split_city_fallback(ken_city)
            lg_code = None

        seirei_city = base_city[: -len(ward)] if (ward and base_city.endswith(ward)) else ""
        aliases = sorted({ken_city, base_city, (county + base_city) if county else base_city} - {""})
        existing_full_names.add(gn.normalize_geo(ken_city))
        existing_short_names.add(gn.normalize_geo(base_city))

        municipalities[code] = {
            "code": code,
            "lg_code": lg_code,
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

    town_index: Dict[str, dict] = {}
    total_norm_keys = 0
    for code in sorted(ken_munis.keys()):
        norm_keys: Set[str] = set()
        for t in ken_raw_towns.get(code, set()) | abr_towns.get(code, set()):
            norm_keys.update(gn.extract_town_variants(t))
        kana_keys = sorted(
            {k for k in (ken_kana_towns.get(code, set()) | abr_kana.get(code, set())) if k}
        )
        total_norm_keys += len(norm_keys)
        town_index[code] = {
            "norm_keys": sorted(norm_keys),
            "kana_keys": kana_keys,
        }

    old_records = []
    for old_name, pref, codes, note in RAW_OLD_MUNICIPALITIES:
        valid_codes = [c for c in codes if c in municipalities and municipalities[c]["pref"] == pref]
        if not valid_codes:
            continue
        norm_old = gn.normalize_geo(old_name)
        if norm_old in existing_full_names or norm_old in existing_short_names:
            continue
        old_records.append(
            {"old_name": old_name, "pref": pref, "codes": valid_codes, "note": note}
        )

    (out_dir / "municipalities.json").write_text(
        json.dumps(municipalities, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "town_index.json").write_text(
        json.dumps(town_index, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "old_municipalities.json").write_text(
        json.dumps(old_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats = {
        "municipality_count": len(municipalities),
        "prefecture_count": len({m["pref"] for m in municipalities.values()}),
        "total_norm_keys": total_norm_keys,
        "old_municipalities_count": len(old_records),
        "abr_used": abr_path is not None,
    }
    (out_dir / "master-stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description="公的データから第2段階辞書リンカー用の辞書ファイルを組み立てます。"
    )
    ap.add_argument("--ken-all", required=True, help="日本郵便 utf_ken_all.csv のパス")
    ap.add_argument("--abr", default=None, help="デジタル庁 ABR 町字マスター (Parquet/CSV, 任意)")
    ap.add_argument("--out", required=True, help="辞書 JSON の出力先ディレクトリ")
    args = ap.parse_args()

    stats = build_dictionary(
        ken_all_path=Path(args.ken_all),
        out_dir=Path(args.out),
        abr_path=Path(args.abr) if args.abr else None,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
