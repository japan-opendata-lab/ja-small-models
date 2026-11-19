#!/usr/bin/env python3
"""地名・住所文字列の正規化および表記ゆれ吸収ユーティリティ。

第2段階辞書リンカー（linker.py）、マスター構築（master_builder.py）、
および合成評価チェックで共通利用する。
"""
import re
import unicodedata
from typing import List, Optional, Set

PREFECTURES: List[str] = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県",
    "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県",
    "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県",
    "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県",
]

PREF_ALIASES: dict[str, str] = {}
for _p in PREFECTURES:
    PREF_ALIASES[_p] = _p
    if _p != "北海道" and _p.endswith(("都", "府", "県")):
        PREF_ALIASES[_p[:-1]] = _p

# 旧字体・異体字の統一テーブル
VARIANT_MAP = str.maketrans({
    "澤": "沢", "櫻": "桜", "廣": "広", "濱": "浜", "邊": "辺", "邉": "辺",
    "齋": "斉", "齊": "斉", "檜": "桧", "龍": "竜", "彌": "弥", "鷗": "鴎",
    "髙": "高", "﨑": "崎", "條": "条", "國": "国", "學": "学", "會": "会",
    "縣": "県", "區": "区", "惠": "恵", "萬": "万", "與": "与", "榮": "栄",
    "關": "関", "嶋": "島", "眞": "真", "壽": "寿", "實": "実", "體": "体",
    "亞": "亜", "傳": "伝", "佛": "仏", "餘": "余", "藏": "蔵", "鹽": "塩",
    "疊": "畳", "圓": "円", "寶": "宝", "舊": "旧", "藝": "芸", "證": "証",
    "鐵": "鉄", "靈": "霊", "黑": "黒", "瀧": "滝", "淵": "渕", "澁": "渋",
    "槇": "槙", "堯": "尭", "德": "徳", "冨": "富", "靜": "静", "驛": "駅",
    "峽": "峡", "狹": "狭", "劍": "剣", "險": "険", "灣": "湾", "廳": "庁",
    "總": "総", "綠": "緑", "絲": "糸", "經": "経", "繪": "絵", "續": "続",
    "豐": "豊", "瀨": "瀬", "應": "応", "壓": "圧", "懷": "懐", "氣": "気",
    "處": "処", "歸": "帰", "樂": "楽", "齒": "歯", "齡": "齢", "龜": "亀",
    "莊": "荘", "稻": "稲", "穗": "穂", "櫻": "桜", "梅": "梅", "柳": "柳",
    "ヶ": "ケ", "ヵ": "ケ", "っ": "ツ", "ッ": "ツ", "之": "ノ", "乃": "ノ",
})

_STRIP_RE = re.compile(r"[\s\u3000・･,，、。．.\-－―ー_]+")
_CHOME_TAIL_RE = re.compile(r"(?:[0-9一二三四五六七八九十百]+丁目?)$")
_KANJI_NUM_MAP = {
    "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9",
    "十": "10", "十一": "11", "十二": "12", "十三": "13", "十四": "14",
    "十五": "15", "十六": "16", "十七": "17", "十八": "18", "十九": "19",
    "二十": "20",
}
_JO_KANJI_RE = re.compile(r"(二十|十[一二三四五六七八九]?|[一二三四五六七八九])(?=[条丁町])")
_JO_ARABIC_RE = re.compile(r"([0-9]{1,2})(?=[条丁町])")


def normalize_geo(s: str) -> str:
    """NFKC正規化・空白記号除去・異体字統一を行う基本正規化。"""
    if not s:
        return ""
    x = unicodedata.normalize("NFKC", s)
    x = _STRIP_RE.sub("", x)
    x = x.translate(VARIANT_MAP)
    return x


def expand_geo_keys(s: str) -> Set[str]:
    """基本正規化に加え、助詞的カナ揺れ（が/ケ、の/ノ、つ/ツ）と条丁漢数字揺れを展開した照合キー集合を返す。"""
    base = normalize_geo(s)
    if not base:
        return set()
    keys: Set[str] = {base}
    # 「が/ガ」->「ケ」、「の」->「ノ」、「つ」->「ツ」の緩やかな正規化キーを追加
    soft = (
        base.replace("が", "ケ")
        .replace("ガ", "ケ")
        .replace("の", "ノ")
        .replace("つ", "ツ")
    )
    keys.add(soft)
    # 「北一条西」↔「北1条西」等の漢数字揺れを吸収
    for k in list(keys):
        k_num = _JO_KANJI_RE.sub(lambda m: _KANJI_NUM_MAP.get(m.group(1), m.group(1)), k)
        keys.add(k_num)
    return keys


def to_katakana(s: str) -> str:
    """ひらがなをカタカナへ変換し、空白記号除去したカナ正規形を返す。"""
    if not s:
        return ""
    x = unicodedata.normalize("NFKC", s)
    x = _STRIP_RE.sub("", x)
    out = []
    for ch in x:
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:
            out.append(chr(code + 0x60))
        else:
            out.append(ch)
    res = "".join(out)
    # カナ内の小書き揺れ統一（濁音「ガ」は識別のため維持する）
    res = res.replace("ヶ", "ケ").replace("ヵ", "ケ").replace("ッ", "ツ")
    return res


def is_all_kana(s: str) -> bool:
    """空白等を除いた文字列がすべてひらがな・カタカナ（＋長音）か判定する。"""
    n = _STRIP_RE.sub("", unicodedata.normalize("NFKC", s or ""))
    if not n:
        return False
    return all(
        ("\u3041" <= c <= "\u309f") or ("\u30a1" <= c <= "\u30ff") or c == "ー"
        for c in n
    )


def normalize_pref(s: str) -> Optional[str]:
    """都道府県表記（省略形・異体字含む）を正規の47都道府県名へ解決する。"""
    if not s:
        return None
    n = normalize_geo(s)
    return PREF_ALIASES.get(n)


def extract_town_variants(town: str) -> Set[str]:
    """町域表記から照合用の複数正規化キー（大字/字除去・末尾丁目除去・カナ揺れ含む）を生成する。"""
    keys: Set[str] = set()
    base = normalize_geo(town)
    if not base:
        return keys
    cands = {base}
    # 先頭の「大字」「字」除去
    for pfx in ("大字", "字"):
        if base.startswith(pfx) and len(base) > len(pfx):
            cands.add(base[len(pfx):])
    # 末尾の「○丁目」除去版（例: "神南1丁目" -> "神南"）
    for c in list(cands):
        stripped = _CHOME_TAIL_RE.sub("", c)
        if stripped and len(stripped) >= 2:
            cands.add(stripped)
    for c in cands:
        keys.update(expand_geo_keys(c))
    return keys
