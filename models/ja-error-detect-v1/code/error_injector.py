#!/usr/bin/env python3
"""第6号 ja-error-detect-v1 用 プログラム的誤り注入器。

HuggingFaceFW/fineweb-2（ODC-By 1.0）から抽出した自然な日本語文に対し、
fugashi + unidic-lite（MIT/BSD）の形態素解析を用いて以下の 6 種別の誤りを注入する。
  1. particle     : 助詞の置換
  2. homophone    : 同音異義語の誤変換（日常語×日常語・同一読み・異意味）
  3. okurigana    : 送り仮名の誤り（文化庁許容表記・正当語衝突を厳格除外）
  4. typo_similar : 形の似た漢字への置換（両ベース語彙に実在する漢字のみ）
  5. omission     : 脱字（口語短縮や曖昧な 1 文字残余を除外した明確な文字欠落）
  6. insertion    : 衍字（文字重複入力・非正当重複助詞・語中不要仮名挿入）

方言・文体差・口語表現・許容表記・近義語表記ゆれは誤りとして扱わない。
"""
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import fugashi

ERROR_TYPES: Tuple[str, ...] = (
    "particle",
    "homophone",
    "okurigana",
    "typo_similar",
    "omission",
    "insertion",
)

_RE_KANJI_ONLY = re.compile(r"^[\u4e00-\u9fff]+$")
_RE_KANJI1_HIRA = re.compile(r"^([\u4e00-\u9fff])([\u3041-\u3096]+)$")
_RE_KATA = re.compile(r"^[ァ-ヴー]{3,8}$")
_RE_KANJI3_5 = re.compile(r"^[\u4e00-\u9fff]{3,5}$")
_SMALL_OR_LONG_KATA = set("ーッャュョァィゥェォ")


def _kata_to_hira(s: str) -> str:
    """カタカナ文字列をひらがな文字列に変換する。"""
    out = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        elif ch == "ー":
            out.append("ー")
        else:
            out.append(ch)
    return "".join(out)


def _is_hiragana(ch: str) -> bool:
    return "\u3041" <= ch <= "\u3096"


def _is_katakana(ch: str) -> bool:
    return "\u30A1" <= ch <= "\u30F6"


def _is_kanji(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


# =====================================================================
# 1. 助詞の置換設定（非文脈的に明らかに不自然な置換のみを許可）
# =====================================================================
_PARTICLE_TARGETS: Dict[str, List[str]] = {
    "を": ["へ", "の", "に", "で"],
    "の": ["を", "へ", "が"],
    "に": ["を", "の"],
    "で": ["を", "へ", "の"],
    "へ": ["を", "で", "の"],
    "が": ["へ", "に", "を"],
    "は": ["を", "へ"],
}
_MOTION_VERBS = {
    "歩く", "走る", "通る", "渡る", "飛ぶ", "進む", "回る", "泳ぐ",
    "登る", "降りる", "越える", "過ぎる", "出る", "去る", "離れる",
}
_STATIVE_GA_PREDICATES = {
    "ある", "いる", "できる", "分かる", "わかる", "好き", "嫌い",
    "必要", "多い", "少ない", "高い", "低い", "良い", "いい", "なる",
}

# =====================================================================
# 2. 同音異義語グループ（日常語×日常語・異意味・近義表記ゆれ除外）
# =====================================================================
_HOMOPHONE_NOUN_RAW: List[List[str]] = [
    ["以外", "意外"], ["以上", "異常"], ["以降", "移行", "意向"],
    ["意義", "異議"], ["移動", "異動"], ["依頼", "以来"],
    ["遺跡", "移籍"], ["回答", "解凍"], ["改正", "快晴"],
    ["科学", "化学"], ["家庭", "過程", "仮定"], ["確率", "確立"],
    ["関心", "感心"], ["期間", "機関", "帰還"], ["企業", "起業"],
    ["気象", "起床"], ["期待", "機体", "気体"], ["機能", "昨日"],
    ["機会", "機械"], ["技術", "記述"], ["強調", "協調"],
    ["空間", "区間"], ["現象", "減少"], ["原則", "減速"],
    ["工業", "鉱業"], ["効果", "高価", "硬貨", "降下"],
    ["後期", "好機", "好奇"], ["構成", "厚生", "校正", "更生"],
    ["行動", "講堂", "公道"], ["効率", "高率"], ["個人", "故人"],
    ["故障", "胡椒", "呼称"], ["最新", "細心"], ["作成", "作戦"],
    ["参加", "酸化"], ["資格", "視覚", "四角", "死角"], ["事故", "自己"],
    ["市場", "史上", "紙上"], ["指示", "支持"], ["姿勢", "市政"],
    ["施設", "私設"], ["自然", "事前"], ["実際", "実在"],
    ["自動", "児童"], ["指導", "始動"], ["修正", "習性"],
    ["終了", "修了"], ["週間", "習慣"], ["収集", "収拾"],
    ["主張", "首長"], ["状況", "上京"], ["状態", "上体"],
    ["条件", "上限"], ["商品", "賞品"], ["承認", "証人", "商人"],
    ["証明", "照明"], ["資料", "飼料", "試料"],
    ["進行", "振興", "信仰"], ["審査", "新車"],
    ["慎重", "身長", "新調"], ["推移", "水位"], ["推進", "水深"],
    ["性格", "正確"], ["成功", "精巧"], ["政策", "制作"],
    ["生産", "精算", "清算"], ["制度", "精度"],
    ["生命", "声明", "姓名"], ["世界", "政界"], ["操作", "捜査"],
    ["想像", "創造"], ["対象", "対称", "対照"], ["体制", "耐性"],
    ["大量", "大漁"], ["通常", "通商"], ["展開", "転回"],
    ["展示", "点字"], ["当時", "冬至"], ["投資", "透視", "闘志"],
    ["背景", "拝啓"], ["配信", "背信"], ["比較", "皮革"],
    ["非常", "非情"], ["負担", "普段"], ["普通", "不通"],
    ["変更", "偏向"], ["変換", "返還"], ["方向", "芳香"],
    ["優先", "有線"], ["有料", "優良"], ["用意", "容易"],
    ["利用", "理容"], ["時期", "磁気", "次期"], ["時間", "字間"],
    ["試験", "私見"], ["発見", "発券"], ["検討", "健闘", "見当"],
    ["文化", "分科"], ["医療", "衣料"], ["成果", "製菓", "青果"],
    ["課題", "過大"], ["完成", "歓声", "感性"],
    ["選考", "先行", "専攻"], ["合格", "互角"],
    ["広報", "後方", "候補"], ["監視", "幹事", "漢字"],
    ["障害", "生涯", "渉外"], ["復旧", "復習"], ["対策", "大作"],
    ["紹介", "照会"], ["更新", "行進", "後進"],
    ["選択", "洗濯"], ["氏名", "使命"], ["維持", "意地"],
    ["向上", "工場", "口上"], ["強化", "教科", "許可"],
    ["共有", "固有"], ["禁止", "近視"], ["構造", "構想"],
    ["実践", "実戦"], ["観察", "監察"], ["気候", "機構", "起工"],
    ["天気", "転機", "電気"], ["放送", "包装", "法曹"],
    ["橋", "箸", "端"], ["雨", "飴"], ["紙", "髪", "神"],
    ["川", "皮"], ["花", "鼻"], ["石", "医師", "意思"],
    ["服", "福"], ["島", "縞"], ["塩", "潮"], ["酒", "鮭"],
    ["帽子", "防止"], ["先生", "宣誓"], ["生徒", "政党"],
    ["面", "麺"], ["線", "千"], ["円", "縁"], ["薬", "役"],
    ["現在", "顕在"], ["将来", "招来"], ["感情", "勘定", "環状"],
    ["価格", "化学", "科学"], ["記録", "気力"], ["予定", "余丁"],
]

# 動詞・形容詞の同音異義（orthBase と読みプレフィックスで曖昧性除去）
# 近義・同訓異字ニュアンスの近いペア（超える/越える、見る/観る、聞く/聴く、計る/測る、使う/遣う、早い/速い）は除外
_HOMOPHONE_STEM_GROUPS: List[Tuple[Tuple[str, ...], str, str]] = [
    (("会う", "合う"), "ア", "動詞"),
    (("買う", "飼う"), "カ", "動詞"),
    (("聞く", "効く"), "キ", "動詞"),
    (("取る", "撮る", "捕る"), "ト", "動詞"),
    (("住む", "済む"), "ス", "動詞"),
    (("降る", "振る"), "フ", "動詞"),
    (("閉める", "締める", "占める"), "シメ", "動詞"),
    (("帰る", "返る"), "カエ", "動詞"),
    (("直す", "治す"), "ナオ", "動詞"),
    (("直る", "治る"), "ナオ", "動詞"),
    (("進める", "勧める"), "ススメ", "動詞"),
    (("務める", "勤める", "努める"), "ツトメ", "動詞"),
    (("延ばす", "伸ばす"), "ノバ", "動詞"),
    (("延びる", "伸びる"), "ノビ", "動詞"),
    (("除く", "覗く"), "ノゾ", "動詞"),
    (("立つ", "建つ", "断つ"), "タ", "動詞"),
    (("立てる", "建てる"), "タテ", "動詞"),
    (("付く", "着く"), "ツ", "動詞"),
    (("泣く", "鳴く"), "ナ", "動詞"),
    (("乗る", "載る"), "ノ", "動詞"),
    (("乗せる", "載せる"), "ノセ", "動詞"),
    (("折る", "織る"), "オ", "動詞"),
    (("解く", "溶く"), "ト", "動詞"),
    (("解ける", "溶ける"), "トケ", "動詞"),
    (("冷める", "覚める"), "サメ", "動詞"),
    (("冷ます", "覚ます"), "サマ", "動詞"),
    (("移る", "映る", "写る"), "ウツ", "動詞"),
    (("移す", "映す", "写す"), "ウツ", "動詞"),
    (("送る", "贈る"), "オク", "動詞"),
    (("押す", "推す"), "オ", "動詞"),
    (("答える", "応える"), "コタエ", "動詞"),
    (("見る", "診る"), "ミ", "動詞"),
    (("刺す", "指す", "差す"), "サ", "動詞"),
    (("積む", "詰む"), "ツ", "動詞"),
    (("留める", "止める", "泊める"), "トメ", "動詞"),
    (("留まる", "止まる", "泊まる"), "トマ", "動詞"),
    (("張る", "貼る"), "ハ", "動詞"),
    (("引く", "弾く"), "ヒ", "動詞"),
    (("吹く", "拭く"), "フ", "動詞"),
    (("掘る", "彫る"), "ホ", "動詞"),
    (("生む", "産む"), "ウ", "動詞"),
    (("追う", "負う"), "オ", "動詞"),
    (("攻める", "責める"), "セメ", "動詞"),
    (("飛ぶ", "跳ぶ"), "ト", "動詞"),
    (("話す", "放す", "離す"), "ハナ", "動詞"),
    (("鳴る", "成る"), "ナ", "動詞"),
    (("刈る", "狩る"), "カ", "動詞"),
    (("熱い", "暑い", "厚い"), "アツ", "形容詞"),
    (("優しい", "易しい"), "ヤサ", "形容詞"),
]

# =====================================================================
# 3. 送り仮名誤差用 除外リスト・送り不足表
# =====================================================================
_OKURIGANA_ALLOW_EXACT = {
    "行う", "行なう", "表す", "表わす", "現れる", "現われる", "現す", "現わす",
    "終わる", "終る", "終える", "変わる", "変る", "変える", "浮かぶ", "浮ぶ",
    "生まれる", "生れる", "生む", "聞こえる", "聞える", "断る", "断わる",
    "秀でる", "落ちる", "過ぎる", "起きる", "降りる", "伸びる", "閉じる", "開く",
    "取り組む", "申し込む", "売り上げる", "受け付ける", "問い合わせる",
    "組み合わせる", "打ち合わせる", "引き渡す", "取り扱う", "書き込む",
    "折り返す", "繰り返す", "見直す", "取り出す", "持ち込む", "差し出す",
    "行く", "言う", "来る", "する", "見る", "出る", "居る", "有る", "無い",
    "良い", "いい", "多い", "少ない", "大きい", "小さい",
}

_OKURI_OMIT_ADJECTIVES: Dict[str, str] = {
    "新しい": "新い",
    "楽しい": "楽い",
    "美しい": "美い",
    "嬉しい": "嬉い",
    "正しい": "正い",
    "優しい": "優い",
    "詳しい": "詳い",
    "厳しい": "厳い",
    "激しい": "激い",
    "珍しい": "珍い",
    "悲しい": "悲い",
    "忙しい": "忙い",
    "明るい": "明い",
    "暖かい": "暖い",
    "柔らかい": "柔い",
    "冷たい": "冷い",
    "素晴らしい": "素晴い",
    "恥ずかしい": "恥い",
    "懐かしい": "懐い",
    "寂しい": "寂い",
}

_OKURI_OMIT_ICHIDAN_VERBS: Dict[str, str] = {
    "考える": "考る",
    "調べる": "調る",
    "集める": "集る",
    "教える": "教る",
    "覚える": "覚る",
    "伝える": "伝る",
    "届ける": "届る",
    "食べる": "食る",
    "並べる": "並る",
    "続ける": "続る",
    "助ける": "助る",
    "求める": "求る",
    "認める": "認る",
    "進める": "進る",
    "高める": "高る",
    "深める": "深る",
    "広める": "広る",
    "含める": "含る",
    "固める": "固る",
    "答える": "答る",
    "迎える": "迎る",
    "支える": "支る",
    "加える": "加る",
    "越える": "越る",
    "比べる": "比る",
    "受ける": "受る",
    "逃げる": "逃る",
    "投げる": "投る",
    "挙げる": "挙る",
    "見つける": "見つる",
    "始める": "始る",
}

# =====================================================================
# 4. 形の似た漢字ペア（両ベース語彙に実在する漢字のみで構成）
# =====================================================================
_TYPO_SIMILAR_PAIRS: List[Tuple[str, str]] = [
    ("問", "間"), ("待", "持"), ("持", "特"), ("待", "特"),
    ("微", "徴"), ("撤", "徹"), ("値", "植"), ("延", "廷"),
    ("建", "健"), ("徒", "従"), ("綱", "網"), ("壁", "璧"),
    ("獲", "穫"), ("暦", "歴"), ("未", "末"), ("土", "士"),
    ("大", "太"), ("王", "玉"), ("白", "自"), ("自", "百"),
    ("白", "百"), ("田", "由"), ("由", "申"), ("開", "閉"),
    ("開", "聞"), ("開", "関"), ("閉", "関"), ("聞", "関"),
    ("晴", "精"), ("精", "清"), ("清", "請"), ("請", "情"),
    ("晴", "情"), ("精", "情"), ("材", "村"), ("休", "体"),
    ("低", "底"), ("儀", "議"), ("議", "義"), ("儀", "義"),
    ("減", "滅"), ("決", "快"), ("測", "側"), ("側", "則"),
    ("測", "則"), ("温", "湿"), ("満", "漏"), ("現", "理"),
    ("発", "登"), ("相", "想"), ("真", "直"), ("科", "料"),
    ("程", "種"), ("積", "績"), ("競", "鏡"), ("管", "館"),
    ("系", "係"), ("約", "納"), ("紀", "記"), ("終", "修"),
    ("経", "軽"), ("結", "給"), ("統", "続"), ("職", "織"),
    ("能", "態"), ("著", "着"), ("葉", "薬"), ("薄", "簿"),
    ("規", "視"), ("覚", "学"), ("観", "歓"), ("設", "説"),
    ("試", "詩"), ("調", "週"), ("論", "輪"), ("販", "板"),
    ("貴", "遺"), ("費", "賞"), ("資", "姿"), ("貸", "貨"),
    ("買", "売"), ("転", "伝"), ("通", "道"), ("速", "連"),
    ("達", "違"), ("適", "過"), ("選", "遷"), ("部", "都"),
    ("鉄", "銀"), ("銀", "限"), ("録", "緑"), ("険", "検"),
    ("階", "際"), ("障", "章"), ("離", "難"), ("静", "浄"),
    ("頂", "項"), ("順", "須"), ("領", "顔"), ("顔", "願"),
    ("駅", "験"), ("駐", "駆"), ("商", "適"), ("変", "恋"),
    ("改", "政"), ("放", "故"), ("敏", "繁"), ("教", "数"),
    ("新", "親"), ("景", "影"), ("暖", "緩"), ("暴", "爆"),
    ("最", "撮"), ("案", "安"), ("極", "様"), ("構", "講"),
    ("標", "票"), ("模", "漠"), ("残", "浅"), ("母", "毎"),
    ("準", "准"), ("熟", "熱"), ("燃", "然"), ("版", "板"),
    ("牧", "枚"), ("物", "特"), ("留", "貿"), ("症", "証"),
    ("痛", "通"), ("監", "鑑"), ("眼", "根"), ("確", "雑"),
    ("祈", "折"), ("福", "幅"), ("移", "秘"), ("穏", "隠"),
    ("第", "弟"), ("筆", "等"), ("筋", "節"), ("簡", "間"),
    ("紛", "粉"), ("純", "鈍"), ("細", "組"), ("綿", "絹"),
    ("練", "錬"), ("繕", "善"), ("罪", "罰"), ("群", "郡"),
    ("老", "考"), ("肌", "机"), ("胸", "脳"), ("腹", "復"),
    ("臓", "蔵"), ("裁", "載"), ("補", "捕"), ("製", "装"),
    ("複", "腹"), ("訪", "防"), ("詳", "洋"), ("誤", "語"),
    ("読", "続"), ("謹", "勤"), ("警", "驚"), ("財", "材"),
    ("貧", "賃"), ("貫", "慣"), ("距", "拒"), ("軌", "軟"),
    ("較", "校"), ("追", "迫"), ("送", "逆"), ("逐", "遂"),
    ("遵", "導"), ("避", "壁"), ("還", "遠"), ("酔", "粋"),
    ("醸", "譲"), ("銅", "鋼"), ("錯", "借"), ("鎮", "鎖"),
    ("閑", "間"), ("閣", "格"), ("陣", "陳"), ("除", "徐"),
    ("陰", "隠"), ("陵", "陸"), ("隊", "墜"), ("雇", "顧"),
    ("震", "振"), ("霧", "務"), ("預", "頂"), ("類", "頻"),
    ("飯", "販"), ("飲", "飯"), ("飼", "詞"), ("鮮", "詳"),
    ("鳴", "鴨"), ("黒", "墨"),
]

# =====================================================================
# 5. 脱字用定型機能表現マップ（実在語衝突・口語縮約を除外）
# =====================================================================
_OMISSION_FUNC_MAP: Dict[str, str] = {
    "について": "につて",
    "によって": "にって",
    "において": "におて",
    "に対して": "に対て",
    "にとって": "にとて",
    "に関して": "に関て",
    "とともに": "ととも",
    "および": "およ",
    "そのため": "そのた",
    "このため": "このた",
    "あるいは": "あるは",
    "もしくは": "もしく",
    "ならびに": "ならび",
    "たとえば": "たとば",
    "すなわち": "すなち",
    "したがって": "したって",
    "なぜなら": "なぜら",
    "ちなみに": "ちなに",
    "おそらく": "おそく",
    "かならず": "かなず",
    "ようやく": "ようく",
    "とにかく": "とにく",
    "しばらく": "しらく",
    "ほとんど": "ほとど",
    "そして": "そし",
    "けれど": "けれ",
    "だから": "だか",
    "くらい": "くら",
    "ぐらい": "ぐら",
}

# =====================================================================
# 6. 衍字用 定数
# =====================================================================
_INVALID_PARTICLE_AFTER: Dict[str, List[str]] = {
    "は": ["を", "が", "へ", "に", "で"],
    "が": ["を", "へ", "に", "で"],
    "を": ["の", "が", "へ", "に", "で"],
    "に": ["を", "が", "で", "へ"],
    "で": ["を", "が", "に", "へ"],
    "へ": ["を", "が", "に", "で"],
}
_FILLER_HIRA = list("あいうえおかきくさしたてなにのはまらりるれを")
_DUPLICATE_EXCLUDED_CHARS = set("っんー、。？！「」『』（）・　 ")
_KNOWN_DOUBLED_WORDS = {
    "ここ", "ただ", "まま", "やや", "もも", "つつ", "すす", "パパ", "ママ", "ココ",
    "はは", "ちち", "みみ", "ほほ", "のの", "なな", "きき", "ささ", "しし", "たた",
}


@dataclass(frozen=True)
class Candidate:
    """1 件の誤り注入候補。元文中の [start, end) を replacement に置換する。"""

    start: int
    end: int
    replacement: str
    err_type: str
    original: str
    corrupted: str
    subkind: str = ""


class ErrorInjector:
    """unidic-lite を用いたプログラム的 6 種誤り注入器。"""

    def __init__(self) -> None:
        self.tagger = fugashi.Tagger()
        self._build_tables()

    def _build_tables(self) -> None:
        # 1. 名詞同音異義マップ（unidic 上の同一読みごとにグループ化）
        self.noun_homophones: Dict[str, Tuple[str, List[str]]] = {}
        for grp in _HOMOPHONE_NOUN_RAW:
            by_kana: Dict[str, List[str]] = {}
            for w in grp:
                ms = list(self.tagger(w))
                if len(ms) != 1 or ms[0].is_unk:
                    continue
                if ms[0].feature.pos1 != "名詞":
                    continue
                kana = ms[0].feature.kana or ""
                if kana:
                    by_kana.setdefault(kana, []).append(w)
            for kana, words in by_kana.items():
                uniq = list(dict.fromkeys(words))
                if len(uniq) >= 2:
                    for w in uniq:
                        others = [o for o in uniq if o != w]
                        if w not in self.noun_homophones:
                            self.noun_homophones[w] = (kana, others)

        # 2. 動詞・形容詞同音異義マップ
        self.stem_homophones: Dict[Tuple[str, str], Tuple[str, List[Tuple[str, str]]]] = {}
        for bases, kprefix, pos1 in _HOMOPHONE_STEM_GROUPS:
            for b in bases:
                others = [(o[0], o) for o in bases if o != b]
                self.stem_homophones[(b, pos1)] = (kprefix, others)

        # 3. 形の似た漢字マップ
        self.typo_map: Dict[str, List[str]] = {}
        for a, b in _TYPO_SIMILAR_PAIRS:
            self.typo_map.setdefault(a, [])
            if b not in self.typo_map[a]:
                self.typo_map[a].append(b)
            self.typo_map.setdefault(b, [])
            if a not in self.typo_map[b]:
                self.typo_map[b].append(a)

    def _is_single_known_word(self, word: str) -> bool:
        """単語が fugashi/unidic-lite で単一の既知語として解析されるか判定する。"""
        ms = list(self.tagger(word))
        return len(ms) == 1 and not ms[0].is_unk

    def _tokenize_with_spans(self, text: str) -> List[Tuple[object, int, int]]:
        """テキストを形態素解析し、(node, start, end) のリストを返す。"""
        nodes = list(self.tagger(text))
        res: List[Tuple[object, int, int]] = []
        pos = 0
        for w in nodes:
            surf = w.surface
            idx = text.find(surf, pos)
            if idx < 0:
                continue
            res.append((w, idx, idx + len(surf)))
            pos = idx + len(surf)
        return res

    def _collect_particle(
        self, text: str, tokens: List[Tuple[object, int, int]]
    ) -> List[Candidate]:
        cands: List[Candidate] = []
        n = len(tokens)
        for i, (w, st, ed) in enumerate(tokens):
            surf = w.surface
            if surf not in _PARTICLE_TARGETS:
                continue
            if w.feature.pos1 != "助詞" or w.feature.pos2 not in ("格助詞", "係助詞"):
                continue
            if i > 0 and tokens[i - 1][0].feature.pos1 == "助詞":
                continue
            if i + 1 < n and tokens[i + 1][0].feature.pos1 == "助詞":
                continue
            if surf == "の":
                if not (
                    i > 0
                    and i + 1 < n
                    and tokens[i - 1][0].feature.pos1 == "名詞"
                    and tokens[i + 1][0].feature.pos1 == "名詞"
                ):
                    continue
            if surf == "は":
                if i == 0 or tokens[i - 1][0].feature.pos1 not in ("名詞", "代名詞"):
                    continue
                if i + 1 >= n or tokens[i + 1][0].feature.pos1 == "補助記号":
                    continue
            if i + 1 >= n or tokens[i + 1][0].feature.pos1 == "補助記号":
                continue

            nxt_w = tokens[i + 1][0]
            nxt_base = nxt_w.feature.orthBase or nxt_w.surface
            opts = list(_PARTICLE_TARGETS[surf])
            if surf == "を" and nxt_base in _MOTION_VERBS:
                opts = [x for x in opts if x not in ("に", "で")]
            elif surf == "が" and nxt_base in _STATIVE_GA_PREDICATES:
                opts = [x for x in opts if x not in ("を", "に")]
            elif surf == "に":
                # 連体修飾節（「環境に配慮した車両」->「環境の配慮した車両」）での「の」を避ける
                if (
                    i + 2 < n
                    and nxt_w.feature.pos1 == "動詞"
                    and (
                        tokens[i + 2][0].feature.pos1 == "名詞"
                        or (
                            i + 3 < n
                            and tokens[i + 2][0].feature.pos1 == "助動詞"
                            and tokens[i + 3][0].feature.pos1 == "名詞"
                        )
                    )
                ):
                    opts = [x for x in opts if x != "の"]

            for rep in opts:
                cands.append(Candidate(st, ed, rep, "particle", surf, rep, surf))
        return cands

    def _collect_homophone(
        self, text: str, tokens: List[Tuple[object, int, int]]
    ) -> List[Candidate]:
        cands: List[Candidate] = []
        for w, st, ed in tokens:
            surf = w.surface
            pos1 = w.feature.pos1
            kana = w.feature.kana or ""

            if pos1 == "名詞" and surf in self.noun_homophones:
                expected_kana, others = self.noun_homophones[surf]
                if kana != expected_kana:
                    continue
                if st > 0 and _is_kanji(text[st - 1]):
                    continue
                if ed < len(text) and _is_kanji(text[ed]):
                    continue
                for rep in others:
                    cands.append(Candidate(st, ed, rep, "homophone", surf, rep, "noun"))
                continue

            orth_base = w.feature.orthBase or ""
            if (
                pos1 in ("動詞", "形容詞")
                and (orth_base, pos1) in self.stem_homophones
                and len(surf) >= 1
                and surf[0] == orth_base[0]
                and _is_kanji(surf[0])
            ):
                kprefix, targets = self.stem_homophones[(orth_base, pos1)]
                if not kana.startswith(kprefix):
                    continue
                for tgt_kanji, tgt_base in targets:
                    rep_word = tgt_kanji + surf[1:]
                    ms = list(self.tagger(rep_word))
                    if (
                        len(ms) == 1
                        and not ms[0].is_unk
                        and ms[0].feature.pos1 == pos1
                        and ms[0].feature.cType == w.feature.cType
                        and ms[0].feature.orthBase == tgt_base
                    ):
                        cands.append(
                            Candidate(st, st + 1, tgt_kanji, "homophone", surf[0], tgt_kanji, "stem")
                        )
        return cands

    def _collect_okurigana(
        self, text: str, tokens: List[Tuple[object, int, int]]
    ) -> List[Candidate]:
        cands: List[Candidate] = []
        for w, st, ed in tokens:
            surf = w.surface
            pos1 = w.feature.pos1
            orth_base = w.feature.orthBase or ""
            if pos1 not in ("動詞", "形容詞"):
                continue
            if orth_base in _OKURIGANA_ALLOW_EXACT:
                continue

            # 方式 B: 送り不足
            if orth_base in _OKURI_OMIT_ADJECTIVES:
                if len(surf) >= 3 and _is_kanji(surf[0]) and _is_hiragana(surf[1]) and _is_hiragana(surf[2]):
                    rep = surf[0] + surf[2:]
                    if not self._is_single_known_word(rep):
                        cands.append(Candidate(st, ed, rep, "okurigana", surf, rep, "omit"))
                        continue
            elif orth_base in _OKURI_OMIT_ICHIDAN_VERBS:
                kanji_prefix_len = len(orth_base) - 2
                if (
                    kanji_prefix_len >= 1
                    and len(surf) >= kanji_prefix_len + 2
                    and surf.startswith(orth_base[: kanji_prefix_len + 1])
                ):
                    rep = surf[:kanji_prefix_len] + surf[kanji_prefix_len + 1 :]
                    if rep != surf and not self._is_single_known_word(rep):
                        cands.append(Candidate(st, ed, rep, "okurigana", surf, rep, "omit"))
                        continue

            # 方式 A: 送りすぎ
            if w.feature.goshu != "和":
                continue
            m = _RE_KANJI1_HIRA.match(orth_base)
            if not m:
                continue
            kanji_ch, okuri_base = m.group(1), m.group(2)
            if len(surf) < 2 or surf[0] != kanji_ch or not _is_hiragana(surf[1]):
                continue
            kana_base = _kata_to_hira(w.feature.kanaBase or "")
            if not kana_base.endswith(okuri_base) or len(kana_base) <= len(okuri_base):
                continue
            stem_hira = kana_base[: len(kana_base) - len(okuri_base)]
            if len(stem_hira) < 2:
                continue
            ins_ch = stem_hira[-1]
            if ins_ch in set("っんゃゅょぁぃぅぇぉー"):
                continue
            if ins_ch == surf[1]:
                continue
            rep = surf[0] + ins_ch + surf[1:]
            if not self._is_single_known_word(rep):
                cands.append(Candidate(st, ed, rep, "okurigana", surf, rep, "excess"))
        return cands

    def _collect_typo_similar(
        self, text: str, tokens: List[Tuple[object, int, int]]
    ) -> List[Candidate]:
        cands: List[Candidate] = []
        for w, st, ed in tokens:
            surf = w.surface
            if not (2 <= len(surf) <= 4 and _RE_KANJI_ONLY.match(surf)):
                continue
            if len(surf) == 2 and surf[0] in set("一二三四五六七八九十上下前後大小新旧各全本該当"):
                continue
            for idx_in_word, ch in enumerate(surf):
                if ch not in self.typo_map:
                    continue
                for rep_ch in self.typo_map[ch]:
                    rep_word = surf[:idx_in_word] + rep_ch + surf[idx_in_word + 1 :]
                    if self._is_single_known_word(rep_word):
                        continue
                    cpos = st + idx_in_word
                    cands.append(
                        Candidate(cpos, cpos + 1, rep_ch, "typo_similar", ch, rep_ch, "typo")
                    )
        return cands

    def _collect_omission(
        self, text: str, tokens: List[Tuple[object, int, int]]
    ) -> List[Candidate]:
        cands: List[Candidate] = []
        n = len(tokens)
        for i, (w, st, ed) in enumerate(tokens):
            surf = w.surface
            pos1 = w.feature.pos1

            # A: 3〜8文字カタカナ語の内側 1 文字削除（長音・促音・拗音前後は保護）
            if _RE_KATA.match(surf):
                for j in range(1, len(surf) - 1):
                    if surf[j] in _SMALL_OR_LONG_KATA:
                        continue
                    if surf[j + 1] in _SMALL_OR_LONG_KATA:
                        continue
                    if surf[j - 1] == surf[j + 1]:
                        continue
                    rep = surf[:j] + surf[j + 1 :]
                    if not self._is_single_known_word(rep):
                        cands.append(Candidate(st, ed, rep, "omission", surf, rep, "kata"))
                continue

            # B: 単一 3〜5 文字漢字名詞の中間 1 文字削除
            if pos1 == "名詞" and _RE_KANJI3_5.match(surf):
                if (st > 0 and _is_kanji(text[st - 1])) or (
                    ed < len(text) and _is_kanji(text[ed])
                ):
                    continue
                for j in range(1, len(surf) - 1):
                    rep = surf[:j] + surf[j + 1 :]
                    if not self._is_single_known_word(rep):
                        cands.append(Candidate(st, ed, rep, "omission", surf, rep, "kanji3"))
                continue

            # C1: 定型機能表現の 1 文字欠落
            if surf in _OMISSION_FUNC_MAP:
                rep = _OMISSION_FUNC_MAP[surf]
                cands.append(Candidate(st, ed, rep, "omission", surf, rep, "func"))
                continue

            # C2: 動詞連用形 (len >= 2) + 「ます」 -> 「ます」が「す」に欠落
            if (
                surf == "ます"
                and pos1 == "助動詞"
                and i > 0
                and tokens[i - 1][0].feature.pos1 == "動詞"
                and len(tokens[i - 1][0].surface) >= 2
            ):
                cands.append(Candidate(st, ed, "す", "omission", "ます", "す", "masu"))
                continue

            # C3: 格助詞「から」「まで」の欠落（名詞 + から/まで + 名詞/動詞）
            if (
                pos1 == "助詞"
                and surf in ("から", "まで")
                and 0 < i < n - 1
                and tokens[i - 1][0].feature.pos1 == "名詞"
                and tokens[i + 1][0].feature.pos1 in ("名詞", "動詞")
            ):
                cands.append(Candidate(st, ed, surf[0], "omission", surf, surf[0], "kara"))
        return cands

    def _collect_insertion(
        self, text: str, tokens: List[Tuple[object, int, int]]
    ) -> List[Candidate]:
        cands: List[Candidate] = []
        n = len(tokens)
        text_len = len(text)

        # A: 文字の重複入力（キー連打）
        for w, st, ed in tokens:
            surf = w.surface
            pos1 = w.feature.pos1
            if pos1 not in ("助詞", "動詞", "形容詞", "名詞"):
                continue
            # 「はは」等を避けるため単独助詞「は」の重複は除外
            if surf == "は" and pos1 == "助詞":
                continue
            for off, ch in enumerate(surf):
                cpos = st + off
                if cpos >= text_len - 2:
                    continue
                if not (_is_hiragana(ch) or _is_katakana(ch)):
                    continue
                if ch in _DUPLICATE_EXCLUDED_CHARS:
                    continue
                if cpos > 0 and text[cpos - 1] == ch:
                    continue
                if cpos + 1 < text_len and text[cpos + 1] == ch:
                    continue
                rep_word = surf[: off + 1] + ch + surf[off + 1 :]
                if rep_word in _KNOWN_DOUBLED_WORDS or self._is_single_known_word(rep_word):
                    continue
                cands.append(Candidate(cpos + 1, cpos + 1, ch, "insertion", "", ch, "dup"))

        # B1: 単一格助詞・係助詞の直後に非正当な助詞 1 文字を挿入
        for i, (w, st, ed) in enumerate(tokens):
            surf = w.surface
            if surf not in _INVALID_PARTICLE_AFTER:
                continue
            if w.feature.pos1 != "助詞" or w.feature.pos2 not in ("格助詞", "係助詞"):
                continue
            if i + 1 >= n or tokens[i + 1][0].feature.pos1 == "補助記号":
                continue
            if tokens[i + 1][0].feature.pos1 == "助詞":
                continue
            if ed >= text_len - 2:
                continue
            for ins_p in _INVALID_PARTICLE_AFTER[surf]:
                cands.append(Candidate(ed, ed, ins_p, "insertion", "", ins_p, "dup_p"))

        # B2: 漢字熟語またはカタカナ語の内部に不要ひらがな 1 文字を挿入
        for w, st, ed in tokens:
            surf = w.surface
            if (2 <= len(surf) <= 4 and _RE_KANJI_ONLY.match(surf)) or _RE_KATA.match(surf):
                ins_pos = st + 1
                if ins_pos >= text_len - 2:
                    continue
                for hch in _FILLER_HIRA[:10]:
                    cands.append(Candidate(ins_pos, ins_pos, hch, "insertion", "", hch, "filler"))
        return cands

    def collect_all_candidates(self, text: str) -> Dict[str, List[Candidate]]:
        """文中の全 6 種別の誤り注入候補を収集して種別ごとの辞書で返す。"""
        tokens = self._tokenize_with_spans(text)
        return {
            "particle": self._collect_particle(text, tokens),
            "homophone": self._collect_homophone(text, tokens),
            "okurigana": self._collect_okurigana(text, tokens),
            "typo_similar": self._collect_typo_similar(text, tokens),
            "omission": self._collect_omission(text, tokens),
            "insertion": self._collect_insertion(text, tokens),
        }

    def inject_errors(
        self,
        text: str,
        target_types: Sequence[str],
        rng: random.Random,
        min_gap: int = 4,
    ) -> Tuple[str, List[dict]]:
        """指定された種別列の順に可能な誤りを選択・注入し、(破損後文, spans) を返す。"""
        by_type = self.collect_all_candidates(text)
        chosen: List[Candidate] = []
        used_ranges: List[Tuple[int, int]] = []

        for etype in target_types:
            pool = by_type.get(etype, [])
            if not pool:
                continue
            avail = []
            for c in pool:
                conflict = False
                for us, ue in used_ranges:
                    if not (c.end + min_gap <= us or c.start >= ue + min_gap):
                        conflict = True
                        break
                if not conflict:
                    avail.append(c)
            if not avail:
                continue

            # サブパターンがある種別（insertion, omission, okurigana 等）はサブパターンごとに均等化
            if etype == "insertion":
                # 重複入力(dup) と 非正当挿入(dup_p/filler) を 50:50 に寄せる
                dup_pool = [c for c in avail if c.subkind == "dup"]
                ins_pool = [c for c in avail if c.subkind in ("dup_p", "filler")]
                if dup_pool and ins_pool:
                    if rng.random() < 0.5:
                        avail = dup_pool
                    else:
                        sub_p = [c for c in ins_pool if c.subkind == "dup_p"]
                        sub_f = [c for c in ins_pool if c.subkind == "filler"]
                        if sub_p and sub_f:
                            avail = sub_p if rng.random() < 0.5 else sub_f
                        else:
                            avail = ins_pool
            elif etype == "okurigana":
                omit_p = [c for c in avail if c.subkind == "omit"]
                exc_p = [c for c in avail if c.subkind == "excess"]
                if omit_p and exc_p:
                    avail = omit_p if rng.random() < 0.45 else exc_p
            elif etype == "omission":
                by_sub: Dict[str, List[Candidate]] = {}
                for c in avail:
                    by_sub.setdefault(c.subkind, []).append(c)
                if len(by_sub) > 1:
                    chosen_sub = rng.choice(list(by_sub.keys()))
                    avail = by_sub[chosen_sub]

            by_pos: Dict[Tuple[int, int], List[Candidate]] = {}
            for c in avail:
                by_pos.setdefault((c.start, c.end), []).append(c)
            pos_key = rng.choice(list(by_pos.keys()))
            c = rng.choice(by_pos[pos_key])
            chosen.append(c)
            used_ranges.append((c.start, c.end))

        if not chosen:
            return text, []

        chosen_desc = sorted(chosen, key=lambda c: (c.start, c.end), reverse=True)
        cur_text = text
        for c in chosen_desc:
            cur_text = cur_text[: c.start] + c.replacement + cur_text[c.end :]

        chosen_asc = sorted(chosen, key=lambda c: (c.start, c.end))
        spans: List[dict] = []
        delta = 0
        for c in chosen_asc:
            new_start = c.start + delta
            new_end = new_start + len(c.replacement)
            assert cur_text[new_start:new_end] == c.corrupted, (
                f"注入スパン不整合: {cur_text[new_start:new_end]!r} != {c.corrupted!r}"
            )
            spans.append(
                {
                    "start": new_start,
                    "end": new_end,
                    "type": c.err_type,
                    "original": c.original,
                    "corrupted": c.corrupted,
                }
            )
            delta += len(c.replacement) - (c.end - c.start)

        return cur_text, spans
