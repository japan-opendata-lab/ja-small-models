#!/usr/bin/env python3
"""第7号 ja-address-ner-v1 パイロットデータ構築スクリプト。

公的マスター（municipalities.json, town_index.json, old_municipalities.json）由来の
合成架空住所のみを使用し、Nemotron-PII Japan（CC BY 4.0、文脈の器のみ）と
分割非重複の定型・自由記述テンプレートへ埋め込んで train / dev / sealed-test を構築する。

出力（data/pilot/）:
  - train.jsonl       (20,000件)
  - dev.jsonl         (2,000件)
  - sealed-test.jsonl (2,000件: 構築後一切開封しない)
  - build-stats.json  (分割別統計・カテゴリ別分布・リーク検査結果)
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from address_synth import AddressSynthesizer  # noqa: E402
import offsets_util as ou  # noqa: E402

NEMOTRON_DIR = Path(os.environ.get("NEMOTRON_DIR", "data"))  # folder holding the Nemotron-PII Japan files
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "pilot"

CATEGORY_RATIOS: List[Tuple[str, float]] = [
    ("standard", 0.25),
    ("omit_pref", 0.12),
    ("same_name_resolved", 0.08),
    ("old_muni", 0.07),
    ("variant", 0.12),
    ("omit_block", 0.08),
    ("reordered", 0.04),
    ("unresolvable_ambiguous", 0.08),
    ("unresolvable_nonexistent", 0.06),
    ("no_address", 0.10),
]

# 分割別・完全非重複テンプレート（単住所・複数住所・ハードネガティブ）
# {A} = 住所（先頭に任意で 〒xxx-xxxx を含む文字列）
SINGLE_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    "train": [
        ("tpl_tr_s01", "申込者住所：{A}"),
        ("tpl_tr_s02", "現住所は{A}となっております。"),
        ("tpl_tr_s03", "お届け先は{A}までお願いいたします。"),
        ("tpl_tr_s04", "書類の送付先：{A}（日中受取可）"),
        ("tpl_tr_s05", "ご登録住所を{A}へ変更いたしました。"),
        ("tpl_tr_s06", "勤務先の所在地は{A}です。"),
        ("tpl_tr_s07", "配送先住所：{A}宛てにて発送手配します。"),
        ("tpl_tr_s08", "領収書記載住所：{A}"),
        ("tpl_tr_s09", "お住まいは{A}でお間違いないでしょうか。"),
        ("tpl_tr_s10", "物件所在地：{A}（最寄り駅より徒歩10分）"),
        ("tpl_tr_s11", "契約書の住所欄に「{A}」とご記入ください。"),
        ("tpl_tr_s12", "返送先はこちらです→ {A}"),
        ("tpl_tr_s13", "開催会場：{A} 受付は13時からです。"),
        ("tpl_tr_s14", "緊急連絡先住所：{A}"),
        ("tpl_tr_s15", "名刺の所在地表記：{A}"),
        ("tpl_tr_s16", "住民票記載の住所は{A}になります。"),
        ("tpl_tr_s17", "荷物の集荷先を{A}に設定しました。"),
        ("tpl_tr_s18", "次回訪問先は{A}を予定しております。"),
        ("tpl_tr_s19", "保険証の登録住所：{A}"),
        ("tpl_tr_s20", "口座開設申込書記載の住所は{A}です。"),
        ("tpl_tr_s21", "請求書の郵送先：{A}"),
        ("tpl_tr_s22", "引越し後の住所は{A}です。ご確認ください。"),
        ("tpl_tr_s23", "タクシーの配車先：{A}の正面にお願いします。"),
        ("tpl_tr_s24", "店舗アクセス：{A}にて営業中です。"),
        ("tpl_tr_s25", "顧客台帳メモ：所在地 {A} / 担当確認済み"),
        ("tpl_tr_s26", "会員登録情報\n住所：{A}\n確認ステータス：済"),
        ("tpl_tr_s27", "【ご住所】{A}\n上記宛先へ案内状をお送りします。"),
        ("tpl_tr_s28", "差出人住所：{A}"),
        ("tpl_tr_s29", "お送りした住所は{A}となります。"),
        ("tpl_tr_s30", "転出先：{A}（来月1日より入居予定）"),
        ("tpl_tr_s31", "定期便のお届け先として{A}を登録しました。"),
        ("tpl_tr_s32", "現地確認場所：{A} 周辺にて待機願います。"),
        ("tpl_tr_s33", "保証人様住所：{A}"),
        ("tpl_tr_s34", "法人登記上の本店所在地は{A}です。"),
        ("tpl_tr_s35", "ふるさと納税ワンストップ特例申請住所：{A}"),
        ("tpl_tr_s36", "運転免許証の表面記載住所：{A}"),
        ("tpl_tr_s37", "工事実施場所：{A}（作業予定時間：午前中）"),
        ("tpl_tr_s38", "メール本文より抜粋：私の住所は{A}になります。"),
        ("tpl_tr_s39", "宅配便の伝票宛先：{A}"),
        ("tpl_tr_s40", "アンケート回答者住所：{A}"),
        ("tpl_tr_s41", "面談場所のご案内：{A}までお越しください。"),
        ("tpl_tr_s42", "返礼品配送先：{A}（時間指定なし）"),
        ("tpl_tr_s43", "災害時避難先登録：{A}"),
        ("tpl_tr_s44", "通知書送付先：{A}宛"),
        ("tpl_tr_s45", "駐車場契約地：{A}"),
        ("tpl_tr_s46", "車庫証明の保管場所：{A}"),
        ("tpl_tr_s47", "問い合わせフォーム入力欄【住所】：{A}"),
        ("tpl_tr_s48", "備考欄メモ：相手方所在地は{A}と記載あり。"),
    ],
    "dev": [
        ("tpl_dv_s01", "申請書の記載住所：{A}（本人確認書類と一致）"),
        ("tpl_dv_s02", "新規お届け先として{A}が指定されました。"),
        ("tpl_dv_s03", "チャットログ抜粋「明日の集合場所は{A}で合っていますか？」"),
        ("tpl_dv_s04", "届出住所：{A}／確認日：今月15日"),
        ("tpl_dv_s05", "資料請求の郵送住所は{A}でございます。"),
        ("tpl_dv_s06", "【配送指示】{A} 玄関前置き配希望"),
        ("tpl_dv_s07", "取引先所在地メモ：{A}（請求書PDFより）"),
        ("tpl_dv_s08", "ご自宅の住所を「{A}」として登録完了しました。"),
        ("tpl_dv_s09", "出張報告：訪問先住所 {A}"),
        ("tpl_dv_s10", "本人連絡先\n居住地：{A}"),
        ("tpl_dv_s11", "修理品の返送先住所：{A}宛にお願いいたします。"),
        ("tpl_dv_s12", "学籍登録情報：保護者住所は{A}です。"),
        ("tpl_dv_s13", "口座振替依頼書 住所：{A}"),
        ("tpl_dv_s14", "受付票より：{A}にお住まいの方からのご照会です。"),
    ],
    "sealed-test": [
        ("tpl_st_s01", "更新手続き用の案内は{A}へ発送済みです。"),
        ("tpl_st_s02", "顧客連絡メモ：「引っ越し先は{A}に決まりました」"),
        ("tpl_st_s03", "【受取場所】{A} 宅配ロッカー利用"),
        ("tpl_st_s04", "登記事項記載の住所は{A}となっております。"),
        ("tpl_st_s05", "利用申込書 住所記入欄：{A}"),
        ("tpl_st_s06", "サンプル品発送先：{A}（平日午後着指定）"),
        ("tpl_st_s07", "提携事務所所在地：{A}"),
        ("tpl_st_s08", "ご指定の住所「{A}」への手配を承りました。"),
        ("tpl_st_s09", "点検作業場所：{A} 担当者までご連絡ください。"),
        ("tpl_st_s10", "申込情報詳細\n現住所：{A}"),
        ("tpl_st_s11", "契約者本人住所：{A}にて受理いたしました。"),
        ("tpl_st_s12", "帰省先連絡住所：{A}"),
        ("tpl_st_s13", "定期点検案内ハガキ宛先：{A}"),
        ("tpl_st_s14", "サポート窓口記録：ユーザー居住地は{A}と申告あり。"),
    ],
}

# 複数住所（2住所：{A1}, {A2}）テンプレート
MULTI_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    "train": [
        ("tpl_tr_m01", "転居前の住所は{A1}、転居後の新住所は{A2}です。"),
        ("tpl_tr_m02", "発送元：{A1}\n配送先：{A2}"),
        ("tpl_tr_m03", "本社所在地は{A1}ですが、実務拠点は{A2}にございます。"),
        ("tpl_tr_m04", "契約者住所：{A1}／連帯保証人住所：{A2}"),
        ("tpl_tr_m05", "登録住所「{A1}」から「{A2}」への変更届を受け付けました。"),
        ("tpl_tr_m06", "住民票上の住所：{A1}、実際の居所：{A2}"),
        ("tpl_tr_m07", "午前中の訪問先は{A1}、午後の訪問先は{A2}となります。"),
        ("tpl_tr_m08", "請求書送付先：{A1}、納品場所：{A2}"),
        ("tpl_tr_m09", "ご本人様住所（{A1}）および緊急連絡先住所（{A2}）を確認しました。"),
        ("tpl_tr_m10", "旧所在地：{A1}\n新所在地：{A2}"),
    ],
    "dev": [
        ("tpl_dv_m01", "引越し元：{A1} → 引越し先：{A2}"),
        ("tpl_dv_m02", "申込者住所は{A1}で、書類受取先は{A2}です。"),
        ("tpl_dv_m03", "集荷先（{A1}）からお届け先（{A2}）への便を手配します。"),
    ],
    "sealed-test": [
        ("tpl_st_m01", "従前の住所：{A1}／変更後の住所：{A2}"),
        ("tpl_st_m02", "差出人（{A1}）より受取人（{A2}）宛に送付されました。"),
        ("tpl_st_m03", "本店：{A1}\n支社：{A2}"),
    ],
}

# ハードネガティブ（no_address）テンプレート（地名・施設風語彙を含みつつ住所ではない文）
NO_ADDR_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    "train": [
        ("tpl_tr_n01", "本日の昼食は北海道産じゃがいもを使ったカレーと札幌ラーメン風スープです。"),
        ("tpl_tr_n02", "明日の会議には株式会社サクラ企画の営業部から3名が参加する予定です。"),
        ("tpl_tr_n03", "東京駅と大阪駅を結ぶ新幹線の座席予約をオンラインで完了しました。"),
        ("tpl_tr_n04", "テスト用の数字列 123-4567 および照合番号 9876-5432 を入力してください。"),
        ("tpl_tr_n05", "京都風抹茶スイーツと名古屋めしフェアを地下1階の特設会場で開催しています。"),
        ("tpl_tr_n06", "システムログ：エラーコード E-404 が午前10時15分に発生しました。"),
        ("tpl_tr_n07", "横浜中華街スタイルの肉まんと仙台牛タン弁当を通販で取り寄せました。"),
        ("tpl_tr_n08", "お支払い金額は合計 24,800円（税込）で、振込期限は今月末です。"),
        ("tpl_tr_n09", "博多うどんや広島お好み焼きなどのご当地グルメ特集がテレビで放送されました。"),
        ("tpl_tr_n10", "お客様サポート窓口の受付時間は平日9時から17時までとなっております。"),
        ("tpl_tr_n11", "渋谷スクランブル風のアート展示が美術館の3階ギャラリーで公開中です。"),
        ("tpl_tr_n12", "担当者名：佐藤 健太、内線番号：4512、部署：情報システム推進室。"),
        ("tpl_tr_n13", "新しいソフトウェアのバージョン 2.4.1 では検索速度が大幅に改善されました。"),
        ("tpl_tr_n14", "神戸牛ステーキのコース料理を予約サイトから3名分で申し込みました。"),
        ("tpl_tr_n15", "注文番号 #A-88219 の出荷準備が完了し、間もなく追跡番号が発行されます。"),
        ("tpl_tr_n16", "沖縄黒糖を使った菓子折りをお土産として取引先の皆様にお渡ししました。"),
        ("tpl_tr_n17", "アンケートへのご協力ありがとうございます。回答内容は統計処理のみに利用します。"),
        ("tpl_tr_n18", "商品型番：XYZ-102-304 の在庫状況は現在「在庫あり」となっています。"),
        ("tpl_tr_n19", "信州そばと讃岐うどんの食べ比べセットが今月の売れ筋ランキング1位です。"),
        ("tpl_tr_n20", "ご不明点がある場合はFAQページをご確認のうえ、フォームよりご連絡ください。"),
    ],
    "dev": [
        ("tpl_dv_n01", "長崎ちゃんぽんと宮崎チキン南蛮のランチセットが今週の日替わりメニューです。"),
        ("tpl_dv_n02", "品番 B-12-345 の交換部品を倉庫の2番棚から取り出して梱包しました。"),
        ("tpl_dv_n03", "新潟産コシヒカリと青森りんごジュースの詰め合わせがギフトに人気です。"),
        ("tpl_dv_n04", "プロジェクト識別子 01101-TEST のビルド検証が全件正常終了しました。"),
        ("tpl_dv_n05", "株式会社みらい商事の経理担当より請求データ修正の連絡が入りました。"),
        ("tpl_dv_n06", "金沢金箔入りの和菓子と宇治茶のセットを来客用のお茶菓子に用意しました。"),
    ],
    "sealed-test": [
        ("tpl_st_n01", "秋田きりたんぽ鍋と盛岡冷麺を冬の定番商品としてラインナップに加えました。"),
        ("tpl_st_n02", "検証用コード 13113-9999 を入力してテスト環境の接続状態を確認します。"),
        ("tpl_st_n03", "高知かつおのたたきと徳島すだちサイダーが物産展で好評を博しています。"),
        ("tpl_st_n04", "有限会社あおば製作所の設計チームが来週の定例レビューに同席します。"),
        ("tpl_st_n05", "浜松うなぎパイ風のお菓子を休憩室のテーブルに置いておきました。"),
        ("tpl_st_n06", "システム更新メンテナンスは日曜日深夜2時から4時までの間に実施されます。"),
    ],
}

# 文中への偶発的な行政住所・郵便番号の紛れ込みを検知するパターン
_PREF_CITY_GUARD_RE = re.compile(
    r"(北海道|東京都|(?:大阪|京都)府|.{2,3}県)[^\s、。・：:\n]{1,8}[市区町村]"
)
_POSTAL_PAT_RE = re.compile(r"〒?\s*[0-9０-９]{3}[-ー－][0-9０-９]{4}")


class MinHashLSH:
    """文字 5-gram MinHash LSH（num_perm=64, bands=16, rows=4）による近傍重複検査器。"""

    def __init__(self, num_perm: int = 64, bands: int = 16) -> None:
        assert num_perm % bands == 0
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.modulus = (1 << 61) - 1
        rng = random.Random(20261119)
        self.coeffs = [
            (rng.randrange(1, self.modulus), rng.randrange(0, self.modulus))
            for _ in range(num_perm)
        ]
        self.buckets: List[Dict[Tuple[int, ...], List[int]]] = [
            collections.defaultdict(list) for _ in range(bands)
        ]
        self.shingle_sets: List[Set[int]] = []
        self.signatures: List[List[int]] = []
        self.split_tags: List[str] = []

    @staticmethod
    def _norm_for_lsh(text: str) -> str:
        return re.sub(r"\s+", "", text)

    def _shingles(self, text: str) -> Set[int]:
        s = self._norm_for_lsh(text)
        if len(s) < 5:
            return {zlib_crc(s)} if s else set()
        return {zlib_crc(s[i : i + 5]) for i in range(len(s) - 4)}

    def signature(self, text: str) -> Tuple[Set[int], List[int]]:
        sh = self._shingles(text)
        if not sh:
            return sh, [0] * self.num_perm
        sig = []
        mod = self.modulus
        for a, b in self.coeffs:
            m = mod
            for x in sh:
                val = (a * x + b) % mod
                if val < m:
                    m = val
            sig.append(m)
        return sh, sig

    def add_doc(self, text: str, split_tag: str) -> int:
        sh, sig = self.signature(text)
        idx = len(self.shingle_sets)
        self.shingle_sets.append(sh)
        self.signatures.append(sig)
        self.split_tags.append(split_tag)
        for bi in range(self.bands):
            st = bi * self.rows
            key = tuple(sig[st : st + self.rows])
            self.buckets[bi][key].append(idx)
        return idx

    def has_near_dup_in_other_splits(
        self, text: str, my_split: str, threshold: float = 0.72
    ) -> bool:
        sh, sig = self.signature(text)
        if not sh:
            return False
        cand_idxs: Set[int] = set()
        for bi in range(self.bands):
            st = bi * self.rows
            key = tuple(sig[st : st + self.rows])
            for c_idx in self.buckets[bi].get(key, ()):
                if self.split_tags[c_idx] != my_split:
                    cand_idxs.add(c_idx)
        for c_idx in cand_idxs:
            other = self.shingle_sets[c_idx]
            union = len(sh | other)
            if union and (len(sh & other) / union) >= threshold:
                return True
        return False


def zlib_crc(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode("utf-8"))


def format_postal(postal7: str, rng: random.Random) -> str:
    p = f"{postal7[:3]}-{postal7[3:]}"
    return f"〒{p}" if rng.random() < 0.85 else p


def build_address_parts(
    addr_info: dict, rng: random.Random, add_postal: bool
) -> Tuple[List[Tuple[Optional[str], str]], dict]:
    """合成住所のセグメント列と住所メタ情報（郵便番号前置の任意付与含む）を組み立てる。"""
    segs: List[Tuple[Optional[str], str]] = []
    if add_postal and addr_info["postal"]:
        segs.append((None, format_postal(addr_info["postal"], rng) + rng.choice([" ", "　", ""])))
    segs.extend(addr_info["segments"])
    return segs, addr_info


def extract_nemotron_window(
    doc: dict, max_chars: int = 230
) -> Optional[Tuple[str, int, int, List[Tuple[int, int]]]]:
    """Nemotron full_1addr 文書から単一 ADDRESS と任意 POSTAL_CODE を含む自然な窓を切り出す。"""
    text = doc["text"]
    spans = doc["spans"]
    asps = [s for s in spans if s["label"] == "ADDRESS"]
    if len(asps) != 1:
        return None
    a0 = asps[0]
    ast, aed = int(a0["start"]), int(a0["end"])
    pre_ctx = text[max(0, ast - 10) : ast]
    if any(k in pre_ctx for k in ("都道府県", "市区町村", "町域", "番地")):
        return None
    if len(a0["text"]) < 8 or not re.search(r"[0-9０-９一二三四五六七八九十丁目番号]", a0["text"]):
        return None

    # 住所前後 ~90文字の範囲から、空行(\n\n)や句点(。)を優先して複数行・文脈ごと切り出す
    left_min = max(0, ast - 105)
    right_max = min(len(text), aed + 105)

    # 左端：空行優先、なければ left_min より右にある最も左の改行/句点
    dbl_l = text.rfind("\n\n", left_min, ast)
    if dbl_l != -1:
        win_s = dbl_l + 2
    else:
        nl_first = text.find("\n", left_min, ast - 12 if ast - 12 > left_min else ast)
        dot_first = text.find("。", left_min, ast)
        cands_l = [p + 1 for p in (nl_first, dot_first) if p != -1]
        win_s = min(cands_l) if cands_l else left_min

    # 右端：空行優先、なければ right_max より左にある最も右の改行/句点
    dbl_r = text.find("\n\n", aed, right_max)
    if dbl_r != -1:
        win_e = dbl_r
    else:
        nl_last = text.rfind("\n", aed + 8 if aed + 8 < right_max else aed, right_max)
        dot_last = text.rfind("。", aed, right_max)
        cands_r = ([dot_last + 1] if dot_last != -1 else []) + (
            [nl_last] if nl_last != -1 else []
        )
        win_e = max(cands_r) if cands_r else right_max

    if win_e - win_s > max_chars:
        win_s = max(win_s, ast - 85)
        win_e = min(win_e, aed + 85)

    raw_slice = text[win_s:win_e]
    win_text = raw_slice.strip()
    if len(win_text) < (aed - ast) + 12:
        return None
    lead_ws = len(raw_slice) - len(raw_slice.lstrip())
    win_s += lead_ws
    win_e = win_s + len(win_text)

    rel_ast = ast - win_s
    rel_aed = aed - win_s
    # ウィンドウ内の ADDRESS 以外のテキストに偶然の都道府県+市区町村がないか検査
    outside_text = win_text[:rel_ast] + "\n" + win_text[rel_aed:]
    if _PREF_CITY_GUARD_RE.search(outside_text):
        return None

    # ウィンドウ内の POSTAL_CODE スパン（または〒+7桁郵便番号）を検出
    pc_rels: List[Tuple[int, int]] = []
    for s in spans:
        if s["label"] == "POSTAL_CODE":
            ps, pe = int(s["start"]), int(s["end"])
            if ps >= win_s and pe <= win_e and (pe <= ast or ps >= aed):
                # 直前の「〒」も置換対象範囲に含める
                rps, rpe = ps - win_s, pe - win_s
                if rps >= 1 and win_text[rps - 1] == "〒":
                    rps -= 1
                pc_rels.append((rps, rpe))
    # スパン外の郵便番号パターンが残っていれば拒否
    masked = list(win_text)
    for rps, rpe in pc_rels:
        for idx in range(rps, rpe):
            masked[idx] = "X"
    masked_outside = "".join(masked[:rel_ast]) + "\n" + "".join(masked[rel_aed:])
    if _POSTAL_PAT_RE.search(masked_outside):
        return None

    return win_text, rel_ast, rel_aed, sorted(pc_rels)


def render_nemotron_sample(
    win_text: str,
    rel_ast: int,
    rel_aed: int,
    pc_rels: List[Tuple[int, int]],
    addr_info: dict,
    rng: random.Random,
) -> Optional[Tuple[str, List[dict], List[dict]]]:
    """Nemotron 切り出し窓の元 ADDRESS（と POSTAL_CODE）を合成住所へ置換しスパンを記録する。"""
    # 置換操作を位置順に組み立て
    repls: List[Tuple[int, int, List[Tuple[Optional[str], str]]]] = []
    addr_segs = list(addr_info["segments"])
    # もし郵便番号スパンが窓内にあり、かつ住所が UNRESOLVABLE なら郵便番号ごと消す（答え漏れ防止）
    for rps, rpe in pc_rels:
        if addr_info["expected_status"] == "RESOLVED" and addr_info["postal"]:
            repls.append((rps, rpe, [(None, format_postal(addr_info["postal"], rng))]))
        else:
            # 「郵便番号: 123-4567」等のラベルごと壊れないよう "記載なし" へ無害化
            repls.append((rps, rpe, [(None, "未記載")]))
    repls.append((rel_ast, rel_aed, addr_segs))
    repls.sort(key=lambda x: x[0])

    # 重なりが起きないことを保証
    for i in range(len(repls) - 1):
        if repls[i][1] > repls[i + 1][0]:
            return None

    out_parts: List[str] = []
    comp_spans: List[dict] = []
    pos = 0
    cursor = 0
    addr_start = -1
    addr_end = -1

    for st, ed, segs in repls:
        pre = win_text[cursor:st]
        out_parts.append(pre)
        pos += len(pre)
        is_addr = st == rel_ast
        if is_addr:
            addr_start = pos
        for lb, txt in segs:
            if lb is not None:
                comp_spans.append(
                    {"start": pos, "end": pos + len(txt), "label": lb, "text": txt}
                )
            out_parts.append(txt)
            pos += len(txt)
        if is_addr:
            addr_end = pos
        cursor = ed

    tail = win_text[cursor:]
    out_parts.append(tail)
    full_text = "".join(out_parts)

    comp_map = {c["label"]: c["text"] for c in comp_spans}
    gold_addr = {
        "start": addr_start,
        "end": addr_end,
        "text": full_text[addr_start:addr_end],
        "category": addr_info["category"],
        "subtype": addr_info["subtype"],
        "expected_status": addr_info["expected_status"],
        "expected_code": addr_info["expected_code"],
        "linker_reason": addr_info["linker_reason"],
        "components": comp_map,
    }
    return full_text, comp_spans, [gold_addr]


def render_template_single(
    tpl_str: str,
    addr_info: dict,
    rng: random.Random,
    add_postal: bool,
) -> Tuple[str, List[dict], List[dict]]:
    """単一住所テンプレートへ合成住所を埋め込む。"""
    prefix, suffix = tpl_str.split("{A}", 1)
    segs, _ = build_address_parts(addr_info, rng, add_postal=add_postal)

    parts: List[str] = [prefix]
    pos = len(prefix)
    comp_spans: List[dict] = []
    a_start: Optional[int] = None
    a_end: Optional[int] = None

    for lb, txt in segs:
        if lb is not None:
            if a_start is None:
                a_start = pos
            comp_spans.append(
                {"start": pos, "end": pos + len(txt), "label": lb, "text": txt}
            )
            a_end = pos + len(txt)
        parts.append(txt)
        pos += len(txt)

    parts.append(suffix)
    full_text = "".join(parts)
    assert a_start is not None and a_end is not None

    comp_map = {c["label"]: c["text"] for c in comp_spans}
    gold_addr = {
        "start": a_start,
        "end": a_end,
        "text": full_text[a_start:a_end],
        "category": addr_info["category"],
        "subtype": addr_info["subtype"],
        "expected_status": addr_info["expected_status"],
        "expected_code": addr_info["expected_code"],
        "linker_reason": addr_info["linker_reason"],
        "components": comp_map,
    }
    return full_text, comp_spans, [gold_addr]


def render_template_multi(
    tpl_str: str,
    addr1: dict,
    addr2: dict,
    rng: random.Random,
    add_postal1: bool,
    add_postal2: bool,
) -> Tuple[str, List[dict], List[dict]]:
    """2住所テンプレートへ合成住所2件を埋め込む。"""
    p1 = tpl_str.index("{A1}")
    p2 = tpl_str.index("{A2}")
    assert p1 < p2
    chunks = [tpl_str[:p1], tpl_str[p1 + 4 : p2], tpl_str[p2 + 4 :]]

    parts: List[str] = [chunks[0]]
    pos = len(chunks[0])
    comp_spans: List[dict] = []
    gold_addrs: List[dict] = []

    for idx_a, (ainfo, add_p, tail) in enumerate(
        [(addr1, add_postal1, chunks[1]), (addr2, add_postal2, chunks[2])]
    ):
        segs, _ = build_address_parts(ainfo, rng, add_postal=add_p)
        a_start: Optional[int] = None
        a_end: Optional[int] = None
        cur_comps: Dict[str, str] = {}
        for lb, txt in segs:
            if lb is not None:
                if a_start is None:
                    a_start = pos
                comp_spans.append(
                    {"start": pos, "end": pos + len(txt), "label": lb, "text": txt}
                )
                cur_comps[lb] = txt
                a_end = pos + len(txt)
            parts.append(txt)
            pos += len(txt)
        assert a_start is not None and a_end is not None
        gold_addrs.append(
            {
                "start": a_start,
                "end": a_end,
                "text": None,  # 後段でスライス確定
                "category": ainfo["category"],
                "subtype": ainfo["subtype"],
                "expected_status": ainfo["expected_status"],
                "expected_code": ainfo["expected_code"],
                "linker_reason": ainfo["linker_reason"],
                "components": cur_comps,
            }
        )
        parts.append(tail)
        pos += len(tail)

    full_text = "".join(parts)
    for ga in gold_addrs:
        ga["text"] = full_text[ga["start"] : ga["end"]]
    return full_text, comp_spans, gold_addrs


def load_nemotron_pools(
    split_file: Path, seed: int
) -> Tuple[List[Tuple[str, str, int, int, List[Tuple[int, int]]]], List[Tuple[str, str]]]:
    """Nemotron JSONL から住所窓候補プールと非住所文プールを抽出する。"""
    addr_pool: List[Tuple[str, str, int, int, List[Tuple[int, int]]]] = []
    noaddr_pool: List[Tuple[str, str]] = []
    seen_doc_hash: Set[str] = set()

    with open(split_file, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            raw_txt = doc.get("text", "")
            doc_hash = hashlib.md5(raw_txt.encode("utf-8")).hexdigest()[:16]
            if doc_hash in seen_doc_hash:
                continue
            seen_doc_hash.add(doc_hash)

            spans = doc.get("spans", [])
            has_addr_or_pc = any(
                s["label"] in ("ADDRESS", "POSTAL_CODE") for s in spans
            )
            if not has_addr_or_pc:
                # 住所・郵便番号を含まない業務文書：都道府県+市区町村等を含まない自然な段落を切り出す
                if _PREF_CITY_GUARD_RE.search(raw_txt) or _POSTAL_PAT_RE.search(raw_txt):
                    continue
                paras = [
                    p.strip()
                    for p in raw_txt.split("\n\n")
                    if 30 <= len(p.strip()) <= 200
                ]
                if not paras:
                    paras = [
                        p.strip()
                        for p in raw_txt.split("\n")
                        if 30 <= len(p.strip()) <= 200
                    ]
                if paras:
                    noaddr_pool.append((doc_hash, paras[0]))
                continue

            if any(s["label"] == "ADDRESS" for s in spans):
                win = extract_nemotron_window(doc)
                if win is not None:
                    win_text, rel_ast, rel_aed, pc_rels = win
                    addr_pool.append((doc_hash, win_text, rel_ast, rel_aed, pc_rels))

    rng = random.Random(seed)
    rng.shuffle(addr_pool)
    rng.shuffle(noaddr_pool)
    return addr_pool, noaddr_pool


def allocate_counts(total: int) -> List[Tuple[str, int]]:
    counts = [(cat, int(round(total * ratio))) for cat, ratio in CATEGORY_RATIOS]
    diff = total - sum(c for _, c in counts)
    if diff != 0:
        counts[0] = (counts[0][0], counts[0][1] + diff)
    return counts


def validate_row_integrity(row: dict, linker) -> None:
    """1レコードのスパン整合性・グルーピング一致・リンカー整合性を全件検証する。"""
    text = row["text"]
    comps = row["components"]
    addrs = row["addresses"]

    # 重なりなし＆文字列完全一致
    prev_e = -1
    for c in comps:
        st, ed = c["start"], c["end"]
        assert 0 <= st < ed <= len(text), f"範囲外: {c}"
        assert st >= prev_e, f"構成要素重複: {comps}"
        assert text[st:ed] == c["text"], f"テキスト不一致: {c['text']!r} != {text[st:ed]!r}"
        prev_e = ed

    # グルーピング規則の一致確認
    grouped = ou.group_components_to_addresses(text, comps)
    assert len(grouped) == len(addrs), (
        f"グルーピング件数不一致 ({row['id']}): {len(grouped)} != {len(addrs)} | text={text!r}"
    )
    for g, ga in zip(grouped, addrs):
        assert g["start"] == ga["start"] and g["end"] == ga["end"], (
            f"住所スパン不一致 ({row['id']}): {g} != {ga}"
        )
        assert g["components"] == ga["components"], (
            f"構成要素マップ不一致 ({row['id']}): {g['components']} != {ga['components']}"
        )
        res = linker.link_components(ga["components"])
        if ga["expected_status"] == "RESOLVED":
            assert (
                res["status"] == "RESOLVED"
                and res["municipality_code"] == ga["expected_code"]
            ), f"リンカー不一致: {ga} -> {res}"
        else:
            assert res["status"] == "UNRESOLVABLE", f"リンカー誤断言: {ga} -> {res}"


def check_leakage(
    splits_rows: Dict[str, List[dict]], lsh: MinHashLSH
) -> Dict[str, dict]:
    """全3ペアの分割間リーク検査を実施し、すべて0であることを assert 検証する。"""
    pairs = [("train", "dev"), ("train", "sealed-test"), ("dev", "sealed-test")]
    meta: Dict[str, dict] = {}
    for sp, rows in splits_rows.items():
        tpls = {r["template_id"] for r in rows if r["source"] == "template"}
        nemos = {r["nemotron_doc_hash"] for r in rows if r["nemotron_doc_hash"]}
        texts = {r["text"] for r in rows}
        addrs = {a["text"] for r in rows for a in r["addresses"]}
        meta[sp] = {"tpls": tpls, "nemos": nemos, "texts": texts, "addrs": addrs}

    # LSH 0.80 全件ペア検査
    near_dup_counts: Dict[Tuple[str, str], int] = collections.Counter()
    doc_refs: List[Tuple[str, str]] = []
    for sp, rows in splits_rows.items():
        for r in rows:
            doc_refs.append((sp, r["id"]))

    for idx, (sp, _) in enumerate(doc_refs):
        sh = lsh.shingle_sets[idx]
        sig = lsh.signatures[idx]
        cands: Set[int] = set()
        for bi in range(lsh.bands):
            st = bi * lsh.rows
            key = tuple(sig[st : st + lsh.rows])
            for c_idx in lsh.buckets[bi].get(key, ()):
                if c_idx > idx and doc_refs[c_idx][0] != sp:
                    cands.add(c_idx)
        for c_idx in cands:
            sp2 = doc_refs[c_idx][0]
            other = lsh.shingle_sets[c_idx]
            union = len(sh | other)
            if union and (len(sh & other) / union) >= 0.80:
                pair_key = (sp, sp2) if (sp, sp2) in pairs else (sp2, sp)
                near_dup_counts[pair_key] += 1

    report: Dict[str, dict] = {}
    for a, b in pairs:
        pair_name = f"{a}__{b}"
        res = {
            "template_id_overlap": len(meta[a]["tpls"] & meta[b]["tpls"]),
            "nemotron_doc_hash_overlap": len(meta[a]["nemos"] & meta[b]["nemos"]),
            "exact_text_overlap": len(meta[a]["texts"] & meta[b]["texts"]),
            "exact_address_surface_overlap": len(meta[a]["addrs"] & meta[b]["addrs"]),
            "near_dup_lsh_080": near_dup_counts[(a, b)],
        }
        report[pair_name] = res
        for k, v in res.items():
            assert v == 0, f"リーク検出 ({pair_name} / {k}): {v}"
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=20000)
    ap.add_argument("--dev-size", type=int, default=2000)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    synth = AddressSynthesizer()

    # Nemotron 分割専用プールの読込（順番注意：dev/sealed-test を先に確保すると衝突ゼロが保ちやすい）
    nemo_pools = {
        "dev": load_nemotron_pools(NEMOTRON_DIR / "nemotron-dev.jsonl", seed=args.seed + 1),
        "sealed-test": load_nemotron_pools(NEMOTRON_DIR / "nemotron-test.jsonl", seed=args.seed + 2),
        "train": load_nemotron_pools(NEMOTRON_DIR / "nemotron-train.jsonl", seed=args.seed + 3),
    }

    # 分割間で共有禁止の既使用住所表層・テキスト集合と LSH インデックス
    addr_to_split: Dict[str, str] = {}
    used_texts_all: Set[str] = set()
    lsh = MinHashLSH(num_perm=64, bands=16)

    split_specs = [
        ("dev", args.dev_size),
        ("sealed-test", args.test_size),
        ("train", args.train_size),
    ]
    splits_rows: Dict[str, List[dict]] = {}

    for split_name, target_n in split_specs:
        cat_counts = allocate_counts(target_n)
        nemo_addr_pool, nemo_noaddr_pool = nemo_pools[split_name]
        nemo_addr_idx = 0
        nemo_noaddr_idx = 0

        single_tpls = SINGLE_TEMPLATES[split_name]
        multi_tpls = MULTI_TEMPLATES[split_name]
        noaddr_tpls = NO_ADDR_TEMPLATES[split_name]

        # 各カテゴリ内の生成アイテムを計画
        tasks: List[str] = []
        for cat, cnt in cat_counts:
            tasks.extend([cat] * cnt)
        rng.shuffle(tasks)

        rows: List[dict] = []

        def sample_unique_addr(cat_req: str) -> dict:
            for _ in range(120):
                ainfo = synth.generate_one(rng, cat_req, split_name)
                surf = ainfo["address_text"]
                # 別分割で既に使われている住所表層文字列は拒否
                prev_sp = addr_to_split.get(surf)
                if prev_sp is not None and prev_sp != split_name:
                    continue
                return ainfo
            raise RuntimeError(f"非重複住所の確保失敗: {cat_req} / {split_name}")

        for idx_row, cat in enumerate(tasks):
            built_row: Optional[dict] = None
            want_nemo = rng.random() < (0.70 if cat == "no_address" else 0.40)
            for attempt in range(40):
                # 最初の8回は Nemotron 優先（弾かれた場合のみテンプレートへフォールバック）
                use_nemo = want_nemo and (attempt < 8)
                if cat == "no_address":
                    if use_nemo and nemo_noaddr_idx < len(nemo_noaddr_pool):
                        doc_hash, ntext = nemo_noaddr_pool[nemo_noaddr_idx]
                        nemo_noaddr_idx += 1
                        if ntext in used_texts_all:
                            continue
                        if lsh.has_near_dup_in_other_splits(ntext, split_name, threshold=0.72):
                            continue
                        built_row = {
                            "id": f"{split_name}-{idx_row:05d}",
                            "split": split_name,
                            "category": "no_address",
                            "source": "nemotron",
                            "template_id": None,
                            "nemotron_doc_hash": doc_hash,
                            "text": ntext,
                            "components": [],
                            "addresses": [],
                        }
                        break
                    else:
                        tpl_id, tpl_str = rng.choice(noaddr_tpls)
                        prefix_tag = rng.choice(["", "連絡メモ：", "件名：確認事項\n", "【備考】", "社内共有：", "通知："])
                        suffix_tag = rng.choice(
                            ["", " 以上です。", " よろしくお願いします。", "（確認済み）", " ※参考まで。", f"（記録No.{rng.randint(100,999)}）"]
                        )
                        ntext = prefix_tag + tpl_str + suffix_tag
                        if ntext in used_texts_all and attempt < 35:
                            continue
                        if lsh.has_near_dup_in_other_splits(ntext, split_name, threshold=0.72):
                            continue
                        built_row = {
                            "id": f"{split_name}-{idx_row:05d}",
                            "split": split_name,
                            "category": "no_address",
                            "source": "template",
                            "template_id": tpl_id,
                            "nemotron_doc_hash": None,
                            "text": ntext,
                            "components": [],
                            "addresses": [],
                        }
                        break

                # 住所ありカテゴリ
                if use_nemo and nemo_addr_idx < len(nemo_addr_pool):
                    doc_hash, win_text, rel_ast, rel_aed, pc_rels = nemo_addr_pool[nemo_addr_idx]
                    nemo_addr_idx += 1
                    ainfo = sample_unique_addr(cat)
                    rendered = render_nemotron_sample(
                        win_text, rel_ast, rel_aed, pc_rels, ainfo, rng
                    )
                    if rendered is None:
                        continue
                    text, comps, addrs = rendered
                    if text in used_texts_all:
                        continue
                    if lsh.has_near_dup_in_other_splits(text, split_name, threshold=0.72):
                        continue
                    built_row = {
                        "id": f"{split_name}-{idx_row:05d}",
                        "split": split_name,
                        "category": cat,
                        "source": "nemotron",
                        "template_id": None,
                        "nemotron_doc_hash": doc_hash,
                        "text": text,
                        "components": comps,
                        "addresses": addrs,
                    }
                    break
                else:
                    # テンプレート埋め込み（約10%を2住所にする）
                    is_multi = rng.random() < 0.10
                    if is_multi:
                        tpl_id, tpl_str = rng.choice(multi_tpls)
                        a1 = sample_unique_addr(cat)
                        # 2件目は別カテゴリでも同一カテゴリでもよい（多様な組合せ）
                        cat2 = cat if rng.random() < 0.45 else rng.choice(
                            ["standard", "omit_pref", "same_name_resolved", "variant", "omit_block", "unresolvable_ambiguous"]
                        )
                        a2 = sample_unique_addr(cat2)
                        add_p1 = (a1["expected_status"] == "RESOLVED") and (rng.random() < 0.18)
                        add_p2 = (a2["expected_status"] == "RESOLVED") and (rng.random() < 0.18)
                        text, comps, addrs = render_template_multi(
                            tpl_str, a1, a2, rng, add_p1, add_p2
                        )
                    else:
                        tpl_id, tpl_str = rng.choice(single_tpls)
                        a1 = sample_unique_addr(cat)
                        add_p = (a1["expected_status"] == "RESOLVED") and (rng.random() < 0.18)
                        text, comps, addrs = render_template_single(
                            tpl_str, a1, rng, add_postal=add_p
                        )

                    if text in used_texts_all:
                        continue
                    if lsh.has_near_dup_in_other_splits(text, split_name, threshold=0.72):
                        continue
                    built_row = {
                        "id": f"{split_name}-{idx_row:05d}",
                        "split": split_name,
                        "category": cat,
                        "source": "template",
                        "template_id": tpl_id,
                        "nemotron_doc_hash": None,
                        "text": text,
                        "components": comps,
                        "addresses": addrs,
                    }
                    break

            assert built_row is not None, f"サンプル構築失敗: split={split_name}, cat={cat}"
            validate_row_integrity(built_row, synth.linker)

            used_texts_all.add(built_row["text"])
            for ga in built_row["addresses"]:
                addr_to_split[ga["text"]] = split_name
            lsh.add_doc(built_row["text"], split_name)
            rows.append(built_row)

        splits_rows[split_name] = rows

    # ID を順番どおり再採番しつつ出力（書き出し順は train / dev / sealed-test）
    leakage_report = check_leakage(splits_rows, lsh)

    stats: dict = {
        "build_sec": round(time.time() - t0, 2),
        "seed": args.seed,
        "splits": {},
        "leakage_checks": leakage_report,
    }

    for sp in ("train", "dev", "sealed-test"):
        rows = splits_rows[sp]
        out_path = OUT_DIR / f"{sp}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        cat_dist = collections.Counter(r["category"] for r in rows)
        src_dist = collections.Counter(r["source"] for r in rows)
        comp_dist = collections.Counter(c["label"] for r in rows for c in r["components"])
        total_addrs = sum(len(r["addresses"]) for r in rows)
        multi_addr_docs = sum(1 for r in rows if len(r["addresses"]) >= 2)
        resolvable_addrs = sum(
            1 for r in rows for a in r["addresses"] if a["expected_status"] == "RESOLVED"
        )
        unresolvable_addrs = sum(
            1 for r in rows for a in r["addresses"] if a["expected_status"] == "UNRESOLVABLE"
        )
        lens = sorted(len(r["text"]) for r in rows)
        stats["splits"][sp] = {
            "total_rows": len(rows),
            "total_addresses": total_addrs,
            "multi_address_docs": multi_addr_docs,
            "resolvable_addresses": resolvable_addrs,
            "unresolvable_addresses": unresolvable_addrs,
            "source_counts": dict(src_dist),
            "category_counts": {c: cat_dist.get(c, 0) for c, _ in CATEGORY_RATIOS},
            "component_counts": {lb: comp_dist.get(lb, 0) for lb in ou.COMPONENTS},
            "text_len": {
                "min": lens[0],
                "median": lens[len(lens) // 2],
                "mean": round(sum(lens) / len(lens), 2),
                "max": lens[-1],
            },
        }

    (OUT_DIR / "build-stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
