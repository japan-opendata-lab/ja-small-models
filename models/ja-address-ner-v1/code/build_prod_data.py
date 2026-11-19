#!/usr/bin/env python3
"""第7号 ja-address-ner-v1 本番データ構築スクリプト。

パイロット（build_pilot_data.py）と同じ方式・カテゴリ配分（10カテゴリ）・
リーク遮断規律で規模を拡大し、data/prod/ に以下を構築する：
  - train.jsonl       (100,000件)
  - dev.jsonl         (5,000件)
  - sealed-test.jsonl (5,000件：構築後、最終評価1回まで一切開封しない)
  - build-stats.json  (分割別統計・カテゴリ別分布・全3ペアリーク検査結果)

注意: パイロットと同じく実在の個人住所は一切使用しない。
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
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from address_synth import AddressSynthesizer  # noqa: E402
import offsets_util as ou  # noqa: E402

NEMOTRON_DIR = Path(os.environ.get("NEMOTRON_DIR", "data"))  # folder holding the Nemotron-PII Japan files
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "prod"

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

_PREF_CITY_GUARD_RE = re.compile(
    r"(北海道|東京都|(?:大阪|京都)府|.{2,3}県)[^\s、。・：:\n]{1,8}[市区町村]"
)
_POSTAL_PAT_RE = re.compile(r"〒?\s*[0-9０-９]{3}[-ー－][0-9０-９]{4}")

# 分割別 業務コンテキスト修飾（文字重なりや住所誤認の恐れがない安全な語のみ）
STAFF_NAMES: Dict[str, List[str]] = {
    "train": ["佐藤", "鈴木", "高橋", "田中", "渡辺", "伊藤", "山本", "中村", "小林", "加藤", "吉田", "山田"],
    "dev": ["斎藤", "松本", "井上", "木村"],
    "sealed-test": ["清水", "山崎", "池田", "橋本"],
}
DEPTS: Dict[str, List[str]] = {
    "train": ["総務部", "営業一課", "カスタマーサポート", "配送管理課", "審査部門", "経理部", "法務室", "契約管理課"],
    "dev": ["受付窓口", "顧客対応室", "事務センター", "業務企画部"],
    "sealed-test": ["品質管理部", "登録係", "サポート室", "運営事務局"],
}

# 単住所テンプレート（train 100種 / dev 30種 / sealed-test 30種 = 計160種、分割完全非重複）
SINGLE_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    "train": [
        ("tpl_ptr_s001", "申込者住所：{A}"),
        ("tpl_ptr_s002", "現住所は{A}となっております。"),
        ("tpl_ptr_s003", "お届け先は{A}までお願いいたします。"),
        ("tpl_ptr_s004", "書類の送付先：{A}（日中受取可）"),
        ("tpl_ptr_s005", "ご登録住所を{A}へ変更いたしました。"),
        ("tpl_ptr_s006", "勤務先の所在地は{A}です。"),
        ("tpl_ptr_s007", "配送先住所：{A}宛てにて発送手配します。"),
        ("tpl_ptr_s008", "領収書記載住所：{A}"),
        ("tpl_ptr_s009", "お住まいは{A}でお間違いないでしょうか。"),
        ("tpl_ptr_s010", "物件所在地：{A}（最寄り駅より徒歩10分）"),
        ("tpl_ptr_s011", "契約書の住所欄に「{A}」とご記入ください。"),
        ("tpl_ptr_s012", "返送先はこちらです→ {A}"),
        ("tpl_ptr_s013", "開催会場：{A} 受付は13時からです。"),
        ("tpl_ptr_s014", "緊急連絡先住所：{A}"),
        ("tpl_ptr_s015", "名刺の所在地表記：{A}"),
        ("tpl_ptr_s016", "住民票記載の住所は{A}になります。"),
        ("tpl_ptr_s017", "荷物の集荷先を{A}に設定しました。"),
        ("tpl_ptr_s018", "次回訪問先は{A}を予定しております。"),
        ("tpl_ptr_s019", "保険証の登録住所：{A}"),
        ("tpl_ptr_s020", "口座開設申込書記載の住所は{A}です。"),
        ("tpl_ptr_s021", "請求書の郵送先：{A}"),
        ("tpl_ptr_s022", "引越し後の住所は{A}です。ご確認ください。"),
        ("tpl_ptr_s023", "タクシーの配車先：{A}の正面にお願いします。"),
        ("tpl_ptr_s024", "店舗アクセス：{A}にて営業中です。"),
        ("tpl_ptr_s025", "顧客台帳メモ：所在地 {A} / 担当確認済み"),
        ("tpl_ptr_s026", "会員登録情報\n住所：{A}\n確認ステータス：済"),
        ("tpl_ptr_s027", "【ご住所】{A}\n上記宛先へ案内状をお送りします。"),
        ("tpl_ptr_s028", "差出人住所：{A}"),
        ("tpl_ptr_s029", "お送りした住所は{A}となります。"),
        ("tpl_ptr_s030", "転出先：{A}（来月1日より入居予定）"),
        ("tpl_ptr_s031", "定期便のお届け先として{A}を登録しました。"),
        ("tpl_ptr_s032", "現地確認場所：{A} 周辺にて待機願います。"),
        ("tpl_ptr_s033", "保証人様住所：{A}"),
        ("tpl_ptr_s034", "法人登記上の本店所在地は{A}です。"),
        ("tpl_ptr_s035", "ふるさと納税ワンストップ特例申請住所：{A}"),
        ("tpl_ptr_s036", "運転免許証の表面記載住所：{A}"),
        ("tpl_ptr_s037", "工事実施場所：{A}（作業予定時間：午前中）"),
        ("tpl_ptr_s038", "メール本文より抜粋：私の住所は{A}になります。"),
        ("tpl_ptr_s039", "宅配便の伝票宛先：{A}"),
        ("tpl_ptr_s040", "アンケート回答者住所：{A}"),
        ("tpl_ptr_s041", "面談場所のご案内：{A}までお越しください。"),
        ("tpl_ptr_s042", "返礼品配送先：{A}（時間指定なし）"),
        ("tpl_ptr_s043", "災害時避難先登録：{A}"),
        ("tpl_ptr_s044", "通知書送付先：{A}宛"),
        ("tpl_ptr_s045", "駐車場契約地：{A}"),
        ("tpl_ptr_s046", "車庫証明の保管場所：{A}"),
        ("tpl_ptr_s047", "問い合わせフォーム入力欄【住所】：{A}"),
        ("tpl_ptr_s048", "備考欄メモ：相手方所在地は{A}と記載あり。"),
        ("tpl_ptr_s049", "商品保証書の記載住所：{A}"),
        ("tpl_ptr_s050", "来週の納品予定先は{A}に決定いたしました。"),
        ("tpl_ptr_s051", "患者登録情報より、居住地は{A}と記録されています。"),
        ("tpl_ptr_s052", "不動産登記情報：対象地番は{A}です。"),
        ("tpl_ptr_s053", "配送伝票控え\nお届け先：{A}\n区分：元払い"),
        ("tpl_ptr_s054", "新規取引先マスタ登録：{A}（請求書原本郵送先）"),
        ("tpl_ptr_s055", "イベント案内状の送付先を{A}へ更新しました。"),
        ("tpl_ptr_s056", "出張時の宿泊先予定：{A}"),
        ("tpl_ptr_s057", "会員証再発行分の郵送住所は{A}で受付済みです。"),
        ("tpl_ptr_s058", "住民異動届による新しい居住地：{A}"),
        ("tpl_ptr_s059", "見積書記載の現場住所：{A}"),
        ("tpl_ptr_s060", "社内便の宛先メモ：外注先拠点 {A}"),
        ("tpl_ptr_s061", "採用応募者の履歴書より、連絡先住所は{A}となっています。"),
        ("tpl_ptr_s062", "クレジットカード利用明細の送付先住所：{A}"),
        ("tpl_ptr_s063", "点検作業報告書：実施場所は{A}にて完了。"),
        ("tpl_ptr_s064", "お客様のお申し出住所：{A}（再配達依頼）"),
        ("tpl_ptr_s065", "【契約物件】{A}\n入居開始日は翌月1日です。"),
        ("tpl_ptr_s066", "賃貸借契約書 借主住所：{A}"),
        ("tpl_ptr_s067", "電話窓口ヒアリング記録：相手方住所は{A}とのこと。"),
        ("tpl_ptr_s068", "ギフト発送用アドレス帳登録：{A}"),
        ("tpl_ptr_s069", "株主総会招集通知の送付先：{A}"),
        ("tpl_ptr_s070", "水道・ガス開栓手続先住所：{A}"),
        ("tpl_ptr_s071", "通信教育教材の毎月お届け先：{A}"),
        ("tpl_ptr_s072", "防災備蓄品の保管倉庫は{A}に設置しています。"),
        ("tpl_ptr_s073", "自動車保険証券の契約者住所欄：{A}"),
        ("tpl_ptr_s074", "本人確認書類（公共料金領収書）の記載：{A}"),
        ("tpl_ptr_s075", "お荷物の転送先として{A}を指定承りました。"),
        ("tpl_ptr_s076", "自治体窓口提出用の居住地記載：{A}"),
        ("tpl_ptr_s077", "展示会出展社情報・事業所所在地：{A}"),
        ("tpl_ptr_s078", "引取回収先住所：{A}（トラック進入可能）"),
        ("tpl_ptr_s079", "定期購読誌の配達先を「{A}」へ変更します。"),
        ("tpl_ptr_s080", "医療費通知書の郵送先住所：{A}"),
        ("tpl_ptr_s081", "修理受付票：返却先住所は{A}になります。"),
        ("tpl_ptr_s082", "取引契約書末尾 乙の住所：{A}"),
        ("tpl_ptr_s083", "オンライン注文確認メール：配送先 {A}"),
        ("tpl_ptr_s084", "研修会場アクセス情報：{A}（受付は2階）"),
        ("tpl_ptr_s085", "介護保険被保険者証の住所記載：{A}"),
        ("tpl_ptr_s086", "児童手当現況届の連絡先住所は{A}です。"),
        ("tpl_ptr_s087", "機器設置先：{A}\n保守担当へ共有してください。"),
        ("tpl_ptr_s088", "名簿データ転記分：登録所在地 {A}"),
        ("tpl_ptr_s089", "カタログ請求フォームより、送付先={A}"),
        ("tpl_ptr_s090", "引越見積訪問先：{A}（エレベーターあり）"),
        ("tpl_ptr_s091", "ふるさと応援寄附金 書類送付先：{A}"),
        ("tpl_ptr_s092", "銀行届出住所：{A}（本人確認完了）"),
        ("tpl_ptr_s093", "委託作業実施場所：{A} にて現地集合。"),
        ("tpl_ptr_s094", "定期点検通知書 宛先：{A}"),
        ("tpl_ptr_s095", "店舗移転のお知らせ：新店舗は{A}にオープンします。"),
        ("tpl_ptr_s096", "資料一式の郵送ご希望先：{A}"),
        ("tpl_ptr_s097", "携帯電話契約時の申告住所は{A}でした。"),
        ("tpl_ptr_s098", "緊急配送手配：受取場所は{A}です。"),
        ("tpl_ptr_s099", "業務連絡：明日の外回り訪問先は{A}から開始します。"),
        ("tpl_ptr_s100", "申請受付確認メール：ご登録の所在地は{A}です。"),
    ],
    "dev": [
        ("tpl_pdv_s001", "申請書の記載住所：{A}（本人確認書類と一致）"),
        ("tpl_pdv_s002", "新規お届け先として{A}が指定されました。"),
        ("tpl_pdv_s003", "チャットログ抜粋「明日の集合場所は{A}で合っていますか？」"),
        ("tpl_pdv_s004", "届出住所：{A}／確認日：今月15日"),
        ("tpl_pdv_s005", "資料請求の郵送住所は{A}でございます。"),
        ("tpl_pdv_s006", "【配送指示】{A} 玄関前置き配希望"),
        ("tpl_pdv_s007", "取引先所在地メモ：{A}（請求書PDFより）"),
        ("tpl_pdv_s008", "ご自宅の住所を「{A}」として登録完了しました。"),
        ("tpl_pdv_s009", "出張報告：訪問先住所 {A}"),
        ("tpl_pdv_s010", "本人連絡先\n居住地：{A}"),
        ("tpl_pdv_s011", "修理品の返送先住所：{A}宛にお願いいたします。"),
        ("tpl_pdv_s012", "学籍登録情報：保護者住所は{A}です。"),
        ("tpl_pdv_s013", "口座振替依頼書 住所：{A}"),
        ("tpl_pdv_s014", "受付票より：{A}にお住まいの方からのご照会です。"),
        ("tpl_pdv_s015", "委任状に記載された代理人住所：{A}"),
        ("tpl_pdv_s016", "商品交換品の送り先を{A}へ登録いたしました。"),
        ("tpl_pdv_s017", "現場調査メモ：対象建物は{A}に位置しています。"),
        ("tpl_pdv_s018", "年末調整申告書より、現住所は{A}です。"),
        ("tpl_pdv_s019", "配送手配依頼書\n宛先：{A}\n午前指定"),
        ("tpl_pdv_s020", "お客様カード記入内容・住所：{A}"),
        ("tpl_pdv_s021", "返還金通知書の発送先：{A}宛て"),
        ("tpl_pdv_s022", "宅配クリーニングの集配住所は{A}になっています。"),
        ("tpl_pdv_s023", "確認メール抜粋：ご指定会場は{A}となります。"),
        ("tpl_pdv_s024", "住民税決定通知書の送付住所：{A}"),
        ("tpl_pdv_s025", "設備保守拠点の所在地として{A}を記録しました。"),
        ("tpl_pdv_s026", "問い合わせログ：配送先「{A}」の番地確認要。"),
        ("tpl_pdv_s027", "賃貸内見予約：待ち合わせ場所は{A}の1階です。"),
        ("tpl_pdv_s028", "健康診断結果の郵送先：{A}"),
        ("tpl_pdv_s029", "払込票の再発行送付先を{A}に変更しました。"),
        ("tpl_pdv_s030", "法人口座審査資料 営業所住所：{A}"),
    ],
    "sealed-test": [
        ("tpl_pst_s001", "更新手続き用の案内は{A}へ発送済みです。"),
        ("tpl_pst_s002", "顧客連絡メモ：「引っ越し先は{A}に決まりました」"),
        ("tpl_pst_s003", "【受取場所】{A} 宅配ロッカー利用"),
        ("tpl_pst_s004", "登記事項記載の住所は{A}となっております。"),
        ("tpl_pst_s005", "利用申込書 住所記入欄：{A}"),
        ("tpl_pst_s006", "サンプル品発送先：{A}（平日午後着指定）"),
        ("tpl_pst_s007", "提携事務所所在地：{A}"),
        ("tpl_pst_s008", "ご指定の住所「{A}」への手配を承りました。"),
        ("tpl_pst_s009", "点検作業場所：{A} 担当者までご連絡ください。"),
        ("tpl_pst_s010", "申込情報詳細\n現住所：{A}"),
        ("tpl_pst_s011", "契約者本人住所：{A}にて受理いたしました。"),
        ("tpl_pst_s012", "帰省先連絡住所：{A}"),
        ("tpl_pst_s013", "定期点検案内ハガキ宛先：{A}"),
        ("tpl_pst_s014", "サポート窓口記録：ユーザー居住地は{A}と申告あり。"),
        ("tpl_pst_s015", "源泉徴収票の郵送先住所：{A}"),
        ("tpl_pst_s016", "レンタル機材の返却前保管場所は{A}です。"),
        ("tpl_pst_s017", "購入者情報チェック：{A}に配送予定です。"),
        ("tpl_pst_s018", "保険金請求書の受取人住所欄：{A}"),
        ("tpl_pst_s019", "社宅入居予定住所：{A}（鍵渡し来週金曜）"),
        ("tpl_pst_s020", "見積依頼フォーム\n施工先住所：{A}"),
        ("tpl_pst_s021", "会報誌の最新送付先を{A}に切り替えました。"),
        ("tpl_pst_s022", "本人確認書類の裏面記載住所は{A}です。"),
        ("tpl_pst_s023", "ネットスーパー配達先：{A} インターホン故障中"),
        ("tpl_pst_s024", "取引先台帳：代表拠点住所 {A}"),
        ("tpl_pst_s025", "通院記録・患者住所：{A}"),
        ("tpl_pst_s026", "工事完了通知書の送付住所は{A}となります。"),
        ("tpl_pst_s027", "研修資料受取先：「{A}」宛て発送。"),
        ("tpl_pst_s028", "遺失物返還の郵送先：{A}"),
        ("tpl_pst_s029", "新規出店予定地メモ：{A}（来春開業目標）"),
        ("tpl_pst_s030", "登録済みの緊急避難先住所は{A}になっております。"),
    ],
}

# 複数住所テンプレート（train 20種 / dev 8種 / sealed-test 8種 = 計36種）
MULTI_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    "train": [
        ("tpl_ptr_m01", "転居前の住所は{A1}、転居後の新住所は{A2}です。"),
        ("tpl_ptr_m02", "発送元：{A1}\n配送先：{A2}"),
        ("tpl_ptr_m03", "本社所在地は{A1}ですが、実務拠点は{A2}にございます。"),
        ("tpl_ptr_m04", "契約者住所：{A1}／連帯保証人住所：{A2}"),
        ("tpl_ptr_m05", "登録住所「{A1}」から「{A2}」への変更届を受け付けました。"),
        ("tpl_ptr_m06", "住民票上の住所：{A1}、実際の居所：{A2}"),
        ("tpl_ptr_m07", "午前中の訪問先は{A1}、午後の訪問先は{A2}となります。"),
        ("tpl_ptr_m08", "請求書送付先：{A1}、納品場所：{A2}"),
        ("tpl_ptr_m09", "ご本人様住所（{A1}）および緊急連絡先住所（{A2}）を確認しました。"),
        ("tpl_ptr_m10", "旧所在地：{A1}\n新所在地：{A2}"),
        ("tpl_ptr_m11", "搬出元は{A1}で、搬入先は{A2}を予定しています。"),
        ("tpl_ptr_m12", "差出人住所：{A1}／受取人住所：{A2}"),
        ("tpl_ptr_m13", "ご自宅（{A1}）ではなく勤務先（{A2}）への郵送を希望します。"),
        ("tpl_ptr_m14", "申込人住所：{A1}、配偶者居住地：{A2}"),
        ("tpl_ptr_m15", "第1倉庫：{A1}\n第2倉庫：{A2}"),
        ("tpl_ptr_m16", "引取場所：{A1}から再配達先：{A2}へ輸送します。"),
        ("tpl_ptr_m17", "登記簿上の所在地は{A1}、現在の事務所は{A2}です。"),
        ("tpl_ptr_m18", "保護者住所：{A1}（帰省先住所：{A2}）"),
        ("tpl_ptr_m19", "甲の住所：{A1}、乙の住所：{A2}"),
        ("tpl_ptr_m20", "変更前「{A1}」→ 変更後「{A2}」として更新しました。"),
    ],
    "dev": [
        ("tpl_pdv_m01", "引越し元：{A1} → 引越し先：{A2}"),
        ("tpl_pdv_m02", "申込者住所は{A1}で、書類受取先は{A2}です。"),
        ("tpl_pdv_m03", "集荷先（{A1}）からお届け先（{A2}）への便を手配します。"),
        ("tpl_pdv_m04", "現住所：{A1}\n実家住所：{A2}"),
        ("tpl_pdv_m05", "本店所在地は{A1}、支社所在地は{A2}となります。"),
        ("tpl_pdv_m06", "返送元：{A1}／再送先：{A2}"),
        ("tpl_pdv_m07", "登録地（{A1}）と請求先（{A2}）の照合が完了しました。"),
        ("tpl_pdv_m08", "出発地：{A1}、目的地：{A2}のルートで手配します。"),
    ],
    "sealed-test": [
        ("tpl_pst_m01", "従前の住所：{A1}／変更後の住所：{A2}"),
        ("tpl_pst_m02", "差出人（{A1}）より受取人（{A2}）宛に送付されました。"),
        ("tpl_pst_m03", "本店：{A1}\n支社：{A2}"),
        ("tpl_pst_m04", "旧住所は{A1}、新住所は{A2}でお間違いありませんか。"),
        ("tpl_pst_m05", "貸主住所：{A1}、借主住所：{A2}"),
        ("tpl_pst_m06", "積込先（{A1}）から荷下ろし先（{A2}）へ移動します。"),
        ("tpl_pst_m07", "本人居住地：{A1}\n連絡先所在地：{A2}"),
        ("tpl_pst_m08", "一次拠点：{A1} および 二次拠点：{A2} を巡回します。"),
    ],
}

# ハードネガティブ no_address テンプレート（支店名・料理・駅・型番等の紛らわしい文を含む、分割非重複）
NO_ADDR_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    "train": [
        ("tpl_ptr_n01", "本日の昼食は北海道産じゃがいもを使ったカレーと札幌ラーメン風スープです。"),
        ("tpl_ptr_n02", "明日の会議には株式会社サクラ企画の営業部から3名が参加する予定です。"),
        ("tpl_ptr_n03", "東京駅と大阪駅を結ぶ新幹線の座席予約をオンラインで完了しました。"),
        ("tpl_ptr_n04", "テスト用の数字列 123-4567 および照合番号 9876-5432 を入力してください。"),
        ("tpl_ptr_n05", "京都風抹茶スイーツと名古屋めしフェアを地下1階の特設会場で開催しています。"),
        ("tpl_ptr_n06", "システムログ：エラーコード E-404 が午前10時15分に発生しました。"),
        ("tpl_ptr_n07", "横浜中華街スタイルの肉まんと仙台牛タン弁当を通販で取り寄せました。"),
        ("tpl_ptr_n08", "お支払い金額は合計 24,800円（税込）で、振込期限は今月末です。"),
        ("tpl_ptr_n09", "博多うどんや広島お好み焼きなどのご当地グルメ特集がテレビで放送されました。"),
        ("tpl_ptr_n10", "お客様サポート窓口の受付時間は平日9時から17時までとなっております。"),
        ("tpl_ptr_n11", "渋谷スクランブル風のアート展示が美術館の3階ギャラリーで公開中です。"),
        ("tpl_ptr_n12", "担当者名：佐藤 健太、内線番号：4512、部署：情報システム推進室。"),
        ("tpl_ptr_n13", "新しいソフトウェアのバージョン 2.4.1 では検索速度が大幅に改善されました。"),
        ("tpl_ptr_n14", "神戸牛ステーキのコース料理を予約サイトから3名分で申し込みました。"),
        ("tpl_ptr_n15", "注文番号 #A-88219 の出荷準備が完了し、間もなく追跡番号が発行されます。"),
        ("tpl_ptr_n16", "沖縄黒糖を使った菓子折りをお土産として取引先の皆様にお渡ししました。"),
        ("tpl_ptr_n17", "アンケートへのご協力ありがとうございます。回答内容は統計処理のみに利用します。"),
        ("tpl_ptr_n18", "商品型番：XYZ-102-304 の在庫状況は現在「在庫あり」となっています。"),
        ("tpl_ptr_n19", "信州そばと讃岐うどんの食べ比べセットが今月の売れ筋ランキング1位です。"),
        ("tpl_ptr_n20", "ご不明点がある場合はFAQページをご確認のうえ、フォームよりご連絡ください。"),
        ("tpl_ptr_n21", "振込先銀行：あおば銀行（0015）桜台支店（203）普通預金口座への入金をお願いします。"),
        ("tpl_ptr_n22", "みらい信用金庫の本町出張所（104）にて口座照会の手続きが完了しました。"),
        ("tpl_ptr_n23", "明日の新幹線は名古屋駅を14時20分に出発し、東京駅へ16時頃到着します。"),
        ("tpl_ptr_n24", "在庫管理コード 01-23-456 の棚卸し作業を第2センター内で実施中です。"),
        ("tpl_ptr_n25", "株式会社ひかり不動産の総務担当より、契約書雛形の改定通知が届きました。"),
        ("tpl_ptr_n26", "富士急風のアトラクション特集や日光カステラのお土産が話題を集めています。"),
        ("tpl_ptr_n27", "ネットワーク設定値 192-168-1-10 のポート番号 8080 を開放して検証しました。"),
        ("tpl_ptr_n28", "宇都宮餃子と浜松餃子を食べ比べる社内懇親会を来週金曜日に開催します。"),
        ("tpl_ptr_n29", "さくら銀行 旭川支店の窓口受付番号は 0421 番でお待ちしております。"),
        ("tpl_ptr_n30", "受付番号：R-9920、件名：パスワードリセット申請、処理状態：完了。"),
        ("tpl_ptr_n31", "青森ふじりんごと長野シャインマスカットの詰め合わせが社内で好評でした。"),
        ("tpl_ptr_n32", "プロジェクト資料 3章2節1項（ページ 12-18）の誤記を修正してください。"),
        ("tpl_ptr_n33", "福岡とんこつラーメンの特製スープが限定500食でオンライン販売されています。"),
        ("tpl_ptr_n34", "株式会社つばさ運輸の配送ドライバーへ午前中の作業予定を連絡しました。"),
        ("tpl_ptr_n35", "システム更新時刻は 2026年11月20日 02:00〜04:00（約2時間）を予定しています。"),
        ("tpl_ptr_n36", "伊勢海老風味のスナックと明石焼き風おつまみを売店で購入しました。"),
    ],
    "dev": [
        ("tpl_pdv_n01", "長崎ちゃんぽんと宮崎チキン南蛮のランチセットが今週の日替わりメニューです。"),
        ("tpl_pdv_n02", "品番 B-12-345 の交換部品を倉庫の2番棚から取り出して梱包しました。"),
        ("tpl_pdv_n03", "新潟産コシヒカリと青森りんごジュースの詰め合わせがギフトに人気です。"),
        ("tpl_pdv_n04", "プロジェクト識別子 01101-TEST のビルド検証が全件正常終了しました。"),
        ("tpl_pdv_n05", "株式会社みらい商事の経理担当より請求データ修正の連絡が入りました。"),
        ("tpl_pdv_n06", "金沢金箔入りの和菓子と宇治茶のセットを来客用のお茶菓子に用意しました。"),
        ("tpl_pdv_n07", "東洋銀行の緑ヶ丘支店（412）にて普通預金の口座名義変更を受理しました。"),
        ("tpl_pdv_n08", "品川駅構内の書店でビジネス書を2冊購入し、領収書を発行してもらいました。"),
        ("tpl_pdv_n09", "ドキュメント番号 DOC-2026-08-19 の承認ワークフローが最終段階です。"),
        ("tpl_pdv_n10", "水戸納豆と喜多方ラーメンの物産展カタログを社内掲示板に公開しました。"),
        ("tpl_pdv_n11", "わかば信用組合の本郷出張所におけるシステム点検は予定通り終了しました。"),
        ("tpl_pdv_n12", "見積金額 158,400円、納期：ご注文確定から10営業日以内となります。"),
    ],
    "sealed-test": [
        ("tpl_pst_n01", "秋田きりたんぽ鍋と盛岡冷麺を冬の定番商品としてラインナップに加えました。"),
        ("tpl_pst_n02", "検証用コード 13113-9999 を入力してテスト環境の接続状態を確認します。"),
        ("tpl_pst_n03", "高知かつおのたたきと徳島すだちサイダーが物産展で好評を博しています。"),
        ("tpl_pst_n04", "有限会社あおば製作所の設計チームが来週の定例レビューに同席します。"),
        ("tpl_pst_n05", "浜松うなぎパイ風のお菓子を休憩室のテーブルに置いておきました。"),
        ("tpl_pst_n06", "システム更新メンテナンスは日曜日深夜2時から4時までの間に実施されます。"),
        ("tpl_pst_n07", "中央信託銀行の春日井支店（518）宛てに定例の振込データを送信しました。"),
        ("tpl_pst_n08", "上野公園口の改札前で集合し、博物館の特別展を見学する予定です。"),
        ("tpl_pst_n09", "契約ID K-7741-02 の更新案内メールが送信エラーとなり再送しました。"),
        ("tpl_pst_n10", "熊本馬刺し風オードブルと大分とり天の試食イベントが盛況でした。"),
        ("tpl_pst_n11", "あさひ信用金庫の大手町出張所で発行された取引明細を確認しています。"),
        ("tpl_pst_n12", "サーバー負荷テストの結果、最大スループットは毎秒 1,250 リクエストでした。"),
    ],
}


class FastMinHashLSH:
    """文字 5-gram MinHash LSH（num_perm=64, bands=16, rows=4）。キャッシュによる高速版。"""

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
        self.split_tags: List[str] = []
        self._crc_cache: Dict[str, int] = {}

    def _shingles(self, text: str) -> Set[int]:
        s = re.sub(r"\s+", "", text)
        cache = self._crc_cache
        if len(s) < 5:
            if not s:
                return set()
            v = cache.get(s)
            if v is None:
                v = zlib.crc32(s.encode("utf-8"))
                cache[s] = v
            return {v}
        res: Set[int] = set()
        for i in range(len(s) - 4):
            g = s[i : i + 5]
            v = cache.get(g)
            if v is None:
                v = zlib.crc32(g.encode("utf-8"))
                cache[g] = v
            res.add(v)
        return res

    def signature(self, text: str) -> Tuple[Set[int], List[int]]:
        sh = self._shingles(text)
        if not sh:
            return sh, [0] * self.num_perm
        sig: List[int] = []
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
        self.split_tags.append(split_tag)
        for bi in range(self.bands):
            st = bi * self.rows
            key = tuple(sig[st : st + self.rows])
            self.buckets[bi][key].append(idx)
        return idx

    def query_near_dup(
        self, text: str, exclude_split: Optional[str] = None, threshold: float = 0.72
    ) -> Optional[str]:
        sh, sig = self.signature(text)
        if not sh:
            return None
        cand_idxs: Set[int] = set()
        for bi in range(self.bands):
            st = bi * self.rows
            key = tuple(sig[st : st + self.rows])
            for c_idx in self.buckets[bi].get(key, ()):
                if exclude_split is None or self.split_tags[c_idx] != exclude_split:
                    cand_idxs.add(c_idx)
        for c_idx in cand_idxs:
            other = self.shingle_sets[c_idx]
            union = len(sh | other)
            if union and (len(sh & other) / union) >= threshold:
                return self.split_tags[c_idx]
        return None


def format_postal(postal7: str, rng: random.Random) -> str:
    p = f"{postal7[:3]}-{postal7[3:]}"
    return f"〒{p}" if rng.random() < 0.85 else p


def build_address_parts(
    addr_info: dict, rng: random.Random, add_postal: bool
) -> List[Tuple[Optional[str], str]]:
    segs: List[Tuple[Optional[str], str]] = []
    if add_postal and addr_info["postal"]:
        segs.append((None, format_postal(addr_info["postal"], rng) + rng.choice([" ", "　", ""])))
    segs.extend(addr_info["segments"])
    return segs


def extract_nemotron_windows_from_doc(
    doc: dict, max_chars: int = 230
) -> List[Tuple[str, int, int, List[Tuple[int, int]]]]:
    """1文書から安全な単一ADDRESS窓を抽出する（離れた複数フル住所セクションにも対応）。"""
    text = doc["text"]
    spans = doc.get("spans", [])
    asps = sorted(
        [s for s in spans if s["label"] == "ADDRESS"],
        key=lambda x: (int(x["start"]), int(x["end"])),
    )
    if not asps:
        return []

    results: List[Tuple[str, int, int, List[Tuple[int, int]]]] = []
    used_ranges: List[Tuple[int, int]] = []

    for ai, a0 in enumerate(asps):
        ast, aed = int(a0["start"]), int(a0["end"])
        pre_ctx = text[max(0, ast - 10) : ast]
        if any(k in pre_ctx for k in ("都道府県", "市区町村", "町域", "番地")):
            continue
        if len(a0["text"]) < 8 or not re.search(
            r"[0-9０-９一二三四五六七八九十丁目番号]", a0["text"]
        ):
            continue

        # 前後の別 ADDRESS スパンと重ならない範囲を算出
        prev_addr_end = int(asps[ai - 1]["end"]) + 2 if ai > 0 else 0
        next_addr_start = int(asps[ai + 1]["start"]) - 2 if ai + 1 < len(asps) else len(text)

        left_min = max(prev_addr_end, ast - 105)
        right_max = min(next_addr_start, aed + 105)

        dbl_l = text.rfind("\n\n", left_min, ast)
        if dbl_l != -1:
            win_s = dbl_l + 2
        else:
            nl_first = text.find("\n", left_min, ast - 12 if ast - 12 > left_min else ast)
            dot_first = text.find("。", left_min, ast)
            cands_l = [p + 1 for p in (nl_first, dot_first) if p != -1]
            win_s = min(cands_l) if cands_l else left_min

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

        if win_s < prev_addr_end or win_e > next_addr_start:
            continue

        raw_slice = text[win_s:win_e]
        win_text = raw_slice.strip()
        if len(win_text) < (aed - ast) + 12:
            continue
        lead_ws = len(raw_slice) - len(raw_slice.lstrip())
        win_s += lead_ws
        win_e = win_s + len(win_text)

        # 他窓と重ならないことを確認
        if any(not (win_e <= us or ue <= win_s) for us, ue in used_ranges):
            continue

        rel_ast = ast - win_s
        rel_aed = aed - win_s
        if not (0 <= rel_ast < rel_aed <= len(win_text)):
            continue

        outside_text = win_text[:rel_ast] + "\n" + win_text[rel_aed:]
        if _PREF_CITY_GUARD_RE.search(outside_text):
            continue

        pc_rels: List[Tuple[int, int]] = []
        for s in spans:
            if s["label"] == "POSTAL_CODE":
                ps, pe = int(s["start"]), int(s["end"])
                if ps >= win_s and pe <= win_e and (pe <= ast or ps >= aed):
                    rps, rpe = ps - win_s, pe - win_s
                    if rps >= 1 and win_text[rps - 1] == "〒":
                        rps -= 1
                    pc_rels.append((rps, rpe))
        masked = list(win_text)
        for rps, rpe in pc_rels:
            for idx in range(rps, rpe):
                masked[idx] = "X"
        masked_outside = "".join(masked[:rel_ast]) + "\n" + "".join(masked[rel_aed:])
        if _POSTAL_PAT_RE.search(masked_outside):
            continue

        used_ranges.append((win_s, win_e))
        results.append((win_text, rel_ast, rel_aed, sorted(pc_rels)))
        if len(results) >= 2:
            break

    return results


def render_nemotron_sample(
    win_text: str,
    rel_ast: int,
    rel_aed: int,
    pc_rels: List[Tuple[int, int]],
    addr_info: dict,
    rng: random.Random,
) -> Optional[Tuple[str, List[dict], List[dict]]]:
    repls: List[Tuple[int, int, List[Tuple[Optional[str], str]]]] = []
    addr_segs = list(addr_info["segments"])
    for rps, rpe in pc_rels:
        if addr_info["expected_status"] == "RESOLVED" and addr_info["postal"]:
            repls.append((rps, rpe, [(None, format_postal(addr_info["postal"], rng))]))
        else:
            repls.append((rps, rpe, [(None, "未記載")]))
    repls.append((rel_ast, rel_aed, addr_segs))
    repls.sort(key=lambda x: x[0])

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

    out_parts.append(win_text[cursor:])
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


def _sample_context_wrapper(rng: random.Random, split: str) -> Tuple[str, str]:
    """テンプレート文の文面多様性を高めるための安全・分割別前後修飾。"""
    r = rng.random()
    if r < 0.45:
        return "", ""
    if r < 0.72:
        staff = rng.choice(STAFF_NAMES[split])
        dept = rng.choice(DEPTS[split])
        return f"[{dept} / 担当:{staff}] ", ""
    if r < 0.90:
        m = rng.randint(1, 12)
        d = rng.randint(1, 28)
        return f"({m}月{d}日受付) ", ""
    ref = f"{rng.choice(['A','B','C','K','M'])}{rng.randint(100,999)}"
    return "", f"（受付ID:{ref}）"


def render_template_single(
    tpl_str: str,
    addr_info: dict,
    rng: random.Random,
    add_postal: bool,
    split: str,
) -> Tuple[str, List[dict], List[dict]]:
    head_wrap, tail_wrap = _sample_context_wrapper(rng, split)
    prefix, suffix = tpl_str.split("{A}", 1)
    prefix = head_wrap + prefix
    suffix = suffix + tail_wrap

    segs = build_address_parts(addr_info, rng, add_postal=add_postal)
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
    split: str,
) -> Tuple[str, List[dict], List[dict]]:
    head_wrap, tail_wrap = _sample_context_wrapper(rng, split)
    p1 = tpl_str.index("{A1}")
    p2 = tpl_str.index("{A2}")
    assert p1 < p2
    chunks = [head_wrap + tpl_str[:p1], tpl_str[p1 + 4 : p2], tpl_str[p2 + 4 :] + tail_wrap]

    parts: List[str] = [chunks[0]]
    pos = len(chunks[0])
    comp_spans: List[dict] = []
    gold_addrs: List[dict] = []

    for ainfo, add_p, tail in [
        (addr1, add_postal1, chunks[1]),
        (addr2, add_postal2, chunks[2]),
    ]:
        segs = build_address_parts(ainfo, rng, add_postal=add_p)
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
                "text": None,
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
                for win_text, rel_ast, rel_aed, pc_rels in extract_nemotron_windows_from_doc(doc):
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
    text = row["text"]
    comps = row["components"]
    addrs = row["addresses"]

    prev_e = -1
    for c in comps:
        st, ed = c["start"], c["end"]
        assert 0 <= st < ed <= len(text), f"範囲外: {c}"
        assert st >= prev_e, f"構成要素重複: {comps}"
        assert text[st:ed] == c["text"], f"テキスト不一致: {c['text']!r} != {text[st:ed]!r}"
        prev_e = ed

    grouped = ou.group_components_to_addresses(text, comps)
    assert len(grouped) == len(addrs), (
        f"グルーピング件数不一致 ({row['id']}): {len(grouped)} != {len(addrs)}"
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
    splits_rows: Dict[str, List[dict]], lsh_eval: FastMinHashLSH
) -> Dict[str, dict]:
    """全3ペアの分割間リーク検査を実施し、全項目0を assert 検証する。"""
    pairs = [("train", "dev"), ("train", "sealed-test"), ("dev", "sealed-test")]
    meta: Dict[str, dict] = {}
    for sp, rows in splits_rows.items():
        tpls = {r["template_id"] for r in rows if r["source"] == "template"}
        nemos = {r["nemotron_doc_hash"] for r in rows if r["nemotron_doc_hash"]}
        texts = {r["text"] for r in rows}
        addrs = {a["text"] for r in rows for a in r["addresses"]}
        meta[sp] = {"tpls": tpls, "nemos": nemos, "texts": texts, "addrs": addrs}

    near_dup_counts: Dict[Tuple[str, str], int] = collections.Counter()

    # 1) dev vs sealed-test (lsh_eval 内に両方登録済み)
    for r in splits_rows["dev"]:
        hit = lsh_eval.query_near_dup(r["text"], exclude_split="dev", threshold=0.80)
        if hit == "sealed-test":
            near_dup_counts[("dev", "sealed-test")] += 1

    # 2) train 100,000件 vs lsh_eval (dev / sealed-test)
    for r in splits_rows["train"]:
        hit = lsh_eval.query_near_dup(r["text"], exclude_split="train", threshold=0.80)
        if hit in ("dev", "sealed-test"):
            near_dup_counts[("train", hit)] += 1

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
    ap.add_argument("--train-size", type=int, default=100000)
    ap.add_argument("--dev-size", type=int, default=5000)
    ap.add_argument("--test-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1042)
    args = ap.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    synth = AddressSynthesizer()

    nemo_pools = {
        "dev": load_nemotron_pools(NEMOTRON_DIR / "nemotron-dev.jsonl", seed=args.seed + 1),
        "sealed-test": load_nemotron_pools(
            NEMOTRON_DIR / "nemotron-test.jsonl", seed=args.seed + 2
        ),
        "train": load_nemotron_pools(
            NEMOTRON_DIR / "nemotron-train.jsonl", seed=args.seed + 3
        ),
    }

    addr_to_split: Dict[str, str] = {}
    used_texts_all: Set[str] = set()
    # dev と sealed-test を登録する評価分割 LSH インデックス（10,000件）
    lsh_eval = FastMinHashLSH(num_perm=64, bands=16)

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

        # Nemotron 住所ありプールは利用可能上限に合わせて比率を自然に配分
        # dev/sealed-test は約 38%、train は約 20% 相当を上限目安にして偏りなく分散配置
        nemo_addr_prob = 0.38 if split_name != "train" else 0.22

        tasks: List[str] = []
        for cat, cnt in cat_counts:
            tasks.extend([cat] * cnt)
        rng.shuffle(tasks)

        rows: List[dict] = []

        def sample_unique_addr(cat_req: str) -> dict:
            for _ in range(200):
                ainfo = synth.generate_one(rng, cat_req, split_name)
                surf = ainfo["address_text"]
                prev_sp = addr_to_split.get(surf)
                if prev_sp is not None and prev_sp != split_name:
                    continue
                return ainfo
            raise RuntimeError(f"非重複住所の確保失敗: {cat_req} / {split_name}")

        for idx_row, cat in enumerate(tasks):
            built_row: Optional[dict] = None
            want_nemo = rng.random() < (0.70 if cat == "no_address" else nemo_addr_prob)

            for attempt in range(45):
                use_nemo = want_nemo and (attempt < 8)
                if cat == "no_address":
                    if use_nemo and nemo_noaddr_idx < len(nemo_noaddr_pool):
                        doc_hash, ntext = nemo_noaddr_pool[nemo_noaddr_idx]
                        nemo_noaddr_idx += 1
                        if ntext in used_texts_all:
                            continue
                        if lsh_eval.query_near_dup(ntext, exclude_split=split_name, threshold=0.72):
                            continue
                        built_row = {
                            "id": f"{split_name}-{idx_row:06d}",
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
                        head_w, tail_w = _sample_context_wrapper(rng, split_name)
                        ntext = head_w + tpl_str + tail_w
                        if ntext in used_texts_all and attempt < 38:
                            continue
                        if lsh_eval.query_near_dup(ntext, exclude_split=split_name, threshold=0.72):
                            continue
                        built_row = {
                            "id": f"{split_name}-{idx_row:06d}",
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
                    if lsh_eval.query_near_dup(text, exclude_split=split_name, threshold=0.72):
                        continue
                    built_row = {
                        "id": f"{split_name}-{idx_row:06d}",
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
                    is_multi = rng.random() < 0.10
                    if is_multi:
                        tpl_id, tpl_str = rng.choice(multi_tpls)
                        a1 = sample_unique_addr(cat)
                        cat2 = (
                            cat
                            if rng.random() < 0.45
                            else rng.choice(
                                [
                                    "standard",
                                    "omit_pref",
                                    "same_name_resolved",
                                    "variant",
                                    "omit_block",
                                    "unresolvable_ambiguous",
                                ]
                            )
                        )
                        a2 = sample_unique_addr(cat2)
                        add_p1 = (a1["expected_status"] == "RESOLVED") and (rng.random() < 0.18)
                        add_p2 = (a2["expected_status"] == "RESOLVED") and (rng.random() < 0.18)
                        text, comps, addrs = render_template_multi(
                            tpl_str, a1, a2, rng, add_p1, add_p2, split_name
                        )
                    else:
                        tpl_id, tpl_str = rng.choice(single_tpls)
                        a1 = sample_unique_addr(cat)
                        add_p = (a1["expected_status"] == "RESOLVED") and (rng.random() < 0.18)
                        text, comps, addrs = render_template_single(
                            tpl_str, a1, rng, add_p, split_name
                        )

                    if text in used_texts_all and attempt < 38:
                        continue
                    if lsh_eval.query_near_dup(text, exclude_split=split_name, threshold=0.72):
                        continue
                    built_row = {
                        "id": f"{split_name}-{idx_row:06d}",
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
            if split_name in ("dev", "sealed-test"):
                lsh_eval.add_doc(built_row["text"], split_name)
            rows.append(built_row)

        splits_rows[split_name] = rows

    leakage_report = check_leakage(splits_rows, lsh_eval)

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
