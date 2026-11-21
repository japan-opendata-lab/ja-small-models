#!/usr/bin/env python3
"""第12号 ja-sentiment-v1 本番データ追加用 シナリオ・レシピ生成モジュール。

設計方針:
1. パイロット (scenario_catalog.py) の全372サブトピックを各プールにそのまま継承した上で、
   train/dev/sealed_test それぞれ専用の新規サブトピック280種を追加し、計652種へ拡充する。
   (human_eval 用44種は完全不変・重複ゼロを強制検証)
2. 第1工程監査で判明した domain_mismatch 対策として、product_review / hotel_dining /
   survey_workplace の着眼点 (aspect) をサブトピック種別に応じて整合フィルタする。
3. 中立 (感情の薄い事実文 U1・均衡両面評価 U2 等) の厚みを維持・強化する。
"""
import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario_catalog as sc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# 各ドメインの既存プール末尾に追加する新規サブトピック（プール排他）
EXTRA_SUBTOPICS: Dict[str, Dict[str, List[str]]] = {
    "product_review": {
        "train": [
            # +40 -> 計90
            "卓上USB扇風機", "電動ネイルケア", "温湿度計デジタル", "センサー付きゴミ箱", "LEDシーリングライト",
            "タブレットスタンド", "USBハブアダプタ", "ノイズキャンセリング耳栓", "スマートプラグ", "電動泡立て器",
            "オーブントースター", "電気圧力鍋", "保冷ランチバッグ", "防水スマホポーチ", "折りたたみ踏み台",
            "布団圧縮袋セット", "ステンレス水切りかご", "ヨガマット", "ダンベルセット", "腹筋ローラー",
            "ボディソープ", "シャンプー・トリートメント", "ヘアオイル", "マウスウォッシュ", "日焼け止めスプレー",
            "シートマスク", "無糖炭酸水ケース", "アーモンドミルク", "グラノーラ", "インスタント味噌汁",
            "パスタソースセット", "発芽玄米パック", "ハーブティー詰め合わせ", "防災用保存食セット", "携帯トイレセット",
            "使い捨てカイロ箱買い", "不織布マスク大容量", "メガネ曇り止めクロス", "靴用消臭スプレー", "ペット用消臭シート",
        ],
        "dev": [
            # +7 -> 計14
            "衣類乾燥除湿機", "充電式湯たんぽ", "シリコン調理スプーン", "トラベル圧縮ポーチ",
            "炭酸入浴タブレット", "低糖質プロテインバー", "猫用キャリーバッグ",
        ],
        "sealed_test": [
            # +7 -> 計14
            "サーキュレーター", "磁気カードケース", "アルミ製洗濯ハンガー", "ガラス製計量カップ",
            "クレンジングオイル", "ルイボスティーバッグ", "犬用リード・ハーネス",
        ],
    },
    "hotel_dining": {
        "train": [
            # +35 -> 計75
            "温泉大浴場付きビジネス宿", "駅直結のシティホテル", "グランピング施設", "海沿いの民宿", "素泊まりゲストハウス",
            "もつ鍋居酒屋", "鉄板焼きダイニング", "沖縄料理店", "ベトナム料理店", "インドカレー店",
            "ネパール料理店", "創作和食ダイニング", "鉄板お好み焼き店", "焼きそば専門店", "つけ麺専門店",
            "担々麺専門店", "油そば店", "牛タン定食専門店", "天丼チェーン店", "親子丼・鶏料理店",
            "生パスタ専門店", "ファミレスの日替わりランチ", "深夜営業の牛丼チェーン", "モーニングカフェチェーン", "タルト・フルーツパーラー",
            "クレープ専門店", "ベーグル専門店", "高級食パン専門店", "量り売り惣菜デリ", "百貨店の弁当売場",
            "駅前のおにぎり専門店", "テイクアウト専門焼き鳥", "デリバリー寿司", "ネット予約のクリスマスケーキ", "冷凍グルメ定期便",
        ],
        "dev": [
            # +6 -> 計12
            "サウナ特化型カプセルホテル", "駅前の大衆食堂", "釜飯専門店", "韓国チキン専門店",
            "テイクアウト専門サンドイッチ店", "ジェラートお取り寄せ便",
        ],
        "sealed_test": [
            # +6 -> 計12
            "高原の貸別荘コテージ", "手羽先居酒屋", "沖縄そば専門店", "ガレット・クレープ店",
            "デパ地下のサラダ専門店", "デリバリーインドカレー",
        ],
    },
    "sns_short": {
        "train": [
            # +35 -> 計75
            "深夜のコンビニ立ち読み中止", "電車の遅延と振替輸送", "土曜日の朝の掃除", "ベランダの洗濯物干し", "冷凍食品のまとめ買い",
            "ドラッグストアのクーポン利用", "銀行ATMの手数料確認", "ネット通販のタイムセール", "中古ショップの掘り出し物", "フリマアプリの値下げ交渉",
            "パソコンのバックアップ作業", "スマホの写真整理", "スマートウォッチの歩数達成", "朝カフェでの資格勉強", "昼休みの短い散歩",
            "夕方のスーパーの見切り品", "帰宅後のストレッチ習慣", "週末の家具店巡り", "植物の植え替え作業", "キッチンの換気扇掃除",
            "エアコンのフィルター洗浄", "車のタイヤ交換予約", "ガソリンスタンドのセルフ給油", "コインパーキングの精算", "高速道路の渋滞回避",
            "空港保安検査場の待ち列", "新幹線車内のWi-Fi利用", "映画館のポップコーン購入", "書店での新刊探し", "古本市での買い込み",
            "季節の変わり目の衣替え", "冬の加湿器の水換え", "雨の日の室内干し", "休日の昼寝とカフェイン", "月曜朝のメール確認",
        ],
        "dev": [
            # +6 -> 計12
            "金曜夜のスーパー半額タイム", "水道検針票の確認", "スマホ決済の還元キャンペーン",
            "週末のホームパーティ準備", "電車内の空調の強さ", "休日の図書館予約本受取",
        ],
        "sealed_test": [
            # +6 -> 計12
            "朝一番のコインランドリー乾燥", "自転車のパンク修理待ち", "オンライン会議前の音声テスト",
            "百貨店の催事場巡り", "雨上がりの夕焼け散歩", "年度末の書類シュレッダー作業",
        ],
    },
    "support_feedback": {
        "train": [
            # +33 -> 計75
            "宅配ボックス誤配の問い合わせ", "クレジットカード不正利用疑い連絡", "モバイルSuicaの機種変更サポート", "高速道路ETCカード再発行窓口", "自動車ディーラーのリコール対応",
            "車検予約サイトの変更手続き", "住宅設備メーカーの部品取り寄せ", "給湯器エラーの修理依頼窓口", "インターネット光回線の解約窓口", "格安SIMのMNP転出サポート",
            "オンライン英会話の講師変更相談", "通信教材の退会連絡窓口", "音楽サブスクのファミリープラン案内", "動画配信の二重課金確認窓口", "電子書籍ストアの購入キャンセル相談",
            "旅行ツアーの台風振替窓口", "高速バス予約の変更・払い戻し", "テーマパーク公式チケット相談", "美術展オンライン券の日時変更", "ホテル予約サイトの領収書発行",
            "ふるさと納税返礼品の配送遅延照会", "ネットショップの領収書宛名変更", "ギフト配送ののし指定ミス対応", "定期配送の住所変更手続き", "家電量販店の配送設置日時調整",
            "家具通販の組立オプション問い合わせ", "眼鏡通販の度数交換サポート", "コンタクト定期便の度数変更", "ペットフード定期便のスキップ", "法人クラウド請求書の支払方法変更",
            "会計ソフトの年末調整サポート", "確定申告ソフトのチャット質問", "自治体粗大ごみ受付センター",
        ],
        "dev": [
            # +6 -> 計12
            "レンタサイクルの返却エラー対応", "ネット銀行のワンタイムパス再設定", "航空マイレージ事後登録の問い合わせ",
            "百貨店オンラインのギフト返品相談", "家具修理の出張見積もり窓口", "カーシェア車両の汚れ報告対応",
        ],
        "sealed_test": [
            # +6 -> 計12
            "宅配便の時間指定変更チャット", "ETCマイレージ登録の問い合わせ", "電子マネー残高移行のサポート",
            "通販サイトのレビュー投稿不具合窓口", "浄水器の取付部品追加発送対応", "スポーツクラブの休会申請確認",
        ],
    },
    "app_review": {
        "train": [
            # +33 -> 計75
            "ポイ活歩数連動アプリ", "バーコード家計簿アプリ", "電子領収書スキャンアプリ", "レシピ買い物リストアプリ", "冷蔵庫在庫管理アプリ",
            "服薬リマインダーアプリ", "生理日・体調管理アプリ", "育児記録アプリ", "ペット散歩記録アプリ", "筋トレインターバルタイマー",
            "ポモドーロ集中タイマー", "ホワイトノイズ睡眠アプリ", "ラジオ録音・聴取アプリ", "FMラジオ配信アプリ", "オーディオブックアプリ",
            "英単語フラッシュカードアプリ", "TOEICリスニング対策アプリ", "漢字・語彙クイズアプリ", "ニュース速報プッシュアプリ", "株価チャート監視アプリ",
            "暗号資産ウォレットアプリ", "クレカ明細管理アプリ", "サブスク管理アプリ", "QRクーポン集約アプリ", "スーパー特売チラシアプリ",
            "ネットスーパー注文アプリ", "ドラッグストア公式アプリ", "家電量販店会員証アプリ", "鉄道運行遅延情報アプリ", "バス接近情報アプリ",
            "シェアサイクルマップアプリ", "観光スタンプラリーアプリ", "ゴルフスコア記録アプリ",
        ],
        "dev": [
            # +6 -> 計12
            "防災ハザードマップアプリ", "外食クーポン比較アプリ", "音声メモ文字起こしアプリ",
            "自動車燃費記録アプリ", "図書館延滞通知アプリ", "引っ越し見積もり比較Webツール",
        ],
        "sealed_test": [
            # +6 -> 計12
            "電子マネー残高確認アプリ", "植物水やり管理アプリ", "写真アルバム共有アプリ",
            "公共交通IC履歴閲覧アプリ", "オンライン診察予約アプリ", "中古車査定比較Webサービス",
        ],
    },
    "survey_workplace": {
        "train": [
            # +30 -> 計70
            "新設オフィスのフリーアドレス運用", "社内カフェスペースの混雑対策", "社員証ICカード入退室ゲート", "社内稟議ワークフローシステム", "オンライン名刺交換ツール",
            "情報セキュリティeラーニング", "コンプライアンス動画研修", "管理職向け1on1ガイド研修", "資格取得支援制度の申請手順", "時短勤務・育児支援制度",
            "定期健康診断の予約サイト", "ストレスチェック後の面談案内", "町内防災倉庫の点検説明会", "市のオンライン粗大ごみ申込", "区民センターの会議室予約",
            "公共体育館のロッカー・シャワー", "市民プールの券売機・入場手順", "自治体コミュニティサイクル運用", "学校連絡メール配信システム", "PTAオンライン総会の投票フォーム",
            "児童館の事前予約システム", "高齢者向けスマホ相談窓口", "図書館のセルフ貸出機操作", "美術館の事前日時指定予約", "市民講座のオンライン申込",
            "商店街デジタル商品券の使い方", "マンション管理組合のオンライン総会", "シェアサイクルポートの返却ラック", "コワーキング施設のWi-Fiと電源", "資格試験会場の受付導線",
        ],
        "dev": [
            # +6 -> 計12
            "社内FAQチャットボット試験運用", "社員食堂のモバイルオーダー", "市役所のマイナンバー窓口予約",
            "市民ホール駐車場の事前精算機", "子ども向けプログラミング体験会", "マンション宅配ロッカーの暗証案内",
        ],
        "sealed_test": [
            # +6 -> 計12
            "社内ヘルプデスクのチケット画面", "オンライン展示会のブース案内", "公立図書館のレファレンス窓口",
            "地域包括支援センターの相談受付", "駅周辺シェアオフィスの静音ブース", "大学公開講座の資料配布サイト",
        ],
    },
}

# サブトピック別タイプ分類（domain_mismatch 防止用）
PRODUCT_CAT_HINTS = {
    "consumable": [
        "洗剤", "柔軟剤", "入浴", "タオル", "コーヒー豆", "ティーバッグ", "ナッツ", "プロテイン", "だし", "餃子", "カレー",
        "オイル", "はちみつ", "化粧水", "日焼け止め", "クリーム", "リップバーム", "ボディソープ", "シャンプー", "オイル",
        "マウスウォッシュ", "シートマスク", "炭酸水", "アーモンドミルク", "グラノーラ", "味噌汁", "パスタソース", "玄米",
        "ハーブティー", "保存食", "携帯トイレ", "カイロ", "マスク", "クロス", "消臭", "ティッシュ", "タブレット", "ルイボス",
        "プロテインバー",
    ],
    "gear": [
        "低反発枕", "羽毛掛け布団", "遮光カーテン", "マグボトル", "フライパン", "バスマット", "シャワーヘッド", "リュック",
        "キャリーケース", "折りたたみ傘", "シューズ", "レインブーツ", "インナー", "パーカー", "サングラス", "革財布",
        "腕時計ベルト", "ボールペン", "システム手帳", "ノートPCスタンド", "モニターアーム", "キャンプ用チェア", "クーラーボックス",
        "猫用爪とぎ", "セラミック包丁", "保存容器セット", "替バンド", "ネックピロー", "ノンアルコールビール",
        "ランチバッグ", "スマホポーチ", "踏み台", "圧縮袋", "水切りかご", "ヨガマット", "ダンベル", "腹筋ローラー",
        "耳栓", "圧縮ポーチ", "ハンガー", "計量カップ", "キャリーバッグ", "リード", "スタンド",
        "調理スプーン", "カードケース",
    ],
}

HOTEL_CAT_HINTS = {
    "hotel": ["ホテル", "旅館", "宿", "コテージ", "ゲストハウス", "ペンション", "グランピング", "民宿", "貸別荘"],
    "takeout": ["テイクアウト", "宅配", "出前", "デリバリー", "総菜", "惣菜", "デリ", "弁当", "お取り寄せ", "定期便", "クリスマスケーキ", "駅弁"],
}

SURVEY_CAT_HINTS = {
    "event": ["セミナー", "研修", "オンボーディング", "訓練", "指導", "教室", "レッスン", "体験", "クラス", "カンファレンス", "清掃活動", "オープンキャンパス", "面談", "説明会", "総会", "懇親会", "企画", "シンポジウム", "展示会", "講座", "スクール"],
    "system": ["ポータル", "システム", "ツール", "画面", "アプリ", "通知設計", "サイト", "フォーム", "チャート", "予約", "ワークフロー", "eラーニング", "メール配信", "セルフ貸出機", "精算機", "券売機", "チャットボット", "モバイルオーダー", "オンライン粗大ごみ申込", "デジタル商品券"],
}

ASPECTS_BY_DOMAIN_TYPED = {
    "product_review": {
        "electric": [
            "使い勝手と操作性", "価格に対するコスパ", "耐久性・作りのしっかり感", "サイズ感と重量",
            "稼働音・動作の静かさ", "デザインと質感", "バッテリー持ち・省エネ性", "手入れのしやすさ",
            "梱包と初期状態", "説明書の分かりやすさ", "期待とのギャップ", "買い替え前との比較",
            "細かい部品の精度", "収納・持ち運びやすさ",
        ],
        "gear": [
            "使い勝手と操作性", "価格に対するコスパ", "耐久性・縫製/素材のしっかり感", "サイズ感と重量",
            "デザインと質感", "手入れ・洗いやすさ", "梱包と初期状態", "期待とのギャップ",
            "買い替え前との比較", "家族・同僚への勧めやすさ", "細部の仕上げ精度", "収納・持ち運びやすさ",
        ],
        "consumable": [
            "使い心地・香り/味わいのバランス", "価格に対するコスパ", "内容量とパッケージの開けやすさ", "肌や体調への合いやすさ",
            "梱包と配送時の状態", "リピートしやすいかどうか", "成分・原材料への安心感", "使い切りやすさ・保存のしやすさ",
            "期待とのギャップ", "他社製品との比較",
        ],
    },
    "hotel_dining": {
        "hotel": [
            "客室の清掃状態と清潔感", "立地の分かりやすさとアクセス", "静かさ・居心地の良さ", "部屋の広さと設備",
            "接客の丁寧さと距離感", "価格とサービス内容の釣り合い", "朝食の内容と品数", "チェックイン・チェックアウトのスムーズさ",
            "空調や水回りの快適さ", "周辺環境との相性", "リピートしたいかどうか",
        ],
        "dine_in": [
            "味・味付けのバランス", "提供スピードとタイミング", "接客の丁寧さと距離感", "価格とボリューム感",
            "店内の清潔感と席の居心地", "立地の分かりやすさとアクセス", "待ち時間と予約の取りやすさ", "盛り付けや見た目の丁寧さ",
            "料理の温度管理", "支払い方法や会計のスムーズさ", "混雑時のオペレーション", "リピートしたいかどうか",
        ],
        "takeout": [
            "味・味付けのバランス", "受け取り・配達のスピードと正確さ", "価格とボリューム感", "持ち帰り用容器・梱包の丁寧さ",
            "受け渡し時の接客と対応", "温かさや鮮度の保たれ方", "予約・注文画面の分かりやすさ", "支払い方法や会計のスムーズさ",
            "リピートしたいかどうか",
        ],
    },
    "survey_workplace": {
        "event": [
            "全体的な構成と時間配分", "資料や案内文の分かりやすさ", "会場や配信環境の快適さ", "事前連絡と案内のタイミング",
            "担当講師・スタッフの進行", "実務・日常への役立ち度", "予約・受付のスムーズさ", "質問や意見の出しやすさ",
            "コストや負担感との釣り合い", "参加者同士や周囲の雰囲気", "事後フォロー・資料共有", "次回も参加したいか",
        ],
        "system": [
            "画面や案内の分かりやすさ", "操作手順のシンプルさと入力しやすさ", "処理速度と動作の安定性", "事前案内とヘルプの分かりやすさ",
            "エラー時や修正時の対応しやすさ", "実務・手続きの効率化度合い", "アクセスや認証の手間", "改善要望の反映度",
            "継続利用しやすいか",
        ],
        "facility": [
            "設備・環境面の快適さ", "案内表示や利用ルールの分かりやすさ", "予約・受付や利用手続きのスムーズさ", "衛生面と安全配慮",
            "立地・アクセスや時間帯の選びやすさ", "混雑状況と待ち時間", "コストや負担感との釣り合い", "スタッフの窓口対応",
            "継続利用したいか",
        ],
    },
}

# 本番追加用サブタイプ定義（商品限定フレーズの汎化・U1/U2中立厚みの明確化）
PROD_SUBTYPES_BY_CLASS: Dict[str, List[Tuple[str, float, str, str]]] = {
    "positive": [
        ("P1_direct_specific", 0.35, "standard", "具体的な良い点や体験に基づく自然な肯定。安易な「最高！完璧！感動！」などの大げさな誇張語は避け、生活感のある表現にする"),
        ("P2_litotes_negation", 0.20, "hard", "否定表現（「悪くない」「文句なし」「不満はない」「期待を裏切らない」「不便を感じない」「選んで正解だった」など）を含みつつ意味は明確にポジティブにする"),
        ("P3_mixed_pos_dominant", 0.20, "hard", "小さな欠点や気になる点にも正直に触れつつ、トータルでは満足しており前向きに肯定・推奨している両面評価にする"),
        ("P4_understated_pos", 0.15, "hard", "派手な褒め言葉を使わず、「地味に助かる」「意外と重宝している」「ちょうど良い」など控えめで落ち着いた肯定にする"),
        ("P5_relief_turnaround", 0.10, "hard", "最初は不安・手間・初期の戸惑いがあったが、使ってみたら／対応してもらったら安心・満足に変わったという好転の肯定にする"),
    ],
    "negative": [
        ("N1_direct_specific", 0.30, "standard", "具体的な不具合・使いづらさ・期待外れの点に基づく自然な不満。暴言や罵倒は使わず生活者の言葉で不満を述べる"),
        ("N2_understated_complaint", 0.22, "hard", "激しい怒りや「最悪」などの強い単語を避け、「期待ほどではなかった」「リピートはしないと思う」「少し残念」「もう一工夫ほしい」など控えめ・遠回しな不満にする"),
        ("N3_mixed_neg_dominant", 0.20, "hard", "見た目や一部機能など良い点も認めつつ、肝心の部分で困っており全体としては不満・非推奨である両面評価にする"),
        ("N4_irony_sarcasm", 0.16, "hard", "皮肉めいた表現（表面上は「さすが」「見事な」「親切にも」「おかげで」等の言葉を用いつつ、文脈全体として明確に呆れ・不満・ネガティブだと読み取れる文。説明臭くせず自然な日本語の皮肉にする）"),
        ("N5_negated_positive", 0.12, "hard", "「満足とは言えない」「良いとは言い難い」「おすすめはできない」「快適とはほど遠い」など肯定語を打ち消す形のネガティブにする"),
    ],
    "neutral": [
        # 追加生成は U1/U2 をやや厚めに生成し、検証通過後の目標比率 (38/26/21/15) を余裕で満たす
        ("U1_factual_report", 0.42, "standard", "評価・好悪・不満・改善要望を一切含めず、利用した事実・スペック・日時・手順・経過のみを淡々と客観的に記す純粋な事実文にする（不具合や不便さの愚痴にしない）"),
        ("U2_balanced_mixed", 0.26, "hard", "良い点と悪い点が完全に拮抗しており、「一長一短」「良くも悪くも普通」「好みや用途で評価が分かれそう」と書き手自身も総合極性を決めていない均衡両面評価にする"),
        ("U4_emotion_word_factual", 0.20, "hard", "文中に「人気」「評判」「満足」「不満」などの語が含まれていても、伝聞・ニュース・条件確認であり書き手個人の評価や感情極性は中立にする"),
        ("U3_question_inquiry", 0.12, "standard", "褒めも貶しもしない中立的な質問・仕様確認・比較検討メモにする"),
    ],
}


def classify_product(subtopic: str) -> str:
    for h in PRODUCT_CAT_HINTS["consumable"]:
        if h in subtopic:
            return "consumable"
    for h in PRODUCT_CAT_HINTS["gear"]:
        if h in subtopic:
            return "gear"
    return "electric"


def classify_hotel(subtopic: str) -> str:
    for h in HOTEL_CAT_HINTS["hotel"]:
        if h in subtopic:
            return "hotel"
    for h in HOTEL_CAT_HINTS["takeout"]:
        if h in subtopic:
            return "takeout"
    return "dine_in"


def classify_survey(subtopic: str) -> str:
    for h in SURVEY_CAT_HINTS["system"]:
        if h in subtopic:
            return "system"
    for h in SURVEY_CAT_HINTS["event"]:
        if h in subtopic:
            return "event"
    return "facility"


def choose_aspect(domain: str, subtopic: str, rng: random.Random) -> str:
    if domain == "product_review":
        cat = classify_product(subtopic)
        return rng.choice(ASPECTS_BY_DOMAIN_TYPED["product_review"][cat])
    if domain == "hotel_dining":
        cat = classify_hotel(subtopic)
        return rng.choice(ASPECTS_BY_DOMAIN_TYPED["hotel_dining"][cat])
    if domain == "survey_workplace":
        cat = classify_survey(subtopic)
        return rng.choice(ASPECTS_BY_DOMAIN_TYPED["survey_workplace"][cat])
    return rng.choice(sc.ASPECTS_BY_DOMAIN[domain])


def build_prod_subtopics() -> Dict[str, Dict[str, List[str]]]:
    base = sc.split_subtopics()
    out: Dict[str, Dict[str, List[str]]] = {}
    for dom, pools in base.items():
        out[dom] = {}
        extra_map = EXTRA_SUBTOPICS[dom]
        for p in ("train", "dev", "sealed_test", "human_eval"):
            merged = list(pools[p]) + list(extra_map.get(p, []))
            assert len(set(merged)) == len(merged), f"Dup inside {dom}:{p}"
            out[dom][p] = merged

    # 全ドメイン×4プール完全排他検査
    by_pool: Dict[str, Set[str]] = {p: set() for p in ("train", "dev", "sealed_test", "human_eval")}
    for dom, pmap in out.items():
        for p, lst in pmap.items():
            by_pool[p].update(f"{dom}::{x}" for x in lst)
    pnames = list(by_pool.keys())
    for i in range(len(pnames)):
        for j in range(i + 1, len(pnames)):
            inter = by_pool[pnames[i]] & by_pool[pnames[j]]
            assert not inter, f"Overlap between {pnames[i]} and {pnames[j]}: {inter}"
    return out


def build_prod_recipes(
    pool_subtopics: Dict[str, Dict[str, List[str]]],
    counts: Dict[str, int],
    seed: int = 20261122,
) -> List[dict]:
    rng = random.Random(seed)
    recipes: List[dict] = []
    dom_specs = list(sc.DOMAIN_WEIGHTS)
    g_idx = 0

    for pool in ("train", "dev", "sealed_test"):
        n_target = counts[pool]
        dom_list: List[str] = []
        for d, w in dom_specs:
            dom_list.extend([d] * int(round(n_target * w)))
        while len(dom_list) < n_target:
            dom_list.append(dom_specs[len(dom_list) % len(dom_specs)][0])
        dom_list = dom_list[:n_target]
        rng.shuffle(dom_list)

        for local_i, dom in enumerate(dom_list):
            subs = pool_subtopics[dom][pool]
            subtopic = subs[local_i % len(subs)]
            aspect = choose_aspect(dom, subtopic, rng)
            persona = rng.choice(sc.PERSONAS)
            style_code, _, style_desc = sc.pick_weighted(rng, sc.STYLES)
            len_code, _, len_desc = sc.pick_weighted(rng, sc.LENGTH_BANDS)

            pos_sub = sc.pick_weighted(rng, PROD_SUBTYPES_BY_CLASS["positive"])
            neg_sub = sc.pick_weighted(rng, PROD_SUBTYPES_BY_CLASS["negative"])
            neu_sub = sc.pick_weighted(rng, PROD_SUBTYPES_BY_CLASS["neutral"])

            recipes.append({
                "recipe_id": f"prod_{pool[:3]}_{local_i:05d}",
                "seed_idx": g_idx,
                "split_pool": pool,
                "domain": dom,
                "subtopic": subtopic,
                "aspect": aspect,
                "persona": persona,
                "style": style_code,
                "style_desc": style_desc,
                "length_band": len_code,
                "length_desc": len_desc,
                "subtypes": {
                    "positive": {"code": pos_sub[0], "difficulty": pos_sub[2], "desc": pos_sub[3]},
                    "negative": {"code": neg_sub[0], "difficulty": neg_sub[2], "desc": neg_sub[3]},
                    "neutral": {"code": neu_sub[0], "difficulty": neu_sub[2], "desc": neu_sub[3]},
                },
            })
            g_idx += 1
    return recipes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "prod_recipes"))
    ap.add_argument("--train-recipes", type=int, default=48000)
    ap.add_argument("--dev-recipes", type=int, default=2000)
    ap.add_argument("--sealed-recipes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261122)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_subtopics = build_prod_subtopics()
    counts = {"train": args.train_recipes, "dev": args.dev_recipes, "sealed_test": args.sealed_recipes}
    recipes = build_prod_recipes(pool_subtopics, counts, seed=args.seed)

    raw_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in recipes) + "\n"
    out_jsonl = out_dir / "prod_recipes.jsonl"
    out_gz = out_dir / "prod_recipes.jsonl.gz"
    out_jsonl.write_text(raw_text, encoding="utf-8")
    with gzip.open(out_gz, "wt", encoding="utf-8", compresslevel=6) as fg:
        fg.write(raw_text)

    # 重複・統計検査
    sub_counts_by_pool = Counter()
    for dom, pmap in pool_subtopics.items():
        for p, lst in pmap.items():
            sub_counts_by_pool[p] += len(lst)

    neu_sub_dist = Counter(r["subtypes"]["neutral"]["code"] for r in recipes)
    summary = {
        "total_subtopics": sum(sub_counts_by_pool.values()),
        "subtopics_by_pool": dict(sub_counts_by_pool),
        "subtopics_by_domain": {dom: {p: len(lst) for p, lst in pmap.items()} for dom, pmap in pool_subtopics.items()},
        "subtopic_cross_pool_overlap": 0,
        "total_recipes": len(recipes),
        "by_pool": dict(Counter(r["split_pool"] for r in recipes)),
        "by_domain": dict(Counter(r["domain"] for r in recipes)),
        "neutral_subtype_dist": dict(neu_sub_dist),
        "jsonl_bytes": out_jsonl.stat().st_size,
        "gz_bytes": out_gz.stat().st_size,
    }
    (out_dir / "prod_recipe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
