#!/usr/bin/env python3
"""第12号 ja-sentiment-v1 シナリオ・レシピ生成モジュール。

6ドメイン × 計372種の具体サブトピックを 4プール (train / dev / sealed_test / human_eval) へ
完全排他分割し、難例サブタイプ・着眼点・文体・長さ帯を均等配分したレシピ JSONL を出力する。
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent

# ドメイン構成と目標比率
DOMAIN_WEIGHTS = [
    ("product_review", 0.22),
    ("hotel_dining", 0.18),
    ("sns_short", 0.18),
    ("support_feedback", 0.16),
    ("app_review", 0.16),
    ("survey_workplace", 0.10),
]

# 6ドメイン × 具体サブトピック（合計372種）
SUBTOPICS_BY_DOMAIN: Dict[str, List[str]] = {
    "product_review": [
        # 72種: train 50 / dev 7 / sealed 7 / human_eval 8
        "ワイヤレスイヤホン", "電気ケトル", "加湿空気清浄機", "コードレス掃除機", "電動歯ブラシ",
        "デスクライト", "USB急速充電器", "モバイルバッテリー", "ノートPCスタンド", "メカニカルキーボード",
        "静音マウス", "モニターアーム", "低反発枕", "羽毛掛け布団", "遮光カーテン",
        "ステンレスマグボトル", "フライパン", "炊飯器", "電子レンジ", "トースター",
        "ハンドブレンダー", "コーヒーメーカー", "珪藻土バスマット", "シャワーヘッド", "ドライヤー",
        "洗濯用洗剤", "柔軟剤", "食器用洗剤", "入浴剤", "フェイスタオル",
        "撥水リュック", "キャリーケース", "折りたたみ傘", "ランニングシューズ", "レインブーツ",
        "保温インナー", "UVカットパーカー", "サングラス", "革財布", "腕時計ベルト",
        "ドリップコーヒー豆", "緑茶ティーバッグ", "ミックスナッツ", "プロテインパウダー", "無添加だしパック",
        "冷凍餃子", "レトルトカレー", "オリーブオイル", "はちみつ", "化粧水",
        # 50↑ train / 50..56 dev
        "日焼け止めジェル", "ハンドクリーム", "リップバーム", "ボールペンセット", "システム手帳",
        "シュレッダー", "ラベルライター",
        # 57..63 sealed_test
        "スマート体重計", "防災ラジオ", "ソーラーランタン", "キャンプ用チェア", "クーラーボックス",
        "ペット用自動給水器", "猫用爪とぎ",
        # 64..71 human_eval
        "セラミック包丁", "保存容器セット", "スマートウォッチ替バンド", "骨伝導ヘッドホン",
        "衣類スチーマー", "首掛けネックピロー", "ノンアルコールビール", "頭皮マッサージブラシ",
    ],
    "hotel_dining": [
        # 60種: train 40 / dev 6 / sealed 6 / human_eval 8
        "駅前のビジネスホテル", "温泉旅館の露天風呂付き客室", "古民家リノベーション宿", "空港直結ホテル", "カプセルホテル",
        "ファミリー向けリゾートホテル", "ペット同伴可のコテージ", "町家の貸切ゲストハウス", "駅前ラーメン店", "街の中華料理屋",
        "手打ち蕎麦屋", "海鮮居酒屋", "焼き鳥屋", "回転寿司チェーン", "本格江戸前寿司店",
        "炭火焼肉店", "しゃぶしゃぶ食べ放題店", "お好み焼き屋", "老舗の洋食店", "イタリアンバル",
        "石窯ピザ専門店", "ビストロフレンチ", "スパイスカレー専門店", "タイ料理店", "ベーカリーカフェ",
        "自家焙煎喫茶店", "駅ナカの立ち食いそば屋", "ショッピングモールのフードコート", "ホテルの朝食ビュッフェ", "ランチ定食屋",
        "クラフトビール専門店", "深夜営業の居酒屋", "テイクアウト専門唐揚げ店", "宅配ピザ", "出前寿司",
        "デパ地下の総菜コーナー", "抹茶スイーツ専門店", "ジェラート専門店", "チーズケーキ専門店", "和菓子屋",
        # 40..45 dev
        "山間の湯治宿", "海辺のペンション", "うどん専門店", "ベトナム料理フォー専門店", "パンケーキ専門カフェ", "駅弁販売店",
        # 46..51 sealed_test
        "デザイナーズホテル", "郊外の大型道の駅食堂", "とんかつ専門店", "韓国料理スンドゥブ専門店", "タピオカ・お茶専門店", "デリバリー弁当サービス",
        # 52..59 human_eval
        "高速道路サービスエリアのレストラン", "シティホテルのラウンジバー", "天ぷら専門店", "スープカレー店",
        "深夜営業のファミレス", "モーニング提供の純喫茶", "串揚げ屋", "ビュッフェ形式の自然食レストラン",
    ],
    "sns_short": [
        # 60種: train 40 / dev 6 / sealed 6 / human_eval 8
        "朝の満員電車での通勤", "急な夕立と傘", "コンビニの新作スイーツ", "週末のスーパーの特売", "ネット通販の置き配",
        "美容院でのヘアカット", "歯医者の定期検診", "健康診断の結果待ち", "役所のマイナンバー手続き", "郵便局の窓口待ち",
        "図書館での読書時間", "映画館のレイトショー", "美術館の特別展", "音楽ライブのチケット抽選", "プロ野球の現地観戦",
        "近所の公園の桜", "真夏のエアコン電気代", "花粉症シーズンの外出", "冬の朝の布団からの脱出", "引っ越しの荷造り",
        "家具の組み立て作業", "部屋の断捨離", "ベランダでの家庭菜園", "近所の商店街の夏祭り", "週末のコインランドリー",
        "免許更新センターの講習", "車検の見積もり", "スマートフォンの機種変更", "パソコンのOSアップデート", "在宅勤務中のオンライン会議",
        "職場の歓迎会・送別会", "資格試験の勉強", "英語のリスニング学習", "朝のジョギング", "筋トレの筋肉痛",
        "週末の日帰り温泉", "実家への帰省新幹線", "友人の結婚式の準備", "年賀状・季節の挨拶", "家電量販店での買い物",
        # 40..45 dev
        "夜の高速バス移動", "休日のホームセンター巡り", "確定申告の書類整理", "近所のパン屋の新メニュー", "スマホの画面保護フィルム貼り", "雨の日の犬の散歩",
        # 46..51 sealed_test
        "早朝の空港チェックイン", "年末の大掃除", "銀行窓口での住所変更", "ドラッグストアのポイント還元", "推しの配信イベント視聴", "冷凍庫の霜取り作業",
        # 52..59 human_eval
        "深夜のコンビニ立ち寄り", "週末の道の駅ドライブ", "市役所の転入届手続き", "オンライン英会話の体験",
        "台風の日のテレワーク", "メガネのレンズ交換", "クリーニング店の受け取り", "朝一番の喫茶店モーニング",
    ],
    "support_feedback": [
        # 60種: train 42 / dev 6 / sealed 6 / human_eval 6
        "家電メーカーの電話修理受付", "ネット通販の返品交換窓口", "宅配業者の再配達手配", "クレジットカード紛失時の緊急窓口", "携帯電話キャリアの料金プラン相談",
        "インターネット回線の開通工事調整", "銀行の振込エラー問い合わせ", "保険会社の給付金請求窓口", "航空会社の欠航振替カウンター", "旅行予約サイトのキャンセル相談",
        "引っ越し業者の見積もり訪問", "水道修理業者の緊急出動", "エアコンクリーニング業者の訪問対応", "車のディーラーの定期点検受付", "パソコンメーカーのチャットサポート",
        "ソフトウェアのライセンス認証問い合わせ", "サブスク解約手続きの案内画面", "クラウドストレージの容量課金相談", "電子決済アプリの不審請求調査", "フリマアプリの取引トラブル仲介",
        "ポイントプログラムの付与漏れ照会", "定期購入の配送スキップ手続き", "ふるさと納税のワンストップ特例窓口", "電力会社のプラン変更サポート", "ガス会社の閉栓立ち会い予約",
        "学習塾の入塾・コース相談", "英会話スクールの休会手続き", "フィットネスジムの退会受付", "不動産管理会社への設備故障連絡", "マンション修繕の事前説明対応",
        "ペット保険の請求サポート", "眼鏡店のフレーム調整・保証対応", "時計修理のオーバーホール見積もり", "靴修理店のソール交換受付", "クリーニング店のシミ抜き相談",
        "自治体コールセンターの制度案内", "病院の予約変更電話窓口", "調剤薬局のオンライン服薬指導", "写真プリントサービスの誤発送対応", "印刷通販のデータ入稿サポート",
        "ゲーム運営への課金反映問い合わせ", "動画配信サービスの視聴不具合チャット",
        # 42..47 dev
        "スマートロックの故障交換窓口", "レンタカー会社の事故・傷確認対応", "証券会社の口座開設不備案内", "浄水器カートリッジ定期便の解約窓口", "楽器店の修理リペア相談", "光回線ルーター交換サポート",
        # 48..53 sealed_test
        "ホテル予約の人数変更電話対応", "Wi-Fiレンタルの返却遅延確認", "通販サイトの二重決済取消対応", "家電量販店の延長保証受付", "自動車保険のロードサービス手配", "介護用品レンタルの変更相談",
        # 54..59 human_eval
        "スマート家電の初期接続チャット案内", "航空券の名前入力訂正サポート", "法人向けSaaSの請求書修正窓口", "自転車通販の初期不良パーツ発送対応", "ネットスーパーの欠品連絡と返金処理", "鉄道会社の忘れ物照会窓口",
    ],
    "app_review": [
        # 60種: train 42 / dev 6 / sealed 6 / human_eval 6
        "家計簿アプリ", "乗換案内アプリ", "天気予報アプリ", "メモ・ノートアプリ", "ToDo・タスク管理アプリ",
        "カレンダー予定共有アプリ", "電子書籍リーダーアプリ", "マンガ配信アプリ", "音楽ストリーミングアプリ", "ポッドキャストアプリ",
        "動画配信アプリ", "写真クラウドバックアップアプリ", "写真レタッチ編集アプリ", "QRコード決済アプリ", "ネットバンキングアプリ",
        "証券取引アプリ", "ポイント管理アプリ", "フリマアプリ", "ネットショッピングアプリ", "フードデリバリーアプリ",
        "飲食店予約アプリ", "ホテル宿泊予約アプリ", "航空券比較アプリ", "カーナビ・地図アプリ", "タクシー配車アプリ",
        "歩数計・健康管理アプリ", "睡眠記録アプリ", "ランニング記録アプリ", "食事カロリー記録アプリ", "語学学習アプリ",
        "ニュースキュレーションアプリ", "地震・防災速報アプリ", "ゴミ出しカレンダーアプリ", "レシピ検索アプリ", "自治体公式行政アプリ",
        "スマートホーム家電操作アプリ", "名刺スキャン管理アプリ", "PDF閲覧・署名アプリ", "パスワード管理アプリ", "ビデオ会議アプリ",
        "グループチャットアプリ", "スケジュール調整Webサービス",
        # 42..47 dev
        "レシート撮影ポイ活アプリ", "割り勘計算アプリ", "自転車シェアリングアプリ", "駐車場予約アプリ", "お薬手帳アプリ", "翻訳カメラアプリ",
        # 48..53 sealed_test
        "図書館蔵書検索アプリ", "電卓・単位変換アプリ", "録音・文字起こしアプリ", "ペット健康記録アプリ", "映画レビュー記録アプリ", "電子チケット発券アプリ",
        # 54..59 human_eval
        "空港ラウンジ・搭乗管理アプリ", "カーシェア予約アプリ", "家電保証書管理アプリ", "登山GPS地図アプリ", "英語単語帳アプリ", "高速バス予約アプリ",
    ],
    "survey_workplace": [
        # 60種: train 40 / dev 6 / sealed 6 / human_eval 8
        "オンライン技術セミナー", "社内セキュリティ研修", "新入社員向けオンボーディング", "社食・カフェテリアのメニュー", "オフィスの空調・照明環境",
        "社内ポータルサイトの使いやすさ", "経費精算システムの操作性", "勤怠管理ツールの入力画面", "リモートワーク制度の運用", "フレックスタイム制度の利用感",
        "市民体育館のトレーニング室", "公共図書館の自習スペース", "市役所のオンライン申請システム", "地域の防災訓練", "公立病院の会計待ち時間システム",
        "人間ドックの受診フロー", "自動車教習所の学科・技能指導", "資格対策の通信講座", "社会人向けプログラミング教室", "絵画・陶芸のカルチャースクール",
        "料理教室の体験レッスン", "ピアノ教室の個人レッスン", "英会話カフェのグループレッスン", "ヨガスタジオの早朝クラス", "スイミングスクールの施設衛生",
        "コワーキングスペースの会議室", "貸し会議室の音響・プロジェクター", "展示会・カンファレンスの運営", "町内会の清掃活動", "PTAの連絡アプリ運用",
        "保育園の連絡帳アプリ", "学童保育のお迎えルール", "大学オープンキャンパスの案内", "キャリア相談会の個別面談", "移住相談窓口のオンライン面談",
        "観光案内所のパンフレット・説明", "博物館の音声ガイド", "音楽ホールの座席・音響", "イベント会場の誘導スタッフ", "社員旅行・懇親会の企画",
        # 40..45 dev
        "社内チャットツールの通知設計", "健康保険組合の特定保健指導", "公民館のパソコン講座", "レンタルトランクルームの手続き", "シェアオフィスの複合機運用", "産業医とのオンライン面談",
        # 46..51 sealed_test
        "福利厚生カフェテリアプラン", "自治体ゴミ分別説明会", "大型展示場のクローク対応", "コミュニティバスの運行ダイヤ", "シニア向けスマホ教室", "職場ストレスチェックの受検画面",
        # 52..59 human_eval
        "社内動画マニュアルの視聴サイト", "地域子育て支援センターの利用環境", "駅前駐輪場の定期更新手続き", "オンライン確定申告相談会",
        "公共プールの更衣室設備", "学会シンポジウムの質問フォーム", "社内備品貸出の申請フロー", "市民ホールのチケット窓口対応",
    ],
}

POOL_QUOTAS = {
    "product_review": {"train": 50, "dev": 7, "sealed_test": 7, "human_eval": 8},
    "hotel_dining": {"train": 40, "dev": 6, "sealed_test": 6, "human_eval": 8},
    "sns_short": {"train": 40, "dev": 6, "sealed_test": 6, "human_eval": 8},
    "support_feedback": {"train": 42, "dev": 6, "sealed_test": 6, "human_eval": 6},
    "app_review": {"train": 42, "dev": 6, "sealed_test": 6, "human_eval": 6},
    "survey_workplace": {"train": 40, "dev": 6, "sealed_test": 6, "human_eval": 8},
}

ASPECTS_BY_DOMAIN: Dict[str, List[str]] = {
    "product_review": [
        "使い勝手と操作性", "価格に対するコスパ", "耐久性・作りのしっかり感", "サイズ感と重量",
        "音や匂い・稼働音", "デザインと質感", "バッテリー持ち・省エネ性", "手入れのしやすさ",
        "梱包と初期状態", "説明書の分かりやすさ", "期待とのギャップ", "買い替え前との比較",
        "家族・同僚への勧めやすさ", "細かい部品の精度", "収納・持ち運びやすさ", "消耗品の入手性",
    ],
    "hotel_dining": [
        "味・味付けのバランス", "提供スピードとタイミング", "接客の丁寧さと距離感", "価格とボリューム感",
        "店内の清潔感・客室の清掃状態", "立地の分かりやすさとアクセス", "静かさ・居心地の良さ", "座席・部屋の広さと設備",
        "メニュー・朝食の品数", "待ち時間と予約の取りやすさ", "盛り付けや見た目の丁寧さ", "リピートしたいかどうか",
        "温度管理（料理の温かさ・空調）", "支払い方法や会計のスムーズさ", "周辺環境との相性", "混雑時のオペレーション",
    ],
    "sns_short": [
        "予定の進み具合", "待ち時間・移動の体感", "ちょっとした出来事への感想", "体力・気分の変化",
        "周囲の混雑や雰囲気", "予想外のハプニングへの反応", "時間とお金の使い方", "天気・気温の影響",
        "作業の手間と仕上がり", "日常の小さな発見", "人とのやり取りの空気感", "次回の予定への気持ち",
        "情報収集と実際の違い", "週末や休日の過ごし方", "生活リズムへの影響", "道具やサービスの相性",
    ],
    "support_feedback": [
        "電話・チャットの繋がりやすさ", "担当者の説明の分かりやすさ", "解決までにかかった日数・時間", "手続きの煩雑さ",
        "案内の一貫性（たらい回しの有無）", "言葉遣いと傾聴の姿勢", "代替案や補償の妥当性", "折り返し連絡の正確さ",
        "FAQや自動ボットの機能", "書類・写真提出の分かりやすさ", "例外対応の柔軟さ", "完了後のフォロー連絡",
        "手数料や負担の納得感", "営業時間の使いやすさ", "進捗確認のしやすさ", "再発防止への姿勢",
    ],
    "app_review": [
        "画面の分かりやすさ・UI", "起動速度と動作の軽さ", "アップデートの安定性・バグ有無", "広告の頻度と邪魔にならなさ",
        "通知の細かな設定可否", "検索・フィルタの精度", "課金プランと無料機能の範囲", "アカウント連携・引き継ぎの簡単さ",
        "バッテリーや通信量の消費", "オフラインや低速回線での挙動", "文字サイズと視認性", "ウィジェットやショートカット対応",
        "データ出力・バックアップ機能", "セキュリティ認証の手間", "サポート・ヘルプの充実度", "競合アプリからの乗り換え感",
    ],
    "survey_workplace": [
        "全体的な構成と時間配分", "資料や案内文の分かりやすさ", "設備・環境面の快適さ", "事前連絡と案内のタイミング",
        "システム・フォームの入力しやすさ", "担当講師・スタッフの進行", "実務・日常への役立ち度", "予約・受付のスムーズさ",
        "質問や意見の出しやすさ", "コストや負担感との釣り合い", "衛生面と安全配慮", "改善要望の反映度",
        "アクセスや時間帯の選びやすさ", "参加者同士や周囲の雰囲気", "事後フォロー・資料共有", "継続利用したいか",
    ],
}

PERSONAS = [
    "30代会社員（平日は効率重視）",
    "40代主婦・主夫（コスパと実用性重視）",
    "20代大学生（SNS慣れした自然な感覚）",
    "50代管理職（落ち着いた観察眼）",
    "60代シニア（丁寧だが率直な評価）",
    "20代社会人1〜2年目（率直な実体験）",
    "フリーランス在宅ワーカー（作業環境やツールに敏感）",
    "子育て世代の親（時間制約が厳しい視点）",
    "地方在住の車・ネット通販ユーザー",
    "ガジェット・道具好きの慎重派",
    "忙しい通勤客（短時間で要点を見る）",
    "初めてそのサービス・製品を使った初心者",
]

STYLES = [
    ("polite", 0.50, "丁寧なです・ます調（自然なレビューやアンケートの文体）"),
    ("plain", 0.30, "落ち着いた常体（だ・である調、または簡潔な体言止め交じり）"),
    ("casual_sns", 0.20, "くだけた口語・SNS投稿調（過剰な絵文字や顔文字は使わない自然な日常文）"),
]

LENGTH_BANDS = [
    ("short", 0.30, "15〜45文字程度の短い1文"),
    ("medium", 0.50, "50〜110文字程度の1〜2文"),
    ("long", 0.20, "120〜220文字程度の具体的な2〜4文"),
]

# 3クラス × 難例サブタイプ定義
SUBTYPES_BY_CLASS: Dict[str, List[Tuple[str, float, str, str]]] = {
    "positive": [
        ("P1_direct_specific", 0.35, "standard", "具体的な良い点や体験に基づく自然な肯定。安易な「最高！完璧！感動！」などの大げさな誇張語は避け、生活感のある表現にする"),
        ("P2_litotes_negation", 0.20, "hard", "否定表現（「悪くない」「文句なし」「不満はない」「期待を裏切らない」「不便を感じない」「買って損はなかった」など）を含みつつ意味は明確にポジティブにする"),
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
        ("U1_factual_report", 0.40, "standard", "評価や好悪の感情を込めず、利用した事実・仕様・経過・スペック・日時・手順だけを淡々と記した事実文にする"),
        ("U2_balanced_mixed", 0.25, "hard", "良い点と悪い点が同程度にあり、書き手自身も「一長一短」「良くも悪くも普通」「人によって評価が分かれそう」と総合的な好悪を決めていない均衡評価にする"),
        ("U3_question_inquiry", 0.15, "standard", "褒めも貶しもしない中立的な質問・確認・検討メモ（例：仕様の問い合わせ、購入や利用を迷っているメモ）にする"),
        ("U4_emotion_word_factual", 0.20, "hard", "文中に「人気」「評判」「苦情」「満足」「不満」などの評価語が含まれていても、伝聞・ニュース・客観観察・条件確認であり書き手個人の感情極性は中立にする"),
    ],
}


def pick_weighted(rng: random.Random, specs: List[tuple]) -> tuple:
    r = rng.random()
    acc = 0.0
    for item in specs:
        acc += item[1]
        if r <= acc:
            return item
    return specs[-1]


def split_subtopics() -> Dict[str, Dict[str, List[str]]]:
    """全ドメインのサブトピックを 4プールへ完全排他的に事前分割して返す。"""
    out: Dict[str, Dict[str, List[str]]] = {}
    for dom, items in SUBTOPICS_BY_DOMAIN.items():
        q = POOL_QUOTAS[dom]
        total_expected = sum(q.values())
        assert len(items) == total_expected, f"{dom}: {len(items)} != {total_expected}"
        assert len(set(items)) == len(items), f"Duplicate subtopics inside {dom}"
        idx = 0
        out[dom] = {}
        for pool in ("train", "dev", "sealed_test", "human_eval"):
            cnt = q[pool]
            out[dom][pool] = items[idx : idx + cnt]
            idx += cnt
    # 念のため全ドメイン横断でもプール間重複ゼロを検査
    all_by_pool = {p: set() for p in ("train", "dev", "sealed_test", "human_eval")}
    for dom, pmap in out.items():
        for p, lst in pmap.items():
            all_by_pool[p].update(f"{dom}::{x}" for x in lst)
    pools = list(all_by_pool.keys())
    for i in range(len(pools)):
        for j in range(i + 1, len(pools)):
            inter = all_by_pool[pools[i]] & all_by_pool[pools[j]]
            assert not inter, f"Overlap between {pools[i]} and {pools[j]}: {inter}"
    return out


def build_pilot_recipes(
    pool_subtopics: Dict[str, Dict[str, List[str]]],
    counts: Dict[str, int],
    seed: int = 42,
) -> List[dict]:
    """train / dev / sealed_test 用の 1レシピ=3クラス生成レシピを構築。"""
    rng = random.Random(seed)
    recipes: List[dict] = []
    dom_specs = [(d, w) for d, w in DOMAIN_WEIGHTS]
    g_idx = 0

    for pool in ("train", "dev", "sealed_test"):
        n_target = counts[pool]
        # ドメイン割当リストを作ってシャッフル
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
            aspect = rng.choice(ASPECTS_BY_DOMAIN[dom])
            persona = rng.choice(PERSONAS)
            style_code, _, style_desc = pick_weighted(rng, STYLES)
            len_code, _, len_desc = pick_weighted(rng, LENGTH_BANDS)

            pos_sub = pick_weighted(rng, SUBTYPES_BY_CLASS["positive"])
            neg_sub = pick_weighted(rng, SUBTYPES_BY_CLASS["negative"])
            neu_sub = pick_weighted(rng, SUBTYPES_BY_CLASS["neutral"])

            recipes.append({
                "recipe_id": f"{pool[:3]}_{local_i:05d}",
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


def build_human_eval_recipes(
    pool_subtopics: Dict[str, Dict[str, List[str]]],
    per_class: int = 350,
    seed: int = 20261121,
) -> List[dict]:
    """human_eval 専用の 1レシピ=1文 独立生成レシピを構築（生成条件を学習データと完全分離）。"""
    rng = random.Random(seed)
    recipes: List[dict] = []
    dom_names = [d for d, _ in DOMAIN_WEIGHTS]
    g_idx = 0

    for lbl in ("negative", "neutral", "positive"):
        sub_specs = SUBTYPES_BY_CLASS[lbl]
        # サブタイプごとに必要数を配分
        sub_list: List[tuple] = []
        for item in sub_specs:
            sub_list.extend([item] * int(round(per_class * item[1])))
        while len(sub_list) < per_class:
            sub_list.append(sub_specs[len(sub_list) % len(sub_specs)])
        sub_list = sub_list[:per_class]
        rng.shuffle(sub_list)

        for i, st_item in enumerate(sub_list):
            dom = dom_names[i % len(dom_names)]
            he_subs = pool_subtopics[dom]["human_eval"]
            subtopic = he_subs[(i // len(dom_names)) % len(he_subs)]
            aspect = rng.choice(ASPECTS_BY_DOMAIN[dom])
            persona = rng.choice(PERSONAS)
            style_code, _, style_desc = pick_weighted(rng, STYLES)
            len_code, _, len_desc = pick_weighted(rng, LENGTH_BANDS)

            recipes.append({
                "recipe_id": f"he_{lbl[:3]}_{i:04d}",
                "seed_idx": g_idx,
                "split_pool": "human_eval",
                "target_label": lbl,
                "domain": dom,
                "subtopic": subtopic,
                "aspect": aspect,
                "persona": persona,
                "style": style_code,
                "style_desc": style_desc,
                "length_band": len_code,
                "length_desc": len_desc,
                "subtype": st_item[0],
                "difficulty": st_item[2],
                "subtype_desc": st_item[3],
            })
            g_idx += 1

    rng.shuffle(recipes)
    return recipes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "recipes"))
    ap.add_argument("--train-recipes", type=int, default=10500)
    ap.add_argument("--dev-recipes", type=int, default=1150)
    ap.add_argument("--sealed-recipes", type=int, default=1150)
    ap.add_argument("--he-per-class", type=int, default=350)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_subtopics = split_subtopics()
    pilot_recipes = build_pilot_recipes(
        pool_subtopics,
        {"train": args.train_recipes, "dev": args.dev_recipes, "sealed_test": args.sealed_recipes},
        seed=args.seed,
    )
    he_recipes = build_human_eval_recipes(pool_subtopics, per_class=args.he_per_class, seed=20261121)

    pilot_path = out_dir / "pilot_recipes.jsonl"
    he_path = out_dir / "human_eval_recipes.jsonl"
    pilot_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in pilot_recipes) + "\n", encoding="utf-8")
    he_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in he_recipes) + "\n", encoding="utf-8")

    # 重複ゼロ検査
    subs_by_pool = defaultdict(set)
    for r in pilot_recipes + he_recipes:
        subs_by_pool[r["split_pool"]].add(f"{r['domain']}::{r['subtopic']}")
    pool_names = list(subs_by_pool.keys())
    overlap_total = 0
    for i in range(len(pool_names)):
        for j in range(i + 1, len(pool_names)):
            overlap_total += len(subs_by_pool[pool_names[i]] & subs_by_pool[pool_names[j]])
    assert overlap_total == 0, f"Subtopic overlap detected: {overlap_total}"

    summary = {
        "total_subtopics": sum(len(v) for v in SUBTOPICS_BY_DOMAIN.values()),
        "pilot_recipes": len(pilot_recipes),
        "pilot_by_pool": dict(Counter(r["split_pool"] for r in pilot_recipes)),
        "pilot_by_domain": dict(Counter(r["domain"] for r in pilot_recipes)),
        "human_eval_recipes": len(he_recipes),
        "human_eval_by_label": dict(Counter(r["target_label"] for r in he_recipes)),
        "human_eval_by_domain": dict(Counter(r["domain"] for r in he_recipes)),
        "subtopic_cross_pool_overlap": overlap_total,
    }
    (out_dir / "recipe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
