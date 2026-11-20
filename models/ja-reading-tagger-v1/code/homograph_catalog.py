#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 同形異音語カタログ。

現代日本語の文脈読み分けが重要かつ辞書方式が誤読しやすい主要な同形異音語を定義する。
各エントリは構築時に必ず UniDic ∪ SudachiDict で辞書検証を行う。
人名の読み・数字・英字は対象外。
"""
from typing import Dict, List

# 各エントリ形式:
# {
#   "surface": 原文上の代表表層（活用語は代表形、複合語統合対象にもなる）,
#   "span_surface": 漢字部分の表層（活用語以外は surface と同じ）,
#   "key": span_extractor 上の識別キー,
#   "category": "compound" | "single_kanji" | "predicate",
#   "readings": [
#       {"reading": "カタカナ表記読み(漢字部分)", "full_reading": "表層全体カタカナ", "sense": "意味", "cues": "判別手がかり"}
#   ]
# }

HOMOGRAPH_CATALOG: List[Dict] = [
    # ==================== 1. 熟語・複合語 (compound) ====================
    {
        "surface": "人気",
        "span_surface": "人気",
        "key": "人気",
        "category": "compound",
        "readings": [
            {"reading": "ニンキ", "full_reading": "ニンキ", "sense": "世間の評判がよいこと・好かれていること", "cues": "店や俳優や商品の人気が高い、人気を集める、人気作品"},
            {"reading": "ヒトケ", "full_reading": "ヒトケ", "sense": "人の気配・人のいる様子", "cues": "人気のない路地、人気が途絶える、人気を感じない"},
        ],
    },
    {
        "surface": "工夫",
        "span_surface": "工夫",
        "key": "工夫",
        "category": "compound",
        "readings": [
            {"reading": "クフウ", "full_reading": "クフウ", "sense": "よい方法ややり方を考え出すこと", "cues": "工夫を凝らす、工夫を重ねる、ちょっとした工夫"},
            {"reading": "コウフ", "full_reading": "コウフ", "sense": "土木・鉄道・鉱山などで働く作業員（人間）", "cues": "工夫たちが線路を直す、工夫を雇う、工事現場の工夫が汗を拭う"},
        ],
    },
    {
        "surface": "大家",
        "span_surface": "大家",
        "key": "大家",
        "category": "compound",
        "readings": [
            {"reading": "オオヤ", "full_reading": "オオヤ", "sense": "貸家・アパートの持ち主（家主）", "cues": "大家に家賃を払う、大家と店子、優しい大家さん"},
            {"reading": "タイカ", "full_reading": "タイカ", "sense": "ある分野・芸術・学問の第一人者や名家", "cues": "日本画の大家、文壇の大家、その道の大家として知られる"},
        ],
    },
    {
        "surface": "一時",
        "span_surface": "一時",
        "key": "一時",
        "category": "compound",
        "readings": [
            {"reading": "イチジ", "full_reading": "イチジ", "sense": "時刻の午後/午前1時、またはしばらくの間・いったん", "cues": "午後一時、一時的に停止する、一時帰国、一時しのぎ"},
            {"reading": "ヒトトキ", "full_reading": "ヒトトキ", "sense": "少しのゆったりした時間・ひととき", "cues": "穏やかな一時を過ごす、至福の一時、静かな一時"},
            {"reading": "イットキ", "full_reading": "イットキ", "sense": "ほんの短い間（いっとき）・昔の一時間単位", "cues": "一時の感情に流される、一時の夢、一時の迷い"},
        ],
    },
    {
        "surface": "下手",
        "span_surface": "下手",
        "key": "下手",
        "category": "compound",
        "readings": [
            {"reading": "ヘタ", "full_reading": "ヘタ", "sense": "技術・やり方が拙いこと、うっかりすると", "cues": "歌が下手だ、言い訳が下手、下手に出る、下手をすると"},
            {"reading": "シモテ", "full_reading": "シモテ", "sense": "舞台・客席から見て左側、川や風の下流側", "cues": "舞台の下手から登場する、舞台の下手袖"},
            {"reading": "シタテ", "full_reading": "シタテ", "sense": "相撲などの組み手・相手の言いなりになること", "cues": "下手投げ、下手から攻める、相手の下手に回る"},
        ],
    },
    {
        "surface": "上手",
        "span_surface": "上手",
        "key": "上手",
        "category": "compound",
        "readings": [
            {"reading": "ジョウズ", "full_reading": "ジョウズ", "sense": "技術・腕前が優れていること", "cues": "料理が上手だ、説明が上手、話し上手"},
            {"reading": "ウワテ", "full_reading": "ウワテ", "sense": "実力が上であること、相撲の上手", "cues": "相手が一枚上手だった、上手投げ、上手を取る"},
            {"reading": "カミテ", "full_reading": "カミテ", "sense": "舞台の右側、川・風の上流側", "cues": "舞台の上手へ退場する、川の上手から流れてくる"},
        ],
    },
    {
        "surface": "明日",
        "span_surface": "明日",
        "key": "明日",
        "category": "compound",
        "readings": [
            {"reading": "アシタ", "full_reading": "アシタ", "sense": "きょうの次の日（日常会話・口語的）", "cues": "明日また学校で会おう、明日の天気はどうかな"},
            {"reading": "アス", "full_reading": "アス", "sense": "きょうの次の日（やや改まった表現・報道・未来）", "cues": "明日の会議に備える、日本の明日を担う、明日への希望"},
            {"reading": "ミョウニチ", "full_reading": "ミョウニチ", "sense": "改まったビジネス・敬語での翌日", "cues": "明日改めてご連絡いたします、明日のご来社をお待ちします"},
        ],
    },
    {
        "surface": "昨日",
        "span_surface": "昨日",
        "key": "昨日",
        "category": "compound",
        "readings": [
            {"reading": "キノウ", "full_reading": "キノウ", "sense": "きょうの前の日（口語・一般）", "cues": "昨日友達と映画を見た、昨日の夜遅く"},
            {"reading": "サクジツ", "full_reading": "サクジツ", "sense": "改まった文章・報道・ビジネスでの前日", "cues": "昨日開催された式典、昨日の会議で決定した"},
        ],
    },
    {
        "surface": "今日",
        "span_surface": "今日",
        "key": "今日",
        "category": "compound",
        "readings": [
            {"reading": "キョウ", "full_reading": "キョウ", "sense": "この日・本日のこと", "cues": "今日は天気がいい、今日中に提出する"},
            {"reading": "コンニチ", "full_reading": "コンニチ", "sense": "現在の時代・いまの世の中（こんにち）", "cues": "今日に至るまで、今日の社会問題、今日的課題"},
        ],
    },
    {
        "surface": "日本",
        "span_surface": "日本",
        "key": "日本",
        "category": "compound",
        "readings": [
            {"reading": "ニホン", "full_reading": "ニホン", "sense": "国名・日本文化・日本語などの一般表現", "cues": "日本料理、日本語を学ぶ、日本各地を旅する"},
            {"reading": "ニッポン", "full_reading": "ニッポン", "sense": "公式・強調・スポーツ応援・放送などでの呼称", "cues": "日本一を目指す、日本列島、頑張れ日本"},
        ],
    },
    {
        "surface": "市場",
        "span_surface": "市場",
        "key": "市場",
        "category": "compound",
        "readings": [
            {"reading": "シジョウ", "full_reading": "シジョウ", "sense": "経済・株式・商品取引などの抽象的な場", "cues": "株式市場が上昇する、国内市場を開拓する、労働市場"},
            {"reading": "イチバ", "full_reading": "イチバ", "sense": "魚や野菜などを売買する具体的な場所・朝市", "cues": "朝早く魚市場へ行く、活気のある市場で買い物をする"},
        ],
    },
    {
        "surface": "生物",
        "span_surface": "生物",
        "key": "生物",
        "category": "compound",
        "readings": [
            {"reading": "セイブツ", "full_reading": "セイブツ", "sense": "生きているもの・動植物・生物学", "cues": "深海の生物、生物の授業、多様な生物が暮らす森"},
            {"reading": "ナマモノ", "full_reading": "ナマモノ", "sense": "火を通していない生の食品・腐りやすい食材", "cues": "夏場は生物の持ち帰りに注意する、生物が苦手だ"},
        ],
    },
    {
        "surface": "一日",
        "span_surface": "一日",
        "key": "一日",
        "category": "compound",
        "readings": [
            {"reading": "イチニチ", "full_reading": "イチニチ", "sense": "朝か晩までの丸一日・1日間の長さ", "cues": "一日中雨が降る、忙しい一日が終わる、一日一善"},
            {"reading": "ツイタチ", "full_reading": "ツイタチ", "sense": "毎月の最初の日（1日）", "cues": "毎月一日に神社へお参りする、元日の朝、今月の一日"},
        ],
    },
    {
        "surface": "大人",
        "span_surface": "大人",
        "key": "大人",
        "category": "compound",
        "readings": [
            {"reading": "オトナ", "full_reading": "オトナ", "sense": "成長した人・成人・成熟した態度", "cues": "大人になる、大人料金、大人びた話し方"},
            {"reading": "タイジン", "full_reading": "タイジン", "sense": "徳の高い人物・大物・巨人（古典/文語）", "cues": "大人の風格を備える、小人に対して大人と呼ぶ"},
        ],
    },
    {
        "surface": "大事",
        "span_surface": "大事",
        "key": "大事",
        "category": "compound",
        "readings": [
            {"reading": "ダイジ", "full_reading": "ダイジ", "sense": "大切であること・重要であること", "cues": "家族を大事にする、大事なお知らせ、健康が一番大事"},
            {"reading": "オオゴト", "full_reading": "オオゴト", "sense": "大変な出来事・大事件・重大事態", "cues": "放っておくと大事になる、大事には至らなかった"},
        ],
    },
    {
        "surface": "文書",
        "span_surface": "文書",
        "key": "文書",
        "category": "compound",
        "readings": [
            {"reading": "ブンショ", "full_reading": "ブンショ", "sense": "文字で書かれた公的/業務用の書類・テキスト", "cues": "公式文書を作成する、文書で回答する、社内文書"},
            {"reading": "モンジョ", "full_reading": "モンジョ", "sense": "歴史学上の古文書・記録資料", "cues": "中世の古文書を解読する、寺に残る古文書"},
        ],
    },
    {
        "surface": "心中",
        "span_surface": "心中",
        "key": "心中",
        "category": "compound",
        "readings": [
            {"reading": "シンチュウ", "full_reading": "シンチュウ", "sense": "心の中・本心・内心", "cues": "心中を察する、心中穏やかではない、心中を打ち明ける"},
            {"reading": "シンジュウ", "full_reading": "シンジュウ", "sense": "古典芸能や歴史小説での愛の誓い・無理心中など", "cues": "近松門左衛門の心中物、一家心中の悲劇"},
        ],
    },
    {
        "surface": "紅葉",
        "span_surface": "紅葉",
        "key": "紅葉",
        "category": "compound",
        "readings": [
            {"reading": "コウヨウ", "full_reading": "コウヨウ", "sense": "秋に木々の葉が赤や黄色に色づく現象・紅葉狩り", "cues": "山々が鮮やかに紅葉する、紅葉の名所を訪ねる"},
            {"reading": "モミジ", "full_reading": "モミジ", "sense": "カエデ類の赤い葉・植物そのもの・もみじおろし", "cues": "真っ赤な紅葉の葉を拾う、庭園の紅葉が美しい"},
        ],
    },
    {
        "surface": "白髪",
        "span_surface": "白髪",
        "key": "白髪",
        "category": "compound",
        "readings": [
            {"reading": "シラガ", "full_reading": "シラガ", "sense": "白くなった髪の毛（日常表現）", "cues": "鏡を見て白髪が増えたと気づく、白髪染め"},
            {"reading": "ハクハツ", "full_reading": "ハクハツ", "sense": "文語的・漢語的な白い頭髪", "cues": "白髪を蓄えた老賢者、白髪交じりの髭"},
        ],
    },
    {
        "surface": "仮名",
        "span_surface": "仮名",
        "key": "仮名",
        "category": "compound",
        "readings": [
            {"reading": "カナ", "full_reading": "カナ", "sense": "ひらがな・カタカナの総称（仮名文字）", "cues": "仮名遣い、仮名で書く、振り仮名、平仮名"},
            {"reading": "カメイ", "full_reading": "カメイ", "sense": "本名ではなく仮に使う名前（偽名・匿名）", "cues": "仮名で投稿する、仮名を使って申し込む"},
        ],
    },
    {
        "surface": "一言",
        "span_surface": "一言",
        "key": "一言",
        "category": "compound",
        "readings": [
            {"reading": "ヒトコト", "full_reading": "ヒトコト", "sense": "ちょっと発する短い言葉", "cues": "最後に一言だけ挨拶する、一言も話さない、温かい一言"},
            {"reading": "イチゴン", "full_reading": "イチゴン", "sense": "ひと言・文句・口を挟むこと（文語/慣用）", "cues": "一言居士、一言申し上げる、一言の弁解もない"},
        ],
    },
    {
        "surface": "見物",
        "span_surface": "見物",
        "key": "見物",
        "category": "compound",
        "readings": [
            {"reading": "ケンブツ", "full_reading": "ケンブツ", "sense": "名所・祭り・試合などを見て楽しむこと", "cues": "京都の名所を見物する、祭り見物に出かける"},
            {"reading": "ミモノ", "full_reading": "ミモノ", "sense": "見る価値のある面白いもの・注目点", "cues": "決勝戦の対決がどうなるか見物だ、迫力が見物"},
        ],
    },
    {
        "surface": "十分",
        "span_surface": "十分",
        "key": "十分",
        "category": "compound",
        "readings": [
            {"reading": "ジュウブン", "full_reading": "ジュウブン", "sense": "不足がなく満ち足りていること", "cues": "準備は十分だ、十分な睡眠をとる、十分理解した"},
            {"reading": "ジュップン", "full_reading": "ジュップン", "sense": "時間の10分間", "cues": "駅から歩いて十分かかる、あと十分で始まる"},
        ],
    },
    {
        "surface": "一分",
        "span_surface": "一分",
        "key": "一分",
        "category": "compound",
        "readings": [
            {"reading": "イップン", "full_reading": "イップン", "sense": "時間の1分間", "cues": "一分も遅れずに到着する、一分間で説明する"},
            {"reading": "イチブ", "full_reading": "イチブ", "sense": "全体の1割の10分の1・少し・武士などの面目", "cues": "一分の隙もない構え、一分一厘の狂いもない、一分の理"},
        ],
    },
    {
        "surface": "気質",
        "span_surface": "気質",
        "key": "気質",
        "category": "compound",
        "readings": [
            {"reading": "キシツ", "full_reading": "キシツ", "sense": "生まれつきの性質・性格傾向", "cues": "繊細な気質の持ち主、気質が合う"},
            {"reading": "カタギ", "full_reading": "カタギ", "sense": "職業や立場に特有の気風", "cues": "職人気質、昔気質の頑固な親父、学者気質"},
        ],
    },
    {
        "surface": "御前",
        "span_surface": "御前",
        "key": "御前",
        "category": "compound",
        "readings": [
            {"reading": "ゴゼン", "full_reading": "ゴゼン", "sense": "貴人や主君・神仏の前", "cues": "将軍の御前へ罷り出る、御前試合、御前会議"},
            {"reading": "オマエ", "full_reading": "オマエ", "sense": "二人称代名詞（古風/親しい間柄）", "cues": "御前に任せるよ、御前の言う通りだ"},
        ],
    },
    {
        "surface": "重複",
        "span_surface": "重複",
        "key": "重複",
        "category": "compound",
        "readings": [
            {"reading": "チョウフク", "full_reading": "チョウフク", "sense": "同じものが重なり合うこと（本来音）", "cues": "データが重複している、内容の重複を避ける"},
            {"reading": "ジュウフク", "full_reading": "ジュウフク", "sense": "同じものが重なり合うこと（慣用読み）", "cues": "予定が重複する、重複登録を確認する"},
        ],
    },
    {
        "surface": "早急",
        "span_surface": "早急",
        "key": "早急",
        "category": "compound",
        "readings": [
            {"reading": "サッキュウ", "full_reading": "サッキュウ", "sense": "非常に急ぐこと（本来音）", "cues": "早急に対応する、早急な検討が必要だ"},
            {"reading": "ソウキュウ", "full_reading": "ソウキュウ", "sense": "非常に急ぐこと（慣用読み）", "cues": "早急に返事をする、早急な対策"},
        ],
    },
    {
        "surface": "貼付",
        "span_surface": "貼付",
        "key": "貼付",
        "category": "compound",
        "readings": [
            {"reading": "チョウフ", "full_reading": "チョウフ", "sense": "切手や印紙・書類などを貼り付けること（本来）", "cues": "収入印紙を貼付する、履歴書に写真を貼付する"},
            {"reading": "テンプ", "full_reading": "テンプ", "sense": "湿布薬やシール・書類を貼り付けること（慣用）", "cues": "湿布を患部に貼付する、ラベルの貼付"},
        ],
    },
    {
        "surface": "何人",
        "span_surface": "何人",
        "key": "何人",
        "category": "compound",
        "readings": [
            {"reading": "ナンニン", "full_reading": "ナンニン", "sense": "人数を尋ねる表現（何名か）", "cues": "明日の参加者は何人ですか、何人か集まった"},
            {"reading": "ナニビト", "full_reading": "ナニビト", "sense": "「だれ・いかなる人」を意味する文語表現", "cues": "何人も立ち入ることを禁ず、何人たりとも許さない"},
            {"reading": "ナンピト", "full_reading": "ナンピト", "sense": "「だれ・いかなる人」（なにごと・なんぴと）", "cues": "それは何人にも明白な誤りだ、何人たりとも"},
        ],
    },
    {
        "surface": "何時",
        "span_surface": "何時",
        "key": "何時",
        "category": "compound",
        "readings": [
            {"reading": "ナンジ", "full_reading": "ナンジ", "sense": "時刻の何時かを尋ねる表現", "cues": "いま何時ですか、電車は何時に発車しますか"},
            {"reading": "イツ", "full_reading": "イツ", "sense": "どの時点・時期か（漢字表記の「いつ」）", "cues": "何時の間にか眠っていた、何時までも変わらない"},
        ],
    },
    {
        "surface": "何方",
        "span_surface": "何方",
        "key": "何方",
        "category": "compound",
        "readings": [
            {"reading": "ドナタ", "full_reading": "ドナタ", "sense": "「どなた・どの人」の丁寧語", "cues": "何方様でしょうか、何方かご存知ありませんか"},
            {"reading": "ドチラ", "full_reading": "ドチラ", "sense": "方向や選択の「どちら」", "cues": "何方へお出かけですか、何方を選んでもよい"},
        ],
    },
    {
        "surface": "人間",
        "span_surface": "人間",
        "key": "人間",
        "category": "compound",
        "readings": [
            {"reading": "ニンゲン", "full_reading": "ニンゲン", "sense": "ヒト・人類・人の性格", "cues": "人間関係、人間らしい温かさ、人間は考える葦"},
            {"reading": "ジンカン", "full_reading": "ジンカン", "sense": "人の住む世の中・世間（文語/古典）", "cues": "人間到る処青山あり、人間五十年"},
        ],
    },
    {
        "surface": "利益",
        "span_surface": "利益",
        "key": "利益",
        "category": "compound",
        "readings": [
            {"reading": "リエキ", "full_reading": "リエキ", "sense": "商売の儲け・得になること", "cues": "企業の利益が伸びる、消費者の利益を守る"},
            {"reading": "リヤク", "full_reading": "リヤク", "sense": "神仏が人間に与える恵み（ご利益）", "cues": "商売繁盛のご利益がある神社、霊験と利益"},
        ],
    },
    {
        "surface": "最中",
        "span_surface": "最中",
        "key": "最中",
        "category": "compound",
        "readings": [
            {"reading": "サイチュウ", "full_reading": "サイチュウ", "sense": "まさにその物事が行われている真っ最中", "cues": "会議の最中に電話が鳴った、食事の最中"},
            {"reading": "モナカ", "full_reading": "モナカ", "sense": "餡を薄い皮で挟んだ和菓子", "cues": "抹茶と一緒に最中をいただく、老舗の最中"},
        ],
    },
    {
        "surface": "大地震",
        "span_surface": "大地震",
        "key": "大地震",
        "category": "compound",
        "readings": [
            {"reading": "ダイジシン", "full_reading": "ダイジシン", "sense": "規模の大きい地震（標準）", "cues": "大地震に備えて防災訓練を行う"},
            {"reading": "オオジシン", "full_reading": "オオジシン", "sense": "大きな揺れの地震（和語的）", "cues": "突然の大地震で建物が揺れた"},
        ],
    },
    {
        "surface": "大風",
        "span_surface": "大風",
        "key": "大風",
        "category": "compound",
        "readings": [
            {"reading": "オオカゼ", "full_reading": "オオカゼ", "sense": "激しい強い風", "cues": "昨夜の大風で木が倒れた、大風が吹き荒れる"},
            {"reading": "タイフウ", "full_reading": "タイフウ", "sense": "大きな風・大風の漢語読み（古典/漢詩的表現）", "cues": "大風起りて雲飛揚す"},
        ],
    },
    {
        "surface": "大水",
        "span_surface": "大水",
        "key": "大水",
        "category": "compound",
        "readings": [
            {"reading": "オオミズ", "full_reading": "オオミズ", "sense": "川が氾濫するほどの洪水・増水", "cues": "長雨で大水が出て橋が流された"},
            {"reading": "ダイスイ", "full_reading": "ダイスイ", "sense": "豊かな水量・大量の水", "cues": "大水をたたえた湖"},
        ],
    },
    {
        "surface": "家内",
        "span_surface": "家内",
        "key": "家内",
        "category": "compound",
        "readings": [
            {"reading": "カナイ", "full_reading": "カナイ", "sense": "自分の妻、または家族全体（家内安全）", "cues": "家内と相談して決める、家内安全を祈る、家内工業"},
            {"reading": "ヤウチ", "full_reading": "ヤウチ", "sense": "家の内部・一族内（古風）", "cues": "家内の者が誰もいない"},
        ],
    },
    {
        "surface": "手前",
        "span_surface": "手前",
        "key": "手前",
        "category": "compound",
        "readings": [
            {"reading": "テマエ", "full_reading": "テマエ", "sense": "自分に近い側、茶の湯の点前、面目・都合", "cues": "交差点の手前で曲がる、約束した手前断れない"},
            {"reading": "テメエ", "full_reading": "テメエ", "sense": "荒っぽい二人称/一人称の俗語", "cues": "手前のことは手前で片付けろ"},
        ],
    },
    {
        "surface": "月日",
        "span_surface": "月日",
        "key": "月日",
        "category": "compound",
        "readings": [
            {"reading": "ツキヒ", "full_reading": "ツキヒ", "sense": "過ぎていく時間・歳月", "cues": "月日が経つのは早い、長い月日をかけて完成させる"},
            {"reading": "ガッピ", "full_reading": "ガッピ", "sense": "生年月日などの暦上の月と日", "cues": "書類に申請年月日を記入する、生年月日の月日"},
        ],
    },
    {
        "surface": "水面",
        "span_surface": "水面",
        "key": "水面",
        "category": "compound",
        "readings": [
            {"reading": "スイメン", "full_reading": "スイメン", "sense": "水の表面（科学・一般）", "cues": "湖の水面が静かに光る、水面に浮上する、水面下の交渉"},
            {"reading": "ミナモ", "full_reading": "ミナモ", "sense": "詩的・文学的な水の表面", "cues": "月光が水面に揺れる、春の小川の水面"},
        ],
    },
    {
        "surface": "生地",
        "span_surface": "生地",
        "key": "生地",
        "category": "compound",
        "readings": [
            {"reading": "キジ", "full_reading": "キジ", "sense": "布地・パンやケーキのたね・素材のまま", "cues": "パンの生地をこねる、柔らかな綿の生地、生地が出る"},
            {"reading": "セイチ", "full_reading": "セイチ", "sense": "生まれた土地・出身地", "cues": "作家の生地を訪ねる"},
        ],
    },
    {
        "surface": "目下",
        "span_surface": "目下",
        "key": "目下",
        "category": "compound",
        "readings": [
            {"reading": "モッカ", "full_reading": "モッカ", "sense": "ちょうど今・現在ただいま", "cues": "目下調査を進めている、目下の課題"},
            {"reading": "メシタ", "full_reading": "メシタ", "sense": "地位・年齢が自分より下の人", "cues": "目下の者にも丁寧に接する"},
        ],
    },
    {
        "surface": "空手",
        "span_surface": "空手",
        "key": "空手",
        "category": "compound",
        "readings": [
            {"reading": "カラテ", "full_reading": "カラテ", "sense": "武道の空手・手ぶら（からて）", "cues": "空手の道場に通う、空手道"},
            {"reading": "クウシュ", "full_reading": "クウシュ", "sense": "何も手に持っていないこと（徒手空拳）", "cues": "空手で帰るわけにはいかない"},
        ],
    },
    {
        "surface": "色紙",
        "span_surface": "色紙",
        "key": "色紙",
        "category": "compound",
        "readings": [
            {"reading": "シキシ", "full_reading": "シキシ", "sense": "書画やサインを書く厚手の正方形の紙", "cues": "お祝いに寄せ書きの色紙を贈る、サイン色紙"},
            {"reading": "イロガミ", "full_reading": "イロガミ", "sense": "折り紙や工作用の色のついた薄い紙", "cues": "色紙を切って飾りを作る"},
        ],
    },
    {
        "surface": "行方",
        "span_surface": "行方",
        "key": "行方",
        "category": "compound",
        "readings": [
            {"reading": "ユクエ", "full_reading": "ユクエ", "sense": "どこへ行ったかという居場所・今後の成り行き", "cues": "行方がわからない、今後の行方を見守る、行方不明"},
            {"reading": "ギョウホウ", "full_reading": "ギョウホウ", "sense": "仏教の修行の作法", "cues": "古来の行方を守る"},
        ],
    },
    {
        "surface": "表裏",
        "span_surface": "表裏",
        "key": "表裏",
        "category": "compound",
        "readings": [
            {"reading": "ヒョウリ", "full_reading": "ヒョウリ", "sense": "表と裏の関係・本音と建前の二面性", "cues": "表裏一体の関係、表裏のない誠実な人柄"},
            {"reading": "オモテウラ", "full_reading": "オモテウラ", "sense": "物理的な表側と裏側", "cues": "紙の表裏を確認して印刷する、コインの表裏"},
        ],
    },
    {
        "surface": "言質",
        "span_surface": "言質",
        "key": "言質",
        "category": "compound",
        "readings": [
            {"reading": "ゲンチ", "full_reading": "ゲンチ", "sense": "あとで証拠となる相手の発言（本来）", "cues": "相手から言質を取る、言質を与えないよう慎重に話す"},
            {"reading": "ゲンシツ", "full_reading": "ゲンシツ", "sense": "あとで証拠となる発言（慣用読み）", "cues": "交渉で言質を取られた"},
        ],
    },
    {
        "surface": "赤子",
        "span_surface": "赤子",
        "key": "赤子",
        "category": "compound",
        "readings": [
            {"reading": "アカゴ", "full_reading": "アカゴ", "sense": "生まれたばかりの赤ちゃん", "cues": "赤子がすやすや眠っている、赤子の手をひねる"},
            {"reading": "セキシ", "full_reading": "セキシ", "sense": "赤ちゃん・民衆を子に喩える文語", "cues": "赤子の心"},
        ],
    },
    {
        "surface": "追従",
        "span_surface": "追従",
        "key": "追従",
        "category": "compound",
        "readings": [
            {"reading": "ツイジュウ", "full_reading": "ツイジュウ", "sense": "他のあとに付き従うこと・他社に追随", "cues": "他社の価格に追従する、流行に追従する"},
            {"reading": "ツイショウ", "full_reading": "ツイショウ", "sense": "権力者におべっかを使うこと・お世辞", "cues": "上司に追従笑いを浮かべる、追従を言う"},
        ],
    },
    {
        "surface": "金色",
        "span_surface": "金色",
        "key": "金色",
        "category": "compound",
        "readings": [
            {"reading": "キンイロ", "full_reading": "キンイロ", "sense": "金属の金のような明るい黄色", "cues": "金色の折り紙、金色のメダルが輝く"},
            {"reading": "コンジキ", "full_reading": "コンジキ", "sense": "荘厳で輝きのある金色（文学・仏教・稲穂）", "cues": "夕日に照らされた金色の稲穂、金色夜叉"},
        ],
    },
    {
        "surface": "風車",
        "span_surface": "風車",
        "key": "風車",
        "category": "compound",
        "readings": [
            {"reading": "カザグルマ", "full_reading": "カザグルマ", "sense": "風を受けて回る子どものおもちゃ・花の名", "cues": "縁日で買った風車がくるくる回る"},
            {"reading": "フウシャ", "full_reading": "フウシャ", "sense": "風力発電や粉挽き用の大きな風車", "cues": "丘の上に並ぶ巨大な発電用の風車"},
        ],
    },
    {
        "surface": "黄金",
        "span_surface": "黄金",
        "key": "黄金",
        "category": "compound",
        "readings": [
            {"reading": "オウゴン", "full_reading": "オウゴン", "sense": "貴金属のゴールド・価値ある時期（黄金時代）", "cues": "黄金時代を築く、黄金比率、黄金の輝き"},
            {"reading": "コガネ", "full_reading": "コガネ", "sense": "古語・文語での金色・金貨・小判", "cues": "黄金色に実る稲穂、山吹色の黄金"},
        ],
    },
    {
        "surface": "大雪",
        "span_surface": "大雪",
        "key": "大雪",
        "category": "compound",
        "readings": [
            {"reading": "オオユキ", "full_reading": "オオユキ", "sense": "激しくたくさん降る雪", "cues": "大雪警報が発令された、大雪で電車が止まる"},
            {"reading": "タイセツ", "full_reading": "タイセツ", "sense": "二十四節気のひとつ（12月7日頃）", "cues": "暦の上では大雪を過ぎ、冬の寒さが本格化する"},
        ],
    },
    {
        "surface": "相殺",
        "span_surface": "相殺",
        "key": "相殺",
        "category": "compound",
        "readings": [
            {"reading": "ソウサイ", "full_reading": "ソウサイ", "sense": "債権債務やプラスマイナスを帳消しにすること", "cues": "利益と損失を相殺する、借金を相殺する"},
            {"reading": "ソウサツ", "full_reading": "ソウサツ", "sense": "お互いに殺し合うこと・刺し違えること", "cues": "二人の剣士が相殺の構えをとる"},
        ],
    },
    {
        "surface": "小川",
        "span_surface": "小川",
        "key": "小川",
        "category": "compound",
        "readings": [
            {"reading": "オガワ", "full_reading": "オガワ", "sense": "さらさら流れる小さな川（標準）", "cues": "里山の小川のせせらぎを聞く"},
            {"reading": "コガワ", "full_reading": "コガワ", "sense": "小さな川（別読み）", "cues": "野原を横切る細い小川"},
        ],
    },
    {
        "surface": "三月",
        "span_surface": "三月",
        "key": "三月",
        "category": "compound",
        "readings": [
            {"reading": "サンガツ", "full_reading": "サンガツ", "sense": "カレンダーの3番目の月", "cues": "三月に卒業式が行われる、三月末に引っ越す"},
            {"reading": "ミツキ", "full_reading": "ミツキ", "sense": "3ヶ月間の長さ（古風/文語）", "cues": "旅に出てから早くも三月が過ぎた"},
        ],
    },
    {
        "surface": "五月",
        "span_surface": "五月",
        "key": "五月",
        "category": "compound",
        "readings": [
            {"reading": "ゴガツ", "full_reading": "ゴガツ", "sense": "カレンダーの5番目の月", "cues": "五月の大型連休に旅行する"},
            {"reading": "サツキ", "full_reading": "サツキ", "sense": "旧暦の5月・皐月（五月晴れ等）", "cues": "五月晴れの下を鯉のぼりが泳ぐ"},
        ],
    },

    # ==================== 2. 単漢字語 (single_kanji) ====================
    {
        "surface": "生",
        "span_surface": "生",
        "key": "生",
        "category": "single_kanji",
        "readings": [
            {"reading": "ナマ", "full_reading": "ナマ", "sense": "加熱・加工していない状態、ライブ中継", "cues": "生クリーム、生ビール、生の野菜を食べる、生放送"},
            {"reading": "セイ", "full_reading": "セイ", "sense": "いのち・生きること（生と死）", "cues": "生を授かる、生と死を見つめる、生を謳歌する"},
            {"reading": "キ", "full_reading": "キ", "sense": "混じりけのない・そのまま（生糸、生一本）", "cues": "生醤油を少しかける、生のままの素材"},
            {"reading": "ショウ", "full_reading": "ショウ", "sense": "生まれつき・いのちの生涯（一生、生を受ける）", "cues": "この世に生を受ける、生あるもの"},
        ],
    },
    {
        "surface": "行",
        "span_surface": "行",
        "key": "行",
        "category": "single_kanji",
        "readings": [
            {"reading": "ギョウ", "full_reading": "ギョウ", "sense": "文章の横・縦の並び、修行（行を空ける）", "cues": "次の行を読む、一行目の文章、滝に打たれる行"},
            {"reading": "コウ", "full_reading": "コウ", "sense": "旅や移動・銀行・おこない", "cues": "一行が駅に到着する、日米親善の行"},
        ],
    },
    {
        "surface": "間",
        "span_surface": "間",
        "key": "間",
        "category": "single_kanji",
        "readings": [
            {"reading": "アイダ", "full_reading": "アイダ", "sense": "空間・時間のへだたり、あいだ（〜の間）", "cues": "山と山の間を流れる川、待っている間に読む"},
            {"reading": "マ", "full_reading": "マ", "sense": "部屋、タイミング・リズム・余白（間が悪い）", "cues": "タイミングの間が悪い、間を置いて話す、和室の間"},
            {"reading": "カン", "full_reading": "カン", "sense": "時間の長さ・2点間の関係（〜時間、日米間）", "cues": "東京と大阪の間を結ぶ新幹線、三日間の出張"},
        ],
    },
    {
        "surface": "風",
        "span_surface": "風",
        "key": "風",
        "category": "single_kanji",
        "readings": [
            {"reading": "カゼ", "full_reading": "カゼ", "sense": "空気の流れ・そよ風", "cues": "涼しい風が吹く、風に揺れる木の葉"},
            {"reading": "フウ", "full_reading": "フウ", "sense": "様子・やり方・スタイル（〜風、洋風）", "cues": "そんな風に考える、昔風の店構え、和風の味付け"},
        ],
    },
    {
        "surface": "金",
        "span_surface": "金",
        "key": "金",
        "category": "single_kanji",
        "readings": [
            {"reading": "カネ", "full_reading": "カネ", "sense": "お金・金銭・鐘の同音でなく貨幣", "cues": "金を稼ぐ、金がかかる、手元に金がない"},
            {"reading": "キン", "full_reading": "キン", "sense": "貴金属のゴールド・金曜日・金メダル", "cues": "金の指輪、金のメダル、金と銀"},
        ],
    },
    {
        "surface": "方",
        "span_surface": "方",
        "key": "方",
        "category": "single_kanji",
        "readings": [
            {"reading": "ホウ", "full_reading": "ホウ", "sense": "方向・比較の一方（こちらの方、行く方がよい）", "cues": "東の方へ歩く、早く寝た方がいい、こちらの方"},
            {"reading": "カタ", "full_reading": "カタ", "sense": "人の丁寧な言い方・やり方（あの方、作り方）", "cues": "とても親切な方です、初めての方へ"},
        ],
    },
    {
        "surface": "角",
        "span_surface": "角",
        "key": "角",
        "category": "single_kanji",
        "readings": [
            {"reading": "カド", "full_reading": "カド", "sense": "道路や物のとがった端（曲がり角）", "cues": "次の角を右へ曲がる、机の角にぶつける、角が立つ"},
            {"reading": "ツノ", "full_reading": "ツノ", "sense": "鹿や牛などの頭から生えた角", "cues": "立派な鹿の角が生えている、牛の角"},
            {"reading": "カク", "full_reading": "カク", "sense": "将棋の角行・角度・四角", "cues": "将棋で角を打つ、角度の角"},
        ],
    },
    {
        "surface": "額",
        "span_surface": "額",
        "key": "額",
        "category": "single_kanji",
        "readings": [
            {"reading": "ガク", "full_reading": "ガク", "sense": "金額の量・絵を入れる額縁", "cues": "大きな額の寄付、壁に掛かった額を眺める、予算額"},
            {"reading": "ヒタイ", "full_reading": "ヒタイ", "sense": "人間の顔のおでこ", "cues": "額の汗をハンカチで拭く、額に手を当てて熱を測る"},
        ],
    },
    {
        "surface": "表",
        "span_surface": "表",
        "key": "表",
        "category": "single_kanji",
        "readings": [
            {"reading": "オモテ", "full_reading": "オモテ", "sense": "裏の反対側・家の外", "cues": "紙の表に名前を書く、表へ出る、裏と表"},
            {"reading": "ヒョウ", "full_reading": "ヒョウ", "sense": "データを並べた一覧表（グラフと表）", "cues": "調査結果を表にまとめる、表と図を比べる"},
        ],
    },
    {
        "surface": "主",
        "span_surface": "主",
        "key": "主",
        "category": "single_kanji",
        "readings": [
            {"reading": "オモ", "full_reading": "オモ", "sense": "中心・主要であること（主な理由）", "cues": "主な原因は疲労だ、主に週末に活動する"},
            {"reading": "ヌシ", "full_reading": "ヌシ", "sense": "持ち主・古い主（飼い主・沼の主）", "cues": "この家の主が現れる、池の主と呼ばれる大魚"},
            {"reading": "シュ", "full_reading": "シュ", "sense": "主体・中心となるもの（主として・主人）", "cues": "安全を主として考える、主と従の関係"},
        ],
    },
    {
        "surface": "空",
        "span_surface": "空",
        "key": "空",
        "category": "single_kanji",
        "readings": [
            {"reading": "ソラ", "full_reading": "ソラ", "sense": "見上げた青空・天気の空・記憶（空で覚える）", "cues": "青い空に雲が浮かぶ、空を見上げる"},
            {"reading": "カラ", "full_reading": "カラ", "sense": "中身が空っぽの状態", "cues": "箱の中が空だった、空のコップを片付ける"},
        ],
    },
    {
        "surface": "音",
        "span_surface": "音",
        "key": "音",
        "category": "single_kanji",
        "readings": [
            {"reading": "オト", "full_reading": "オト", "sense": "耳に聞こえる物理的な物音・音楽", "cues": "大きな音が響く、波の音を聞く"},
            {"reading": "ネ", "full_reading": "ネ", "sense": "楽器や鳥などの美しい響き（音色、弱音）", "cues": "鈴の音が涼やかに鳴る、悲しげな笛の音"},
            {"reading": "オン", "full_reading": "オン", "sense": "漢字の音読み・音声・音節", "cues": "漢字の音と訓、一つの音を発音する"},
        ],
    },
    {
        "surface": "柄",
        "span_surface": "柄",
        "key": "柄",
        "category": "single_kanji",
        "readings": [
            {"reading": "ガラ", "full_reading": "ガラ", "sense": "布の模様・デザイン・体格や人柄", "cues": "花柄ではなく落ち着いた柄を選ぶ、柄が悪い"},
            {"reading": "エ", "full_reading": "エ", "sense": "包丁・フライパン・傘などの握る棒部分", "cues": "包丁の柄をしっかりと握る、傘の柄が折れた"},
        ],
    },
    {
        "surface": "素",
        "span_surface": "素",
        "key": "素",
        "category": "single_kanji",
        "readings": [
            {"reading": "ス", "full_reading": "ス", "sense": "飾らないそのままの状態（素のまま）", "cues": "素の自分を出す、素で驚く"},
            {"reading": "モト", "full_reading": "モト", "sense": "料理や成分のベース・原料（鍋の素）", "cues": "スープの素をお湯に溶かす、料理の素を使う"},
        ],
    },
    {
        "surface": "家",
        "span_surface": "家",
        "key": "家",
        "category": "single_kanji",
        "readings": [
            {"reading": "イエ", "full_reading": "イエ", "sense": "建物としての住宅・住まい", "cues": "新しい家を建てる、駅から家まで歩く"},
            {"reading": "ウチ", "full_reading": "ウチ", "sense": "家庭・わが家・親しみのある自宅", "cues": "早く家へ帰ろう、家の犬が待っている"},
        ],
    },
    {
        "surface": "私",
        "span_surface": "私",
        "key": "私",
        "category": "single_kanji",
        "readings": [
            {"reading": "ワタシ", "full_reading": "ワタシ", "sense": "一般的な一人称", "cues": "私は毎朝散歩をします、私にお任せください"},
            {"reading": "ワタクシ", "full_reading": "ワタクシ", "sense": "改まった一人称・公と私（プライベート）", "cues": "公と私を区別する、私事で恐縮ですが"},
        ],
    },
    {
        "surface": "頭",
        "span_surface": "頭",
        "key": "頭",
        "category": "single_kanji",
        "readings": [
            {"reading": "アタマ", "full_reading": "アタマ", "sense": "身体の頭部・知能・先頭", "cues": "頭が痛い、頭を使って考える、列の頭"},
            {"reading": "トウ", "full_reading": "トウ", "sense": "牛や馬などの助数詞（〜頭）・頭領", "cues": "牧場に牛が三頭いる、群れの頭として動く"},
        ],
    },
    {
        "surface": "星",
        "span_surface": "星",
        "key": "星",
        "category": "single_kanji",
        "readings": [
            {"reading": "ホシ", "full_reading": "ホシ", "sense": "夜空に光る星・マーク", "cues": "夜空に美しい星が輝く、星を見上げる"},
            {"reading": "セイ", "full_reading": "セイ", "sense": "惑星などの天体・星座", "cues": "地球から最も近い星である太陽"},
        ],
    },
    {
        "surface": "体",
        "span_surface": "体",
        "key": "体",
        "category": "single_kanji",
        "readings": [
            {"reading": "カラダ", "full_reading": "カラダ", "sense": "人間や動物の肉体・健康", "cues": "体を動かして健康を保つ、体が温まる"},
            {"reading": "タイ", "full_reading": "タイ", "sense": "仏像・石像を数える助数詞・形や様子", "cues": "石仏が一体安置されている、体よく断る"},
        ],
    },
    {
        "surface": "中",
        "span_surface": "中",
        "key": "中",
        "category": "single_kanji",
        "readings": [
            {"reading": "ナカ", "full_reading": "ナカ", "sense": "内部・内側（箱の中・家の中）", "cues": "箱の中を確認する、部屋の中で待つ"},
            {"reading": "チュウ", "full_reading": "チュウ", "sense": "最中・程度（会議中・大中小）", "cues": "ただいま会議中です、三段階の中を選ぶ"},
            {"reading": "ジュウ", "full_reading": "ジュウ", "sense": "全体・ずっと（一日中・世界中）", "cues": "世界中を旅する、一日中雨が降る"},
        ],
    },
    {
        "surface": "後",
        "span_surface": "後",
        "key": "後",
        "category": "single_kanji",
        "readings": [
            {"reading": "アト", "full_reading": "アト", "sense": "順番や時間・位置のうしろ（その後・後ろ）", "cues": "仕事の後に食事をする、後を頼む"},
            {"reading": "ノチ", "full_reading": "ノチ", "sense": "それからずっと後の将来・後日", "cues": "後の世まで語り継がれる、雨が降り後に晴れる"},
            {"reading": "ゴ", "full_reading": "ゴ", "sense": "ある時点よりあと（〜年後・食後）", "cues": "三日後に連絡する、会議の後に提出する"},
        ],
    },
    {
        "surface": "前",
        "span_surface": "前",
        "key": "前",
        "category": "single_kanji",
        "readings": [
            {"reading": "マエ", "full_reading": "マエ", "sense": "正面・時間的に先（家の前・三日前）", "cues": "駅の前に集合する、約束の時間の前に着く"},
            {"reading": "ゼン", "full_reading": "ゼン", "sense": "前半・直前・前任（前の〜）", "cues": "前と後ろ、前大統領、午前と午後"},
        ],
    },
    {
        "surface": "上",
        "span_surface": "上",
        "key": "上",
        "category": "single_kanji",
        "readings": [
            {"reading": "ウエ", "full_reading": "ウエ", "sense": "高い位置・表面（机の上・年上）", "cues": "机の上に本を置く、上に登る"},
            {"reading": "ジョウ", "full_reading": "ジョウ", "sense": "上部・巻の上・〜の点において", "cues": "業務上の連絡、上巻と下巻"},
        ],
    },
    {
        "surface": "下",
        "span_surface": "下",
        "key": "下",
        "category": "single_kanji",
        "readings": [
            {"reading": "シタ", "full_reading": "シタ", "sense": "低い位置・下側（木の下・年下）", "cues": "木の下で休む、階段の下で待つ"},
            {"reading": "カ", "full_reading": "カ", "sense": "下の巻・影響下・部下", "cues": "指導の下で研究する、上中下の段階"},
            {"reading": "ゲ", "full_reading": "ゲ", "sense": "下巻・品等が劣ること（下の下）", "cues": "物語の下巻を読む、上の下から下の上"},
        ],
    },
    {
        "surface": "内",
        "span_surface": "内",
        "key": "内",
        "category": "single_kanji",
        "readings": [
            {"reading": "ウチ", "full_reading": "ウチ", "sense": "範囲の中・その期間の間（今のうち）", "cues": "明るいうちに帰る、その内わかる"},
            {"reading": "ナイ", "full_reading": "ナイ", "sense": "組織や範囲の内部（〜以内・国内外）", "cues": "規定の範囲内で処理する、城の内と外"},
        ],
    },
    {
        "surface": "外",
        "span_surface": "外",
        "key": "外",
        "category": "single_kanji",
        "readings": [
            {"reading": "ソト", "full_reading": "ソト", "sense": "建物や枠の外側・屋外", "cues": "天気が良いので外へ出る、窓の外を見る"},
            {"reading": "ガイ", "full_reading": "ガイ", "sense": "領域・想定の範囲外（対象外・国外）", "cues": "想定外の出来事、専門の外になる"},
            {"reading": "ホカ", "full_reading": "ホカ", "sense": "それ以外・別（この外に、思いの外）", "cues": "これより外に方法がない、予想の外に良い結果"},
        ],
    },
    {
        "surface": "先",
        "span_surface": "先",
        "key": "先",
        "category": "single_kanji",
        "readings": [
            {"reading": "サキ", "full_reading": "サキ", "sense": "前方・先端・将来（この先・鉛筆の先）", "cues": "この先の交差点を曲がる、先の予定"},
            {"reading": "セン", "full_reading": "セン", "sense": "さきほど・以前（先ほど・先月）", "cues": "先の試合で優勝した、先を越される"},
        ],
    },
    {
        "surface": "本",
        "span_surface": "本",
        "key": "本",
        "category": "single_kanji",
        "readings": [
            {"reading": "ホン", "full_reading": "ホン", "sense": "書籍・本来の・細長い物の助数詞", "cues": "図書館で本を借りる、鉛筆を二本買う"},
            {"reading": "モト", "full_reading": "モト", "sense": "根本・もと（本を正せば・木の本）", "cues": "本を正せば同じ起源だ、本も子もない"},
        ],
    },
    {
        "surface": "気",
        "span_surface": "気",
        "key": "気",
        "category": "single_kanji",
        "readings": [
            {"reading": "キ", "full_reading": "キ", "sense": "気持ち・精神・空気（気をつける）", "cues": "気が進まない、気をつけて帰る"},
            {"reading": "ケ", "full_reading": "ケ", "sense": "気配・様子・兆候（嫌な気・塩気）", "cues": "風邪の気がある、人の気配を少しも感じない"},
        ],
    },
    {
        "surface": "力",
        "span_surface": "力",
        "key": "力",
        "category": "single_kanji",
        "readings": [
            {"reading": "チカラ", "full_reading": "チカラ", "sense": "腕力・能力・助け（力を合わせる）", "cues": "みんなで力を合わせる、全力の力を使う"},
            {"reading": "リョク", "full_reading": "リョク", "sense": "物理的な力・能力（力の単位・磁力）", "cues": "言葉の力が強い、電気の力"},
        ],
    },
    {
        "surface": "世",
        "span_surface": "世",
        "key": "世",
        "category": "single_kanji",
        "readings": [
            {"reading": "ヨ", "full_reading": "ヨ", "sense": "世の中・社会・時代（世のため）", "cues": "世のため人のため、世の中が変わる"},
            {"reading": "セ", "full_reading": "セ", "sense": "時代・世間・世代（〜世）", "cues": "現世と来世、三世にわたる"},
        ],
    },
    {
        "surface": "代",
        "span_surface": "代",
        "key": "代",
        "category": "single_kanji",
        "readings": [
            {"reading": "ダイ", "full_reading": "ダイ", "sense": "世代・時代・料金（交通費代・三十代）", "cues": "親の代から続く店、昼食代を払う"},
            {"reading": "ヨ", "full_reading": "ヨ", "sense": "治世・時代（御代・神の代）", "cues": "平和な代が長く続く"},
            {"reading": "シロ", "full_reading": "シロ", "sense": "代わり・代金・材料（代物・のり代）", "cues": "糊の代を残して切る、珍しい代だ"},
        ],
    },
    {
        "surface": "度",
        "span_surface": "度",
        "key": "度",
        "category": "single_kanji",
        "readings": [
            {"reading": "ド", "full_reading": "ド", "sense": "回数・角度・温度・程度（一度・三十度）", "cues": "もう一度確認する、度を超える"},
            {"reading": "タビ", "full_reading": "タビ", "sense": "毎回そのときどき（会う度・来る度）", "cues": "春が来る度に思い出す、失敗する度に学ぶ"},
        ],
    },
    {
        "surface": "何",
        "span_surface": "何",
        "key": "何",
        "category": "single_kanji",
        "readings": [
            {"reading": "ナニ", "full_reading": "ナニ", "sense": "物事や理由を問う疑問詞（何がある・何か）", "cues": "何を食べたいですか、何か手伝いましょうか"},
            {"reading": "ナン", "full_reading": "ナン", "sense": "助数詞や「だ・です・の」の前の疑問詞（何ですか・何個）", "cues": "これは何ですか、全部で何個ありますか"},
        ],
    },
    {
        "surface": "月",
        "span_surface": "月",
        "key": "月",
        "category": "single_kanji",
        "readings": [
            {"reading": "ツキ", "full_reading": "ツキ", "sense": "天体の月・1ヶ月間（月を見る・月一回）", "cues": "夜空に満月と美しい月が出ている、月に一度"},
            {"reading": "ガツ", "full_reading": "ガツ", "sense": "暦の何月（四月・今月でなく暦月名）", "cues": "十月と十一月の予定を調べる"},
            {"reading": "ゲツ", "full_reading": "ゲツ", "sense": "月数・月曜日（来月・何ヶ月）", "cues": "三ヶ月前に始めた、月と火の曜日"},
        ],
    },
    {
        "surface": "日",
        "span_surface": "日",
        "key": "日",
        "category": "single_kanji",
        "readings": [
            {"reading": "ヒ", "full_reading": "ヒ", "sense": "太陽・日差し・特定の日（日が沈む・良い日）", "cues": "暖かい日が差す、思い出の日になる"},
            {"reading": "ニチ", "full_reading": "ニチ", "sense": "日数・日付の助数詞（三日・日曜日）", "cues": "予定日を十日に設定する、土と日"},
            {"reading": "カ", "full_reading": "カ", "sense": "2日〜10日などの和語日数助数詞", "cues": "五日から七日にかけて旅行する"},
        ],
    },

    # ==================== 3. 用言・活用語 (predicate) ====================
    {
        "surface": "行った",
        "span_surface": "行",
        "key": "行+った",
        "category": "predicate",
        "readings": [
            {"reading": "イ", "full_reading": "イッタ", "sense": "場所へ移動した（「行く」の過去形）", "cues": "友達と海へ行った、図書館へ行った帰り"},
            {"reading": "オコナ", "full_reading": "オコナッタ", "sense": "実施した・実行した（「行う」の過去形）", "cues": "実験を行った、会議で調査を行った、式典を行った"},
        ],
    },
    {
        "surface": "行く",
        "span_surface": "行",
        "key": "行+く",
        "category": "predicate",
        "readings": [
            {"reading": "イ", "full_reading": "イク", "sense": "目的地へ移動する（日常・標準）", "cues": "明日は病院へ行く、仕事に行く準備"},
            {"reading": "ユ", "full_reading": "ユク", "sense": "過ぎ去る・旅立つ・進む（文語的・詩的）", "cues": "行く春を惜しむ、前途をゆく旅人、流れ行く雲"},
        ],
    },
    {
        "surface": "開く",
        "span_surface": "開",
        "key": "開+く",
        "category": "predicate",
        "readings": [
            {"reading": "ヒラ", "full_reading": "ヒラク", "sense": "本・会議・店・傘などをオープンする・咲く", "cues": "新しい店を開く、本を開く、会議を開く、桜が開く"},
            {"reading": "ア", "full_reading": "アク", "sense": "閉じていたドアや蓋や穴が自然に開く・隙間ができる", "cues": "ドアがゆっくり開く、次の電車まで時間が開く、穴が開く"},
        ],
    },
    {
        "surface": "開いた",
        "span_surface": "開",
        "key": "開+いた",
        "category": "predicate",
        "readings": [
            {"reading": "ヒラ", "full_reading": "ヒライタ", "sense": "開催した・広げた・花が咲いた", "cues": "ノートを開いた、相談会を開いた、花が咲いて開いた"},
            {"reading": "ア", "full_reading": "アイタ", "sense": "ドアがオープンした・差がついた", "cues": "自動ドアが開いた、先頭との差が大きく開いた"},
        ],
    },
    {
        "surface": "空く",
        "span_surface": "空",
        "key": "空+く",
        "category": "predicate",
        "readings": [
            {"reading": "ア", "full_reading": "アク", "sense": "席や予定に空きができる・中が空っぽになる", "cues": "席が空くのを待つ、今週の予定が空く、空き箱"},
            {"reading": "ス", "full_reading": "スク", "sense": "お腹が減る・混雑が減って閑散とする", "cues": "昼前でお腹が空く、夕方になって道路が空く、電車が空く"},
        ],
    },
    {
        "surface": "空いた",
        "span_surface": "空",
        "key": "空+いた",
        "category": "predicate",
        "readings": [
            {"reading": "ア", "full_reading": "アイタ", "sense": "空き時間ができた・席が空席になった", "cues": "空いた時間に読書をする、ちょうど席が空いた"},
            {"reading": "ス", "full_reading": "スイタ", "sense": "空腹になった・人通りが少なくなった", "cues": "すごくお腹が空いた、雨で遊園地が空いた"},
        ],
    },
    {
        "surface": "止める",
        "span_surface": "止",
        "key": "止+める",
        "category": "predicate",
        "readings": [
            {"reading": "ト", "full_reading": "トメル", "sense": "動いている車や水や時計をストップさせる・固定する", "cues": "車を駐車場に止める、出血を止める、ボタンで留め止める"},
            {"reading": "ヤ", "full_reading": "ヤメル", "sense": "習慣・仕事・計画を中止・断念する", "cues": "無理な残業を止める、甘い物を食べるのを止める"},
        ],
    },
    {
        "surface": "止めた",
        "span_surface": "止",
        "key": "止+めた",
        "category": "predicate",
        "readings": [
            {"reading": "ト", "full_reading": "トメタ", "sense": "動きを停止させた", "cues": "玄関の前に自転車を止めた、時計を止めた"},
            {"reading": "ヤ", "full_reading": "ヤメタ", "sense": "行為や計画を中止した", "cues": "結局その旅行へ行くのを止めた、会社を止めた"},
        ],
    },
    {
        "surface": "辛い",
        "span_surface": "辛",
        "key": "辛+い",
        "category": "predicate",
        "readings": [
            {"reading": "カラ", "full_reading": "カライ", "sense": "唐辛子などスパイシーな味・塩気・点数が厳しい", "cues": "このカレーはとても辛い、採点が辛い先生"},
            {"reading": "ツラ", "full_reading": "ツライ", "sense": "精神的・肉体的に苦しく耐え難い", "cues": "別れの瞬間は本当に辛い、冬の朝の練習が辛い"},
        ],
    },
    {
        "surface": "弾く",
        "span_surface": "弾",
        "key": "弾+く",
        "category": "predicate",
        "readings": [
            {"reading": "ヒ", "full_reading": "ヒク", "sense": "ピアノやギターなどの楽器を演奏する", "cues": "ピアノを弾く、バイオリンを美しく弾く"},
            {"reading": "ハジ", "full_reading": "ハジク", "sense": "指ではね飛ばす・水を跳ね返す・そろばん", "cues": "おはじきを指で弾く、撥水コートが雨を弾く、電卓やそろばんを弾く"},
        ],
    },
    {
        "surface": "弾いた",
        "span_surface": "弾",
        "key": "弾+いた",
        "category": "predicate",
        "readings": [
            {"reading": "ヒ", "full_reading": "ヒイタ", "sense": "楽器を演奏した", "cues": "発表会でギターを弾いた"},
            {"reading": "ハジ", "full_reading": "ハジイタ", "sense": "指ではね飛ばした・水を跳ね返した", "cues": "傘が雨粒をきれいに弾いた、コインを指で弾いた"},
        ],
    },
    {
        "surface": "避ける",
        "span_surface": "避",
        "key": "避+ける",
        "category": "predicate",
        "readings": [
            {"reading": "サ", "full_reading": "サケル", "sense": "問題・危険・渋滞などを回避する（抽象/一般）", "cues": "渋滞を避けるため早朝に出発する、衝突を避ける"},
            {"reading": "ヨ", "full_reading": "ヨケル", "sense": "飛んできたボールや水たまりなどを身をかわしてよける", "cues": "走ってくる自転車をさっと避ける、水たまりを避ける"},
        ],
    },
    {
        "surface": "解く",
        "span_surface": "解",
        "key": "解+く",
        "category": "predicate",
        "readings": [
            {"reading": "ト", "full_reading": "トク", "sense": "問題・謎を解明する、緊張や誤解を和らげる", "cues": "数学の難問を解く、誤解を解く、警戒を解く"},
            {"reading": "ホド", "full_reading": "ホドク", "sense": "結んだ紐や包帯・帯をほどく", "cues": "きつく結んだ靴紐を解く、荷物の縄を解く"},
        ],
    },
    {
        "surface": "被る",
        "span_surface": "被",
        "key": "被+る",
        "category": "predicate",
        "readings": [
            {"reading": "カブ", "full_reading": "カブル", "sense": "帽子や頭から覆いをかぶる・水をかぶる・重複する", "cues": "日差しを避けて帽子を被る、予定が被る、ほこりを被る"},
            {"reading": "コウム", "full_reading": "コウムル", "sense": "損害や被害・恩恵を受ける", "cues": "台風で大きな被害を被る、多大なるご迷惑を被る"},
        ],
    },
    {
        "surface": "潜る",
        "span_surface": "潜",
        "key": "潜+る",
        "category": "predicate",
        "readings": [
            {"reading": "モグ", "full_reading": "モグル", "sense": "水や布団の下へ沈み込む・地下に隠れる", "cues": "プールの中へ深く潜る、寒いので布団に潜る"},
            {"reading": "クグ", "full_reading": "クグル", "sense": "門・のれん・危険な網をくぐり抜ける", "cues": "神社の鳥居を潜る、店ののれんを潜る、法網を潜る"},
        ],
    },
    {
        "surface": "退く",
        "span_surface": "退",
        "key": "退+く",
        "category": "predicate",
        "readings": [
            {"reading": "シリゾ", "full_reading": "シリゾク", "sense": "後ろへ下がる・役職から身を引く", "cues": "一歩後ろへ退く、第一線から退く"},
            {"reading": "ヒ", "full_reading": "ヒク", "sense": "部隊や人・潮・腫れが引き下がる", "cues": "相手が思わず一歩退く、腫れが退く"},
            {"reading": "ノ", "full_reading": "ノク", "sense": "邪魔にならない場所へよける（口語）", "cues": "通路を塞がないように横へ退く"},
        ],
    },
    {
        "surface": "強い",
        "span_surface": "強",
        "key": "強+い",
        "category": "predicate",
        "readings": [
            {"reading": "ツヨ", "full_reading": "ツヨイ", "sense": "力・勢い・意志が大きい（ストロング）", "cues": "今日は風が強い、責任感が強い、強い選手"},
            {"reading": "コワ", "full_reading": "コワイ", "sense": "硬くて柔軟性がない（こわい飯・髪の毛がごわごわ）", "cues": "少し強めに炊き上がったご飯、強い毛質"},
        ],
    },
    {
        "surface": "溢れる",
        "span_surface": "溢",
        "key": "溢+れる",
        "category": "predicate",
        "readings": [
            {"reading": "アフ", "full_reading": "アフレル", "sense": "水や感情・魅力がいっぱいになり外へ流れ出る", "cues": "コップから水が溢れる、笑顔が溢れる会場"},
            {"reading": "コボ", "full_reading": "コボレル", "sense": "容器や手から水や涙・粒がこぼれ落ちる", "cues": "思わず涙が頬に溢れる、お茶が少し溢れる"},
        ],
    },
    {
        "surface": "逸れる",
        "span_surface": "逸",
        "key": "逸+れる",
        "category": "predicate",
        "readings": [
            {"reading": "ソ", "full_reading": "ソレル", "sense": "予定のコースや話題の本筋から外れる", "cues": "話題が本筋から逸れる、ボールが的から逸れる"},
            {"reading": "ハグ", "full_reading": "ハグレル", "sense": "人混みなどで連れを見失う・離れる", "cues": "人混みで友人とはぐれて逸れる、親鳥から逸れる"},
        ],
    },
    {
        "surface": "良い",
        "span_surface": "良",
        "key": "良+い",
        "category": "predicate",
        "readings": [
            {"reading": "ヨ", "full_reading": "ヨイ", "sense": "優れている・適切だ（やや改まった/文語寄り）", "cues": "非常に良い結果だった、良き友に恵まれる"},
            {"reading": "イ", "full_reading": "イイ", "sense": "優れている・大丈夫（口語的）", "cues": "それでいいよ、今日は気分が良い"},
        ],
    },
]

# ==================== 4. パイロット低頻度一般多読語 (20語) ====================
LOW_FREQ_GENERAL_HOMOGRAPHS: List[Dict] = [
    {
        "surface": "他",
        "span_surface": "他",
        "key": "他",
        "category": "low_freq_general",
        "readings": [
            {"reading": "ホカ", "full_reading": "ホカ", "sense": "それ以外・別のもの（〜の他に、他の人）", "cues": "他の人と相談する、他に選択肢がない"},
            {"reading": "タ", "full_reading": "タ", "sense": "他者・自と他（他を圧する、自他共に）", "cues": "自他共に認める実力、他を思いやる"},
        ],
    },
    {
        "surface": "翁",
        "span_surface": "翁",
        "key": "翁",
        "category": "low_freq_general",
        "readings": [
            {"reading": "オキナ", "full_reading": "オキナ", "sense": "年老いた男性の雅語・民話の老人", "cues": "竹取の翁が光る竹を見つける、白髪の翁"},
            {"reading": "オウ", "full_reading": "オウ", "sense": "名士・老先生への尊称（〜翁）", "cues": "渋沢栄一に倣う名士として翁と呼ばれる、先代の翁"},
        ],
    },
    {
        "surface": "米",
        "span_surface": "米",
        "key": "米",
        "category": "low_freq_general",
        "readings": [
            {"reading": "コメ", "full_reading": "コメ", "sense": "食用の稲の実・お米・ごはん", "cues": "新米の米を炊く、毎日米を食べる"},
            {"reading": "ベイ", "full_reading": "ベイ", "sense": "アメリカの略称・米軍（米大統領等）", "cues": "日米の外交関係、米国の経済指標"},
        ],
    },
    {
        "surface": "市",
        "span_surface": "市",
        "key": "市",
        "category": "low_freq_general",
        "readings": [
            {"reading": "シ", "full_reading": "シ", "sense": "地方自治体の都市（〜市、市の中心）", "cues": "市の中心部に図書館が建つ、隣りの市へ引っ越す"},
            {"reading": "イチ", "full_reading": "イチ", "sense": "人々が集まって物を売買する定期市・市場", "cues": "朝早くから賑やかな市が立つ、古本の市を歩く"},
        ],
    },
    {
        "surface": "傍",
        "span_surface": "傍",
        "key": "傍",
        "category": "low_freq_general",
        "readings": [
            {"reading": "ソバ", "full_reading": "ソバ", "sense": "すぐ近く・隣（傍に座る、傍を離れない）", "cues": "母親の傍を離れない、机の傍に置く"},
            {"reading": "カタワラ", "full_reading": "カタワラ", "sense": "すぐ脇・〜の合間に（仕事の傍ら）", "cues": "本業の傍ら小説を書く、道の傍に咲く花"},
        ],
    },
    {
        "surface": "牧場",
        "span_surface": "牧場",
        "key": "牧場",
        "category": "low_freq_general",
        "readings": [
            {"reading": "ボクジョウ", "full_reading": "ボクジョウ", "sense": "牛や馬などを飼育する施設（一般・観光）", "cues": "高原の牧場でソフトクリームを食べる、乳牛を育てる牧場"},
            {"reading": "マキバ", "full_reading": "マキバ", "sense": "放牧された草原・牧草地（詩的・和語）", "cues": "緑豊かな牧場に馬が駆ける、夕暮れの牧場"},
        ],
    },
    {
        "surface": "念仏",
        "span_surface": "念仏",
        "key": "念仏",
        "category": "low_freq_general",
        "readings": [
            {"reading": "ネンブツ", "full_reading": "ネンブツ", "sense": "仏教の南無阿弥陀仏・お経を唱えること", "cues": "静かに念仏を唱える、馬の耳に念仏"},
            {"reading": "ネブツ", "full_reading": "ネブツ", "sense": "念仏の古風・民間音変化（大念仏・寒念仏等）", "cues": "冬の夜に寒念仏の声が高らかに響く"},
        ],
    },
    {
        "surface": "外面",
        "span_surface": "外面",
        "key": "外面",
        "category": "low_freq_general",
        "readings": [
            {"reading": "ガイメン", "full_reading": "ガイメン", "sense": "物の外側の表面（内面に対する外面）", "cues": "建物の外面を塗り替える、内面と外面の違い"},
            {"reading": "ソトヅラ", "full_reading": "ソトヅラ", "sense": "他人に対する表面上の愛想や振る舞い", "cues": "あの人は外面だけは良いが家では厳しい"},
        ],
    },
    {
        "surface": "憤怒",
        "span_surface": "憤怒",
        "key": "憤怒",
        "category": "low_freq_general",
        "readings": [
            {"reading": "フンヌ", "full_reading": "フンヌ", "sense": "激しい怒り・仏像の憤怒相", "cues": "憤怒の形相で睨みつける、不動明王の憤怒"},
            {"reading": "フンド", "full_reading": "フンド", "sense": "激しい怒り（漢音読み・慣用）", "cues": "不当な扱いに憤怒を覚える"},
        ],
    },
    {
        "surface": "歩",
        "span_surface": "歩",
        "key": "歩",
        "category": "low_freq_general",
        "readings": [
            {"reading": "ホ", "full_reading": "ホ", "sense": "歩く歩数・歩み（一歩前に出る）", "cues": "歩を進める、一歩ずつ確実に歩く"},
            {"reading": "フ", "full_reading": "フ", "sense": "将棋の駒の歩兵", "cues": "将棋で歩を突く、歩が成ってと金になる"},
        ],
    },
    {
        "surface": "大金",
        "span_surface": "大金",
        "key": "大金",
        "category": "low_freq_general",
        "readings": [
            {"reading": "タイキン", "full_reading": "タイキン", "sense": "非常に高額なお金（多額の資金）", "cues": "大金をはたいて家を買う、大金を稼ぐ"},
            {"reading": "オオガネ", "full_reading": "オオガネ", "sense": "大きな金額のお金・大金持ちの金（和語調）", "cues": "思いがけず大金が転がり込む、大金を掴む"},
        ],
    },
    {
        "surface": "上物",
        "span_surface": "上物",
        "key": "上物",
        "category": "low_freq_general",
        "readings": [
            {"reading": "ジョウモノ", "full_reading": "ジョウモノ", "sense": "品質の非常に良い品物・極上品", "cues": "今日のマグロはなかなかの玉揃いで上物だ"},
            {"reading": "ウワモノ", "full_reading": "ウワモノ", "sense": "土地の上の建物・料理の上に乗せる具材", "cues": "土地はそのままに上物だけを取り壊す、うどんの上物"},
        ],
    },
    {
        "surface": "昨夜",
        "span_surface": "昨夜",
        "key": "昨夜",
        "category": "low_freq_general",
        "readings": [
            {"reading": "サクヤ", "full_reading": "サクヤ", "sense": "昨日の夜（ニュース・書面・硬め）", "cues": "昨夜遅くに連絡が入った、昨夜の雨"},
            {"reading": "ユウベ", "full_reading": "ユウベ", "sense": "昨日の晩（日常会話・口語）", "cues": "昨夜はぐっすり眠れた、昨夜見た夢の話"},
        ],
    },
    {
        "surface": "続柄",
        "span_surface": "続柄",
        "key": "続柄",
        "category": "low_freq_general",
        "readings": [
            {"reading": "ツヅキガラ", "full_reading": "ツヅキガラ", "sense": "親族・家族同士の血縁関係（本来）", "cues": "書類の続柄欄に記入する、世帯主との続柄"},
            {"reading": "ゾクガラ", "full_reading": "ゾクガラ", "sense": "親族間の関係（慣用読み）", "cues": "申込書に本人との続柄を書く"},
        ],
    },
    {
        "surface": "豚肉",
        "span_surface": "豚肉",
        "key": "豚肉",
        "category": "low_freq_general",
        "readings": [
            {"reading": "ブタニク", "full_reading": "ブタニク", "sense": "豚の食用肉（日常料理・標準）", "cues": "スーパーで豚肉のロースを買う、豚肉を炒める"},
            {"reading": "トンニク", "full_reading": "トンニク", "sense": "豚の肉（音読み・漢語的/専門表現）", "cues": "豚肉を使った伝統的な煮込み料理"},
        ],
    },
    {
        "surface": "物怪",
        "span_surface": "物怪",
        "key": "物怪",
        "category": "low_freq_general",
        "readings": [
            {"reading": "モノノケ", "full_reading": "モノノケ", "sense": "人に取り憑く妖怪や怨霊", "cues": "古い屋敷に物怪が現れるという噂"},
            {"reading": "モッケ", "full_reading": "モッケ", "sense": "思いがけない幸運（物怪の幸い）", "cues": "雨が降らずに済んだのは物怪の幸いだ"},
        ],
    },
    {
        "surface": "下男",
        "span_surface": "下男",
        "key": "下男",
        "category": "low_freq_general",
        "readings": [
            {"reading": "ゲナン", "full_reading": "ゲナン", "sense": "昔の家で使われていた男性の奉公人", "cues": "大店の屋敷で働く下男が荷物を運ぶ"},
            {"reading": "シモオトコ", "full_reading": "シモオトコ", "sense": "雑役をする男性奉公人（訓読み表現）", "cues": "台所の薪割りをする下男"},
        ],
    },
    {
        "surface": "実事",
        "span_surface": "実事",
        "key": "実事",
        "category": "low_freq_general",
        "readings": [
            {"reading": "ジツゴト", "full_reading": "ジツゴト", "sense": "歌舞伎で誠実な立ち役や現実的な演技", "cues": "歌舞伎の舞台で見せる重厚な実事の芸"},
            {"reading": "ジツジ", "full_reading": "ジツジ", "sense": "実際に起きた事実・実務上の事柄", "cues": "歴史的な実事に基づいて物語を構成する"},
        ],
    },
    {
        "surface": "生々",
        "span_surface": "生々",
        "key": "生々",
        "category": "low_freq_general",
        "readings": [
            {"reading": "イキイキ", "full_reading": "イキイキ", "sense": "活気にあふれて元気な様子（生々した）", "cues": "子どもたちが生々とした表情で走る"},
            {"reading": "ナマナマ", "full_reading": "ナマナマ", "sense": "まだ新しく生々しい状態（生々しい）", "cues": "当時の記憶がいまも生々しく残っている"},
        ],
    },
    {
        "surface": "関西",
        "span_surface": "関西",
        "key": "関西",
        "category": "low_freq_general",
        "readings": [
            {"reading": "カンサイ", "full_reading": "カンサイ", "sense": "日本の近畿地方の呼び方（大阪・京都など）", "cues": "来週から関西方面へ出張する、関西の食文化"},
            {"reading": "カンセイ", "full_reading": "カンセイ", "sense": "古い呼称での関西・函谷関の西（関西学院等）", "cues": "歴史書における関西の地域区分"},
        ],
    },
]

HOMOGRAPH_CATALOG.extend(LOW_FREQ_GENERAL_HOMOGRAPHS)
