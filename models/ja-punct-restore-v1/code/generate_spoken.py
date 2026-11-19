#!/usr/bin/env python3
"""第3号：口語（話し言葉）合成データの生成。

疑問符・感嘆符が乏しい書き言葉コーパスを補うため、場面ごとの文型から
話し言葉の文を作り、複数文を連結して学習用ウィンドウにする。
出力は build_punct_data.py と同じ形式（text は句読点なし、labels は文字ラベル）。
生成は学習側専用。評価には使わない。

使い方:
  python generate_spoken.py --windows 15000 --out data/spoken-train.jsonl
"""
import argparse, hashlib, json, random, sys
from pathlib import Path

TARGETS = {"、": 1, "。": 2, "？": 3, "！": 4}

TOPICS = ["新製品の発売", "来期の予算", "店舗の改装", "採用の計画", "システムの移行",
          "イベントの準備", "売上", "在庫", "納期", "品質", "研修の日程", "移転の件",
          "キャンペーン", "問い合わせの件数", "新サービスの評判", "会議室の予約"]
THINGS = ["資料", "数字", "サンプル", "写真", "試作品", "報告書", "画面", "音声",
          "地図", "一覧表", "見積書", "契約書の案", "アンケートの結果", "予約表"]
PLACES = ["駅前", "三階の会議室", "本社", "支店", "会場", "受付", "倉庫", "オンライン",
          "現地", "事務所", "工場", "店舗"]
TIMES = ["来週", "明日", "今月末", "来月", "金曜日の午後", "午前中", "来週の月曜日",
         "今週中", "月末まで", "来週の水曜日"]
NAMES = ["田中さん", "佐藤さん", "鈴木さん", "高橋さん", "山田さん", "伊藤さん",
         "渡辺さん", "山本さん", "中村さん", "小林さん"]

FILLERS = ["ええと、", "あの、", "そうですね、", "はい、", "なるほど、", "うーん、", ""]

S_DECL = [
    "{topic}については、予定どおり進んでいます。",
    "{thing}は、{time}までに用意できます。",
    "先ほどの数字ですが、去年の同じ時期と比べて、少し増えています。",
    "{name}から、{thing}を受け取りました。",
    "この件は、{place}で確認することになっています。",
    "{topic}の進み具合は、全体の七割ほどです。",
    "費用の見積もりは、来週の初めに出る見込みです。",
    "担当の者から、後ほど詳しく説明いたします。",
    "スケジュールに大きな変更はありません。",
    "先方の反応は、おおむね良好でした。",
]
S_QUEST = [
    "{topic}は、いつまでに決めればいいですか？",
    "この{thing}は、誰が用意するのですか？",
    "{time}に、{place}で集まれますか？",
    "予算の上限は、どれくらいを想定していますか？",
    "その話は、本当ですか？",
    "もう一度、最初から説明してもらえますか？",
    "{name}は、もう到着しましたか？",
    "こちらの案と、あちらの案と、どちらが早いですか？",
    "在庫は、まだ残っていますか？",
    "この方法で、問題ないでしょうか？",
    "参加者は、何人くらいになりそうですか？",
    "締め切りは、延ばせますか？",
]
S_EXCL = [
    "それはすごいですね！",
    "本当ですか、助かります！",
    "間に合ってよかったです！",
    "こんなに集まるとは思いませんでした！",
    "数字が、去年の倍になっています！",
    "これは、予想以上の出来です！",
    "ありがとうございます、本当に助かりました！",
    "ついに完成しましたね！",
    "{thing}の出来が、想像以上です！",
    "{topic}が、もう決まったのですか！",
    "{name}、よくやりましたね！",
    "参加者が、{num}人も集まりました！",
    "売上が、去年の{num}倍になりました！",
    "こんなに早く終わるとは思いませんでした！",
    "これは、今年一番のニュースです！",
    "{place}の景色が、最高でした！",
    "みなさんのおかげで、成功しました！",
    "やっと、{thing}が届きました！",
    "想定より、{num}件も多い結果です！",
    "これは驚きました、本当に見事です！",
    "{topic}の反響が、すごいことになっています！",
    "よくぞここまで仕上げましたね！",
    "拍手が止まりませんでした！",
    "こんなチャンスは、めったにありません！",
    "結果が、予想をはるかに上回りました！",
    "{name}のアイデアが、大当たりでした！",
]
S_TALK = [
    "じゃあ、明日の朝、もう一度集まりましょう。",
    "分かりました、その方向で進めましょう。",
    "ちょっと待ってください、今、確認します。",
    "すみません、もう一度お願いできますか。",
    "お先に失礼します、お疲れさまでした。",
    "では、資料を画面に出しますね。",
    "ありがとうございます、よろしくお願いします。",
    "そうしましょう、異論はありません。",
]


def fill(tpl, rng):
    return tpl.format(topic=rng.choice(TOPICS), thing=rng.choice(THINGS),
                      place=rng.choice(PLACES), time=rng.choice(TIMES),
                      name=rng.choice(NAMES), num=rng.choice(
                          ["2", "3", "5", "10", "50", "100", "300", "1000"]))


def make_labels(text):
    chars, labels = [], []
    for ch in text:
        if ch in TARGETS:
            if labels:
                labels[-1] = TARGETS[ch]
            continue
        chars.append(ch)
        labels.append(0)
    return "".join(chars), labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=15000)
    ap.add_argument("--out", default="data/spoken-train.jsonl")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    seen, rows = set(), []
    attempts = 0
    while len(rows) < args.windows and attempts < args.windows * 30:
        attempts += 1
        n = rng.randint(5, 10)
        sents = []
        for _ in range(n):
            r = rng.random()
            if r < 0.42:
                s = fill(rng.choice(S_DECL), rng)
            elif r < 0.72:
                s = fill(rng.choice(S_QUEST), rng)
            elif r < 0.84:
                s = fill(rng.choice(S_EXCL), rng)
            else:
                s = fill(rng.choice(S_TALK), rng)
            if rng.random() < 0.25:
                s = rng.choice(FILLERS) + s
            sents.append(s)
        text = "".join(sents)
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        inp, labels = make_labels(text)
        rows.append({"text": inp, "labels": labels, "doc": "spoken"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in rows:
        for lb in r["labels"]:
            if lb:
                counts[lb] = counts.get(lb, 0) + 1
    print(json.dumps({"windows": len(rows), "unique": len(seen),
                      "label_counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
