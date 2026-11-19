#!/usr/bin/env python3
"""第5号 ja-phish-guard-v1 本番データ生成（合成のみ＋任意で FineWeb benign）。

分割: train / dev / sealed-test。dev と sealed には学習に使わない
未知ファミリーを含める（dev用と sealed用で別ファミリー）。
sealed は統計のみを出力し、本文は開かない前提で扱う。

使い方:
  python gen_phish_prod.py --out data/prod [--fineweb benign_web.jsonl]
  benign_web.jsonl は {"text": ...} の行形式（FineWeb-2 由来の通常文）。
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

BRANDS = ["みらい銀行", "さくらカード", "日本宅配便", "モバイルワン",
          "あおぞら証券", "ひかり電力", "中央ガス", "タウン水道局"]
ROLES = ["大手銀行", "宅配業者", "携帯キャリア", "クレジットカード会社",
         "税務署", "市役所", "電力会社", "証券会社"]
AMOUNTS = ["3,000円", "12,800円", "49,800円", "150,000円", "2,400円",
           "980円", "55,000円", "7,700円"]
DEADLINES = ["本日中", "24時間以内", "明日まで", "48時間以内", "今週中",
             "3日以内", "月末まで"]
GREET = ["お客様各位", "いつもご利用ありがとうございます", "ご利用者様",
         "会員様", "お客様"]
DOMPAT = ["secure-{s}.example.jp", "{s}-verify.example.com",
          "{s}-support.example.net", "login-{s}.example.jp",
          "{s}-auth.example.com", "www-{s}.example.net"]
PROFS = ["投資顧問", "資産運用アドバイザー", "元証券マン", "副業コンサルタント"]
PRODS = ["ダイエットサプリ", "育毛剤", "副業教材", "美容液", "健康茶"]
TOPICS = ["会議の日程", "資料の共有", "来週の打ち合わせ", "報告書の提出"]
WHENS = ["今度の休み", "来週末", "今夜", "明日"]
PLANS = ["一緒に食事しませんか", "映画を見に行きませんか",
         "買い物に付き合ってください"]
HOURS = ["10時", "14時", "18時"]
SUBJECTS = ["重要なお知らせ", "ご確認ください", "お知らせ", "ご案内",
            "確認のお願い", "【至急】ご確認ください"]
FOOTERS = ["このメールに心当たりがない場合は破棄してください。",
           "本メールは送信専用アドレスからお送りしています。",
           "ご不明な点はお問い合わせ窓口までご連絡ください。",
           "配信停止をご希望の場合はマイページよりお手続きください。"]

# (family, label, scope, テンプレート群) scope: train / dev_holdout / sealed_holdout
FAMILIES = [
    ("phish_urgent", "phishing", "train", [
        "【重要】{brand}からのお知らせ\n{greet}、不正なログインが検出されました。{deadline}に確認がない場合、アカウントを停止します。下記より本人確認を行ってください。\n{url}",
        "{brand}です。お客様のアカウントに異常なアクセスがありました。{deadline}に下記ページで確認をお願いします。\n{url}",
        "セキュリティ警告：通常と異なる場所からログインがありました。ご本人でない場合は直ちに確認してください。\n{url}",
    ]),
    ("phish_delivery", "phishing", "train", [
        "{brand}です。お荷物のお届けに伺いましたがご不在でした。再配達の手続きはこちらから行ってください。\n{url}\n保管期限は{deadline}です。",
        "お荷物の配達に関するお知らせです。住所の確認が必要なため、下記より情報を更新してください。\n{url}",
    ]),
    ("phish_refund", "phishing", "train", [
        "{org}からのお知らせです。還付金 {amount} の受取手続きが完了していません。ATMまたは下記ページより口座情報を登録してください。\n{url}",
        "還付金のご案内です。{amount} の払い戻しには口座登録が必要です。{deadline}にお手続きください。\n{url}",
    ]),
    ("phish_unpaid", "phishing", "train", [
        "{brand}です。ご利用料金 {amount} のお支払いが確認できていません。{deadline}に下記よりお支払いください。未納の場合、サービスを停止します。\n{url}",
        "料金未納のお知らせです。{amount} が未払いとなっています。今すぐ確認してください。\n{url}",
    ]),
    ("phish_virus", "phishing", "train", [
        "警告：お使いの端末がウイルスに感染している可能性があります。今すぐ下記から対策ツールを取得してください。\n{url}",
        "セキュリティスキャンの結果、重大な脅威が検出されました。{deadline}に対処してください。\n{url}",
    ]),
    ("phish_verify", "phishing", "train", [
        "本人確認のお願いです。法令に基づき、{deadline}に身分証明書の画像と口座情報を登録してください。\n{url}",
        "{brand}です。登録情報の更新が必要です。確認が取れない場合、取引を制限します。\n{url}",
    ]),
    ("fraud_invest", "fraud", "train", [
        "はじめまして、{prof}です。月利10%を保証する特別な投資枠が残りわずかです。まずは無料セミナーへご参加ください。参加費 {amount} は後日返金されます。",
        "確実に増える投資案件のご案内です。元本保証で月利8%。枠が埋まる前にご連絡ください。",
        "未公開株の特別案内です。一般には出回らない情報のため、口外はご遠慮ください。まずは {amount} から始められます。",
    ]),
    ("fraud_job", "fraud", "train", [
        "簡単な作業で日給 {amount}。スマホだけで完結、未経験歓迎。登録には保証金 {amount} が必要です。詳しくはメッセージでご案内します。",
        "在宅でできる高収入バイトのご案内です。まずは登録料 {amount} をお支払いください。",
    ]),
    ("fraud_task", "fraud", "train", [
        "動画にいいねするだけで1件300円。続けるには VIP プラン {amount} への加入が必要です。報酬はまとめて出金できます。",
        "スクリーンショットを送るだけの簡単タスクです。ランクを上げると報酬が増えます。アップグレード費用は {amount} です。",
    ]),
    ("fraud_romance", "fraud", "train", [
        "やっとあなたに出会えました。実は投資で成功していて、あなたにも教えたいんです。まずは少額から一緒に始めましょう。",
        "将来のために二人の資金を増やしませんか。私の言う通りに送金すれば必ず増えます。",
    ]),
    ("fraud_unpaid", "fraud", "train", [
        "コンテンツ利用料 {amount} が未納です。本日中に連絡がない場合、法的手続きに移行します。窓口までお電話ください。",
        "登録した覚えのないサイトの料金 {amount} が発生しています。退会手続きには電子マネーでの支払いが必要です。",
    ]),
    ("spam_ad", "spam", "train", [
        "【期間限定】{prod}が今だけ半額。残りわずかです。注文はこちら {url} ※効果には個人差があります。",
        "大人気の{prod}が再入荷しました。今なら送料無料です。 {url}",
    ]),
    ("spam_dating", "spam", "train", [
        "はじめまして。近所に住む30代です。よかったらお話ししませんか。返信待ってます。",
        "寂しい夜に連絡できる相手を探しています。まずはメッセージからお願いします。",
    ]),
    ("phish_password", "phishing", "dev_holdout", [
        "{brand}セキュリティセンターです。パスワードの有効期限が切れます。継続利用には再設定が必要です。下記よりお手続きください。\n{url}",
        "パスワード再設定のご案内です。{deadline}に再設定がない場合、ログインできなくなります。\n{url}",
    ]),
    ("fraud_prize", "fraud", "dev_holdout", [
        "おめでとうございます。キャンペーン抽選で現金 {amount} が当選しました。受取には手数料 {amount} の事前振込が必要です。",
        "当選のお知らせです。賞金 {amount} の受取手続きは{deadline}までです。手数料の振込先をご案内します。",
    ]),
    ("phish_sms_code", "phishing", "sealed_holdout", [
        "【{brand}】認証コードの確認のため、届いた6桁のコードを下記ページへ入力してください。\n{url}",
        "ワンタイムパスワードの再発行が必要です。本人確認のため情報を入力してください。\n{url}",
    ]),
    ("fraud_rent", "fraud", "sealed_holdout", [
        "格安物件のご案内です。入居審査の前に保証金 {amount} の振込が必要です。人気物件のためお早めにご連絡ください。",
        "賃貸の仮押さえには申込金 {amount} が必要です。他の希望者がいるため、本日中の振込をお願いします。",
    ]),
    ("hard_delivery", "benign_hard", "train", [
        "{brand}です。お荷物をお届けしました。伝票番号は配達完了メールでご確認いただけます。再配達は公式アプリまたはお電話にて承っております。",
        "配達完了のお知らせです。宅配ボックスへお届けしました。",
    ]),
    ("hard_security", "benign_hard", "train", [
        "{brand}からのセキュリティのお知らせです。最近、弊社を装った不審なメールが確認されています。弊社がメールでパスワードや暗証番号をお尋ねすることはありません。ご注意ください。",
        "不審なメールへの注意喚起です。心当たりのないメール内のリンクは開かないでください。",
    ]),
    ("hard_billing", "benign_hard", "train", [
        "{brand}です。今月のご利用料金 {amount} が確定しました。明細はマイページでご確認いただけます。口座振替日は月末です。",
        "料金確定のお知らせです。今月分は {amount} です。詳細はアプリでご確認ください。",
    ]),
    ("hard_reserve", "benign_hard", "train", [
        "ご予約の確認です。{when} {hour}よりご来店をお待ちしております。変更はお電話で承ります。",
        "予約が確定しました。当日は予約番号をお持ちください。",
    ]),
    ("hard_point", "benign_hard", "train", [
        "ポイント有効期限のお知らせです。お持ちのポイントは{deadline}までご利用いただけます。残高はマイページでご確認ください。",
        "ポイントが貯まりました。交換は公式アプリからお手続きいただけます。",
    ]),
    ("hard_app", "benign_hard", "train", [
        "アプリの新しいバージョンを公開しました。アプリストアから更新してお使いください。",
        "メンテナンスのお知らせです。{deadline}の深夜にサービスを一時停止します。ご不便をおかけします。",
    ]),
    ("benign_work", "benign", "train", [
        "お疲れさまです。{topic}について、資料を共有しました。ご確認のうえ、気になる点があればご連絡ください。",
        "{topic}の件、明日の会議で確認しましょう。",
    ]),
    ("benign_news", "benign", "train", [
        "今週のニュースレターをお届けします。地域のイベント情報と新着記事をまとめました。次号は来週配信予定です。",
        "月刊誌の最新号が発行されました。特集は地域の食文化です。",
    ]),
    ("benign_personal", "benign", "train", [
        "{when}、{plan}。都合を教えてください。",
        "昨日の件ありがとうございました。また{when}に集まりましょう。",
    ]),
]



def amt(rng):
    if rng.random() < 0.5:
        return rng.choice(AMOUNTS)
    v = rng.choice([rng.randrange(500, 3000, 10),
                    rng.randrange(3000, 50000, 100),
                    rng.randrange(50000, 300000, 1000)])
    return format(v, ",") + "円"


def render(rng, tpl):
    brand = rng.choice(BRANDS + ROLES)
    slug = "bank" if "銀行" in brand else "service"
    url = "https://" + rng.choice(DOMPAT).format(s=slug) + "/login"
    text = tpl.format(brand=brand,
                      org=rng.choice(["税務署", "市役所", "年金事務所"]),
                      greet=rng.choice(GREET),
                      deadline=rng.choice(DEADLINES),
                      amount=amt(rng), prof=rng.choice(PROFS),
                      prod=rng.choice(PRODS), topic=rng.choice(TOPICS),
                      when=rng.choice(WHENS), plan=rng.choice(PLANS),
                      hour=rng.choice(HOURS), url=url)
    if rng.random() < 0.35:
        text = "件名：" + rng.choice(SUBJECTS) + "\n" + text
    if rng.random() < 0.4:
        text += "\n管理番号：" + str(rng.randrange(100000, 999999))
    if rng.random() < 0.3:
        text += "\n" + rng.choice(FOOTERS)
    if rng.random() < 0.25:
        text = str(rng.randrange(1, 13)) + "月" + \
            str(rng.randrange(1, 29)) + "日\n" + text
    if rng.random() < 0.3:
        text += "\n受付番号：" + str(rng.randrange(1000, 9999)) + "-" + \
            str(rng.randrange(1000, 9999))
    if rng.random() < 0.2:
        text += "\nお問い合わせは" + rng.choice(["平日9時から17時", "24時間受付", "チャット窓口"]) + "です。"
    return text


def fill(rng, fams, total, seen, out_rows):
    if not fams:
        return
    per = max(1, total // len(fams))
    for name, label, _scope, tpls in fams:
        got = 0
        attempts = 0
        while got < per and attempts < per * 40:
            attempts += 1
            text = render(rng, rng.choice(tpls))
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            out_rows.append({"text": text, "label": label, "family": name,
                             "channel": "mixed"})
            got += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/prod")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--train-n", type=int, default=40000)
    ap.add_argument("--dev-n", type=int, default=4000)
    ap.add_argument("--sealed-n", type=int, default=3000)
    ap.add_argument("--fineweb", default=None,
                    help='{"text": ...} 行形式の通常文ファイル（任意）')
    args = ap.parse_args()
    rng = random.Random(args.seed)
    by_scope = {"dev_holdout": [], "sealed_holdout": []}
    hard = []
    benign = []
    atk_train = []
    for f in FAMILIES:
        if f[2] in by_scope:
            by_scope[f[2]].append(f)
        elif f[1] == "benign_hard":
            hard.append(f)
        elif f[1] == "benign":
            benign.append(f)
        else:
            atk_train.append(f)
    seen = set()

    train = []
    fill(rng, atk_train, int(args.train_n * 0.45), seen, train)
    fill(rng, hard, int(args.train_n * 0.30), seen, train)
    fill(rng, benign, int(args.train_n * 0.15), seen, train)
    if args.fineweb:
        web = [json.loads(l)["text"] for l in
               open(args.fineweb, encoding="utf-8")]
        rng.shuffle(web)
        for t in web[:int(args.train_n * 0.10)]:
            h = hashlib.md5(t.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                train.append({"text": t, "label": "benign",
                              "family": "benign_web", "channel": "mixed"})
    rng.shuffle(train)

    dev = []
    fill(rng, atk_train, int(args.dev_n * 0.35), seen, dev)
    fill(rng, by_scope["dev_holdout"], int(args.dev_n * 0.10), seen, dev)
    fill(rng, hard, int(args.dev_n * 0.30), seen, dev)
    fill(rng, benign, int(args.dev_n * 0.25), seen, dev)
    rng.shuffle(dev)

    sealed = []
    fill(rng, atk_train, int(args.sealed_n * 0.35), seen, sealed)
    fill(rng, by_scope["sealed_holdout"], int(args.sealed_n * 0.20),
         seen, sealed)
    fill(rng, hard, int(args.sealed_n * 0.25), seen, sealed)
    fill(rng, benign, int(args.sealed_n * 0.20), seen, sealed)
    rng.shuffle(sealed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, rows in [("train", train), ("dev", dev),
                       ("sealed-test", sealed)]:
        with open(out / (name + ".jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[name] = {"rows": len(rows),
                       "labels": dict(Counter(r["label"] for r in rows)),
                       "families": dict(Counter(r["family"] for r in rows))}
    (out / "prod-stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v["rows"] for k, v in stats.items()},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
