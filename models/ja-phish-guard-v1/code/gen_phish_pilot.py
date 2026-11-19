#!/usr/bin/env python3
"""第5号 ja-phish-guard-v1 パイロット用合成データ生成。

手口の型（ファミリー）ごとにテンプレートを持ち、スロット差し替えで文面を作る。
dev には学習に使わないファミリー（holdout）を含め、未知の手口への転移を測る。
実在ブランド名は使わず、役割名と架空ブランドで構成する。

使い方:
  python gen_phish_pilot.py --out data/pilot --seed 42
出力: train.jsonl / dev.jsonl / gen-stats.json
  各行: {"text", "label", "family", "channel"}
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
AMOUNTS = ["3,000円", "12,800円", "49,800円", "150,000円", "2,400円"]
DEADLINES = ["本日中", "24時間以内", "明日まで", "48時間以内", "今週中"]
GREET = ["お客様各位", "いつもご利用ありがとうございます", "ご利用者様",
         "会員様", "お客様"]
DOMPAT = ["secure-{s}.example.jp", "{s}-verify.example.com",
          "{s}-support.example.net", "login-{s}.example.jp"]
PROFS = ["投資顧問", "資産運用アドバイザー", "元証券マン"]
PRODS = ["ダイエットサプリ", "育毛剤", "副業教材", "美容液"]
TOPICS = ["会議の日程", "資料の共有", "来週の打ち合わせ", "報告書の提出"]
WHENS = ["今度の休み", "来週末", "今夜", "明日"]
PLANS = ["一緒に食事しませんか", "映画を見に行きませんか",
         "買い物に付き合ってください"]
HOURS = ["10時", "14時", "18時"]
SUBJECTS = ["重要なお知らせ", "ご確認ください", "お知らせ", "ご案内",
            "確認のお願い"]
FOOTERS = ["このメールに心当たりがない場合は破棄してください。",
           "本メールは送信専用アドレスからお送りしています。",
           "ご不明な点はお問い合わせ窓口までご連絡ください。",
           "配信停止をご希望の場合はマイページよりお手続きください。"]

# (family, label, holdoutか, テンプレート群)
FAMILIES = [
    ("phish_urgent", "phishing", False, [
        "【重要】{brand}からのお知らせ\n{greet}、不正なログインが検出されました。{deadline}に確認がない場合、アカウントを停止します。下記より本人確認を行ってください。\n{url}",
        "{brand}です。お客様のアカウントに異常なアクセスがありました。{deadline}に下記ページで確認をお願いします。\n{url}",
    ]),
    ("phish_delivery", "phishing", False, [
        "{brand}です。お荷物のお届けに伺いましたがご不在でした。再配達の手続きはこちらから行ってください。\n{url}\n保管期限は{deadline}です。",
        "お荷物の配達に関するお知らせです。住所の確認が必要なため、下記より情報を更新してください。\n{url}",
    ]),
    ("phish_refund", "phishing", False, [
        "{org}からのお知らせです。還付金 {amount} の受取手続きが完了していません。ATMまたは下記ページより口座情報を登録してください。\n{url}",
        "還付金のご案内です。{amount} の払い戻しには口座登録が必要です。{deadline}にお手続きください。\n{url}",
    ]),
    ("fraud_invest", "fraud", False, [
        "はじめまして、{prof}です。月利10%を保証する特別な投資枠が残りわずかです。まずは無料セミナーへご参加ください。参加費 {amount} は後日返金されます。",
        "確実に増える投資案件のご案内です。元本保証で月利8%。枠が埋まる前にご連絡ください。",
    ]),
    ("fraud_job", "fraud", False, [
        "簡単な作業で日給 {amount}。スマホだけで完結、未経験歓迎。登録には保証金 {amount} が必要です。詳しくはメッセージでご案内します。",
        "在宅でできる高収入バイトのご案内です。まずは登録料 {amount} をお支払いください。",
    ]),
    ("spam_ad", "spam", False, [
        "【期間限定】{prod}が今だけ半額。残りわずかです。注文はこちら {url} ※効果には個人差があります。",
        "大人気の{prod}が再入荷しました。今なら送料無料です。 {url}",
    ]),
    ("phish_password", "phishing", True, [
        "{brand}セキュリティセンターです。パスワードの有効期限が切れます。継続利用には再設定が必要です。下記よりお手続きください。\n{url}",
        "パスワード再設定のご案内です。{deadline}に再設定がない場合、ログインできなくなります。\n{url}",
    ]),
    ("fraud_prize", "fraud", True, [
        "おめでとうございます。キャンペーン抽選で現金 {amount} が当選しました。受取には手数料 {amount} の事前振込が必要です。",
        "当選のお知らせです。賞金 {amount} の受取手続きは{deadline}までです。手数料の振込先をご案内します。",
    ]),
    ("hard_delivery", "benign_hard", False, [
        "{brand}です。お荷物をお届けしました。伝票番号は配達完了メールでご確認いただけます。再配達は公式アプリまたはお電話にて承っております。",
        "配達完了のお知らせです。宅配ボックスへお届けしました。",
    ]),
    ("hard_security", "benign_hard", False, [
        "{brand}からのセキュリティのお知らせです。最近、弊社を装った不審なメールが確認されています。弊社がメールでパスワードや暗証番号をお尋ねすることはありません。ご注意ください。",
        "不審なメールへの注意喚起です。心当たりのないメール内のリンクは開かないでください。",
    ]),
    ("hard_billing", "benign_hard", False, [
        "{brand}です。今月のご利用料金 {amount} が確定しました。明細はマイページでご確認いただけます。口座振替日は月末です。",
        "料金確定のお知らせです。今月分は {amount} です。詳細はアプリでご確認ください。",
    ]),
    ("hard_reserve", "benign_hard", False, [
        "ご予約の確認です。{when} {hour}よりご来店をお待ちしております。変更はお電話で承ります。",
        "予約が確定しました。当日は予約番号をお持ちください。",
    ]),
    ("benign_work", "benign", False, [
        "お疲れさまです。{topic}について、資料を共有しました。ご確認のうえ、気になる点があればご連絡ください。",
        "{topic}の件、明日の会議で確認しましょう。",
    ]),
    ("benign_news", "benign", False, [
        "今週のニュースレターをお届けします。地域のイベント情報と新着記事をまとめました。次号は来週配信予定です。",
        "月刊誌の最新号が発行されました。特集は地域の食文化です。",
    ]),
    ("benign_personal", "benign", False, [
        "{when}、{plan}。都合を教えてください。",
        "昨日の件ありがとうございました。また{when}に集まりましょう。",
    ]),
]


def render(rng, tpl):
    brand = rng.choice(BRANDS + ROLES)
    slug = "bank" if "銀行" in brand else "service"
    url = "https://" + rng.choice(DOMPAT).format(s=slug) + "/login"
    text = tpl.format(brand=brand,
                      org=rng.choice(["税務署", "市役所", "年金事務所"]),
                      greet=rng.choice(GREET),
                      deadline=rng.choice(DEADLINES),
                      amount=rng.choice(AMOUNTS), prof=rng.choice(PROFS),
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
    return text


def build(rng, n, include_holdout):
    attack = [f for f in FAMILIES if f[1] in ("phishing", "fraud", "spam")
              and (include_holdout or not f[2])]
    hard = [f for f in FAMILIES if f[1] == "benign_hard"]
    benign = [f for f in FAMILIES if f[1] == "benign"]
    plan = [(attack, int(n * 0.45)), (hard, int(n * 0.30)),
            (benign, int(n * 0.25))]
    seen = set()
    uniq = []
    for fams, total in plan:
        per = max(1, total // len(fams))
        for name, label, _ho, tpls in fams:
            got = 0
            attempts = 0
            while got < per and attempts < per * 30:
                attempts += 1
                text = render(rng, rng.choice(tpls))
                h = hashlib.md5(text.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                uniq.append({"text": text, "label": label, "family": name,
                             "channel": "mixed"})
                got += 1
    rng.shuffle(uniq)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pilot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-n", type=int, default=10000)
    ap.add_argument("--dev-n", type=int, default=2000)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    train = build(rng, args.train_n, include_holdout=False)
    dev = build(rng, args.dev_n, include_holdout=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("dev", dev)]:
        with open(out / (name + ".jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"train": len(train), "dev": len(dev),
             "train_labels": dict(Counter(r["label"] for r in train)),
             "dev_labels": dict(Counter(r["label"] for r in dev)),
             "dev_families": dict(Counter(r["family"] for r in dev))}
    (out / "gen-stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
