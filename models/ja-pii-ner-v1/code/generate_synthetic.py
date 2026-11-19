#!/usr/bin/env python3
"""自作合成データ生成器 v1（ja-pii-ner-v1 補強データ用）

方針（仕様書6節）:
- 実在の個人情報は使わない。氏名は姓・名リストの組み合わせ、住所は市区町村マスターの
  地名（都道府県・市区町村名という事実情報のみ）＋架空の番地・建物名で作る。
- 電話番号は実在し得る番号の大量生成を避け、携帯は 070/080/090 の形でも末尾を
  ランダム化した学習用の架空番号として扱い、固定電話は地名に紐づかない市外局番を使う。
  ※公開データセット化の前に、番号帯の扱いはメインと再確認する。
- マイナンバーはチェックデジット規則に沿った架空番号、カード番号は Luhn を通る
  テスト用の番号だけを正例にし、通らない数字列は負例（ラベルなし）に入れる。
- 出力は JSONL: {"text": str, "spans": [{"start","end","text","label"}]}
  ラベルは v1 の18種。スパンは文字単位で、text[start:end] == span["text"] を保証する。
"""
import argparse, csv, json, random, string, sys
from pathlib import Path

LABELS = ["PERSON","ORGANIZATION","ADDRESS","POSTAL_CODE","PHONE","EMAIL",
          "DATE_OF_BIRTH","MY_NUMBER","BANK_ACCOUNT","CREDIT_CARD","DRIVER_LICENSE",
          "PASSPORT","HEALTH_INSURANCE","EMPLOYEE_ID","IP_ADDRESS","URL_PERSONAL",
          "USERNAME","SECRET"]

FAMILY = ["佐藤","鈴木","高橋","田中","渡辺","伊藤","山本","中村","小林","加藤",
          "吉田","山田","佐々木","山口","松本","井上","木村","林","斎藤","清水",
          "森","池田","橋本","阿部","石川","中島","前田","藤田","小川","岡田"]
GIVEN_M = ["健太","翔太","大輝","悠真","蓮","颯太","樹","陽翔","陸","湊","蒼","律"]
GIVEN_F = ["陽菜","結衣","葵","凛","美咲","花","結菜","莉子","紬","澪","楓","栞"]
ORG_A = ["あおぞら","ひかり","さくら","みらい","つばさ","ほくと","わかば","あすか"]
ORG_B = ["商事","システムズ","フーズ","メディカル","不動産","運輸","印刷","電機"]
ORG_SUFFIX = ["株式会社","有限会社","合同会社"]
BUILDINGS = ["パークハイツ","サンライズマンション","グリーンコート","リバーサイド荘","第二コーポ"]
STREETS = ["中央","本町","駅前","大通","緑町","栄町","旭町","港町","西町","東町"]

def luhn_complete(prefix: str, total_len: int, rng: random.Random) -> str:
    digits = [int(c) for c in prefix]
    while len(digits) < total_len - 1:
        digits.append(rng.randrange(10))
    s = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9: d -= 9
        s += d
    check = (10 - (s % 10)) % 10
    return "".join(map(str, digits)) + str(check)

def luhn_ok(num: str) -> bool:
    s = 0
    for i, c in enumerate(reversed(num)):
        d = int(c)
        if i % 2 == 1:
            d *= 2
            if d > 9: d -= 9
        s += d
    return s % 10 == 0

def my_number(rng: random.Random) -> str:
    body = [rng.randrange(10) for _ in range(11)]
    total = 0
    for i, d in enumerate(reversed(body)):  # i=0 が右端 (p1)
        n = i + 1
        q = n + 1 if n <= 6 else n - 5
        total += d * q
    check = 11 - (total % 11)
    if check >= 10: check = 0
    return "".join(map(str, body)) + str(check)

# 実在の市外局番（総務省「市外局番の一覧」に基づく代表例。都道府県→市外局番）
# 部品の出典とライセンスは PARTS-SOURCES.md を参照。
AREA_CODES = {
    "北海道": ["011", "0134", "0154"], "青森県": ["017"], "岩手県": ["019"], "宮城県": ["022"],
    "秋田県": ["018"], "山形県": ["023"], "福島県": ["024"], "茨城県": ["029"],
    "栃木県": ["028"], "群馬県": ["027"], "埼玉県": ["048", "04"], "千葉県": ["043", "047"],
    "東京都": ["03", "042"], "神奈川県": ["045", "046", "042"], "新潟県": ["025"],
    "富山県": ["076"], "石川県": ["076"], "福井県": ["0776"], "山梨県": ["055"],
    "長野県": ["026"], "岐阜県": ["058"], "静岡県": ["054", "053"], "愛知県": ["052", "0561"],
    "三重県": ["059"], "滋賀県": ["077"], "京都府": ["075"], "大阪府": ["06", "072"],
    "兵庫県": ["078", "079"], "奈良県": ["0742"], "和歌山県": ["073"], "鳥取県": ["0857"],
    "島根県": ["0852"], "岡山県": ["086"], "広島県": ["082"], "山口県": ["083"],
    "徳島県": ["088"], "香川県": ["087"], "愛媛県": ["089"], "高知県": ["088"],
    "福岡県": ["092", "093"], "佐賀県": ["0952"], "長崎県": ["095"], "熊本県": ["096"],
    "大分県": ["097"], "宮崎県": ["0985"], "鹿児島県": ["099"], "沖縄県": ["098"],
}

def load_places(path: str):
    """places.csv を読む。列は prefecture, city が必須で、town, postal は任意。
    town / postal は日本郵便の郵便番号データや町字マスターから作った拡張版で使う。"""
    places = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pref, city = (row.get("prefecture") or "").strip(), (row.get("city") or "").strip()
            town = (row.get("town") or "").strip()
            postal = (row.get("postal") or "").strip()
            if pref and city:
                places.append({"pref": pref, "city": city, "town": town, "postal": postal})
    return places

def load_orgs(path: str):
    """組織名の部品リスト（1行1件）。法人番号公表や医療機関データから作る。"""
    if not path:
        return []
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]

class Builder:
    """文字列を組み立てながらスパンを記録する。"""
    def __init__(self):
        self.parts, self.spans, self.pos = [], [], 0
    def add(self, text: str, label: str = None):
        if label:
            self.spans.append({"start": self.pos, "end": self.pos + len(text),
                               "text": text, "label": label})
        self.parts.append(text); self.pos += len(text)
        return self
    def build(self):
        return {"text": "".join(self.parts), "spans": self.spans}

def gen_person(rng):
    fam = rng.choice(FAMILY)
    given = rng.choice(GIVEN_M + GIVEN_F)
    return fam, given

def gen_address(rng, place):
    """place は load_places の1件。town があれば町名・大字レベルまで実在の表層を使う。"""
    base = f"{place['pref']}{place['city']}{place['town'] if place['town'] else rng.choice(STREETS)}"
    banchi = f"{rng.randrange(1,30)}番{rng.randrange(1,40)}号"
    if rng.random() < 0.4:
        banchi += f" {rng.choice(BUILDINGS)}{rng.randrange(101,999)}号室"
    return base + banchi

def gen_phone(rng, pref=None):
    kind = rng.random()
    if kind < 0.45:
        return f"{rng.choice(['070','080','090'])}-{rng.randrange(1000,9999)}-{rng.randrange(1000,9999)}"
    if kind < 0.8:
        # 固定電話は地域に対応する実在の市外局番を使う。市内局番以降は生成する
        codes = AREA_CODES.get(pref)
        if codes:
            ac = rng.choice(codes)
            rest_len = 10 - len(ac)  # 0 を含め10桁にそろえる
            half = rest_len // 2
            return f"{ac}-{rng.randrange(10**(half-1), 10**half)}-{rng.randrange(10**(rest_len-half-1), 10**(rest_len-half))}"
        return f"0{rng.randrange(2,9)}-{rng.randrange(1000,9999)}-{rng.randrange(1000,9999)}"
    return f"{rng.choice(['0120','0800'])}-{rng.randrange(100,999)}-{rng.randrange(100,999)}"

def gen_email(rng, fam, given):
    dom = rng.choice(["example.co.jp","example.jp","mail.example.ne.jp","example.com"])
    user = rng.choice([f"{fam}.{given}", f"{given}{rng.randrange(100,999)}", f"{fam}{rng.randrange(10,99)}"])
    # ローマ字化は簡易にプレースホルダではなく英字ランダムで作る（氏名の読みは持たない）
    user = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randrange(5,9))) + str(rng.randrange(0,999))
    return f"{user}@{dom}"

def gen_dob(rng):
    y, m, d = rng.randrange(1950,2010), rng.randrange(1,13), rng.randrange(1,29)
    style = rng.random()
    if style < 0.4: return f"{y}年{m}月{d}日"
    if style < 0.7: return f"{y}/{m:02d}/{d:02d}"
    era_y = y - 1988
    return f"平成{era_y}年{m}月{d}日" if era_y > 0 else f"{y}年{m}月{d}日"

def doc_chat(rng, places):
    fam, given = gen_person(rng)
    place = rng.choice(places)
    b = Builder()
    b.add("お世話になっております、").add(f"{fam}{given}", "PERSON").add("です。電話番号が変わりまして、")
    b.add(gen_phone(rng, place["pref"]), "PHONE").add("になりました。")
    if place["postal"] and rng.random() < 0.7:
        b.add("郵便番号は").add(f"{place['postal'][:3]}-{place['postal'][3:]}" if len(place["postal"]) == 7 else place["postal"], "POSTAL_CODE").add("、")
    b.add("住所は").add(gen_address(rng, place), "ADDRESS").add("です。よろしくお願いします。")
    return b.build()

def doc_org(rng, places, orgs):
    """組織名の部品（法人番号公表・医療機関データ由来のリスト）を使う文書。"""
    fam, given = gen_person(rng)
    place = rng.choice(places)
    b = Builder()
    b.add("勤務先は").add(rng.choice(orgs), "ORGANIZATION").add("で、担当の")
    b.add(f"{fam}{given}", "PERSON").add("までご連絡ください。電話は")
    b.add(gen_phone(rng, place["pref"]), "PHONE").add("です。")
    return b.build()

def doc_boost_scarce(rng, places):
    """薄いラベル（MY_NUMBER / CREDIT_CARD / SECRET / DRIVER_LICENSE / PASSPORT）を厚くする文書。"""
    fam, given = gen_person(rng)
    b = Builder()
    which = rng.random()
    b.add(f"申請者：").add(f"{fam} {given}", "PERSON").add("\n")
    if which < 0.25:
        b.add("個人番号は").add(my_number(rng), "MY_NUMBER").add("です。本人確認書類として運転免許証番号")
        b.add("".join(str(rng.randrange(10)) for _ in range(12)), "DRIVER_LICENSE").add("を提出しました。")
    elif which < 0.5:
        b.add("支払いカードは").add(luhn_complete(rng.choice(["4","5","3"]), 16, rng), "CREDIT_CARD")
        b.add("で、有効期限の管理のため旅券番号").add(
            rng.choice(string.ascii_uppercase) + rng.choice(string.ascii_uppercase) +
            "".join(str(rng.randrange(10)) for _ in range(7)), "PASSPORT").add("も登録しています。")
    elif which < 0.75:
        key = "sk-" + "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(24))
        b.add("接続確認のため、APIキー ").add(key, "SECRET").add(" を一時的に共有します。取り扱いにご注意ください。")
    else:
        b.add("生年月日は").add(gen_dob(rng), "DATE_OF_BIRTH").add("、連絡先は")
        b.add(gen_email(rng, fam, given), "EMAIL").add("、口座番号は")
        b.add("".join(str(rng.randrange(10)) for _ in range(7)), "BANK_ACCOUNT").add("です。")
    return b.build()

def doc_profile(rng, places):
    """URL_PERSONAL と USERNAME を厚くする文書。Nemotron 側の URL_PERSONAL が均一な合成形式
    （/document パスと .example 系）しかないため、SNS や個人ページの表層の多様さを足す。"""
    fam, given = gen_person(rng)
    user = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randrange(4, 9))) + str(rng.randrange(10, 99))
    b = Builder()
    kind = rng.random()
    b.add("担当は").add(f"{fam}{given}", "PERSON").add("です。")
    if kind < 0.4:
        host = rng.choice(["x.com", "twitter.com", "instagram.com", "note.com"])
        b.add("SNSは ").add(f"https://{host}/{user}", "URL_PERSONAL").add(" で発信しています。")
    elif kind < 0.7:
        b.add("個人サイトは ").add(f"https://{user}.example.jp/profile", "URL_PERSONAL")
        b.add("、ユーザー名は ").add(user, "USERNAME").add(" です。")
    else:
        b.add("連絡はユーザー名 ").add(user, "USERNAME").add(" まで。プロフィールURLは ")
        b.add(f"https://example.com/users/{user}", "URL_PERSONAL").add(" です。")
    return b.build()

def doc_negative(rng, places):
    """PIIなし文書。固有名詞や数字列は含むが、個人を特定する情報は入れない（ハード負例）。"""
    place = rng.choice(places)
    pref, city = place["pref"], place["city"]
    b = Builder()
    kind = rng.random()
    if kind < 0.34:
        b.add(f"{pref}{city}の来年度予算について、総額は{rng.randrange(100,999)}億円で、前年度比{rng.randrange(1,9)}%増となりました。")
    elif kind < 0.67:
        b.add(f"注文番号 No.{rng.randrange(100000,999999)} の商品は、型番 XK-{rng.randrange(1000,9999)} で、来週出荷予定です。請求書番号は INV-{rng.randrange(2020,2030)}-{rng.randrange(1000,9999)} です。")
    else:
        bad = luhn_complete(rng.choice(["4","5"]), 16, rng)
        bad = bad[:-1] + str((int(bad[-1]) + 1) % 10)  # Luhnを通らないカード風の数字列は正例にしない
        b.add(f"サンプルデータの検証用に、数字列 {bad} と郵便番号に似た {rng.randrange(100,999)}-{rng.randrange(1000,9999)} を使います。これらは実在の番号ではありません。")
    return b.build()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--places", required=True, help="地名CSV（prefecture / city 必須、town / postal 任意）")
    ap.add_argument("--orgs", default=None, help="組織名の部品リスト（1行1件、任意）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negative-ratio", type=float, default=0.2)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    places = load_places(args.places)
    orgs = load_orgs(args.orgs)
    if not places:
        sys.exit("places が空です")
    rows = []
    for _ in range(args.count):
        r = rng.random()
        if r < args.negative_ratio:
            rows.append(doc_negative(rng, places))
        elif r < args.negative_ratio + 0.35:
            rows.append(doc_boost_scarce(rng, places))
        elif r < args.negative_ratio + 0.5:
            rows.append(doc_profile(rng, places))
        elif orgs and r < args.negative_ratio + 0.65:
            rows.append(doc_org(rng, places, orgs))
        else:
            rows.append(doc_chat(rng, places))
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            for sp in row["spans"]:  # 自己検証
                assert row["text"][sp["start"]:sp["end"]] == sp["text"], sp
                assert sp["label"] in LABELS
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")

if __name__ == "__main__":
    main()
