#!/usr/bin/env python3
"""オープンデータから合成部品リストを作る前処理。

出典とライセンスの確認記録は PARTS-SOURCES.md が正。ここでは確認済みのものだけを扱う。

1) 郵便番号データ (日本郵便, utf_ken_all.csv) -> places-extended.csv
   列: prefecture, city, town, postal
   utf_ken_all.csv の列順 (1レコード1行・UTF-8版):
     0:全国地方公共団体コード 1:旧郵便番号 2:郵便番号7桁 3-5:都道府県/市区町村/町域のカナ
     6:都道府県 7:市区町村 8:町域 9:複数番号フラグ ... (公式の説明ページで確認してから使う)
   使い方: python3 prep_parts.py postal --in utf_ken_all.csv --out data/places-extended.csv

2) 法人番号の全件/都道府県別CSV (国税庁) -> orgs.txt (法人名のみ, 重複除去, 無作為抽出)
   使い方: python3 prep_parts.py orgs --in houjin.csv --name-col 6 --limit 30000 --out data/orgs.txt
   列番号はリソース定義書で確認してから指定する (既定値は置かない)。
"""
import argparse, csv, random, sys

def postal(args):
    out = []
    seen = set()
    with open(args.infile, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 9:
                continue
            postal, pref, city, town = row[2].strip(), row[6].strip(), row[7].strip(), row[8].strip()
            if town in ("以下に掲載がない場合", ""):
                town = ""
            key = (pref, city, town, postal)
            if pref and city and postal and key not in seen:
                seen.add(key)
                out.append({"prefecture": pref, "city": city, "town": town, "postal": postal})
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prefecture", "city", "town", "postal"])
        w.writeheader(); w.writerows(out)
    print(f"postal: wrote {len(out)} rows -> {args.out}")

def orgs(args):
    if args.name_col is None:
        sys.exit("--name-col は必須です。法人番号CSVのリソース定義書で商号の列番号を確認してから指定してください")
    names = set()
    with open(args.infile, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) > args.name_col:
                name = row[args.name_col].strip()
                if name:
                    names.add(name)
    names = sorted(names)
    random.Random(42).shuffle(names)
    names = names[: args.limit]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    print(f"orgs: wrote {len(names)} names -> {args.out}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("postal"); p1.add_argument("--in", dest="infile", required=True); p1.add_argument("--out", required=True); p1.set_defaults(fn=postal)
    p2 = sub.add_parser("orgs"); p2.add_argument("--in", dest="infile", required=True); p2.add_argument("--out", required=True)
    p2.add_argument("--name-col", type=int, default=None); p2.add_argument("--limit", type=int, default=30000); p2.set_defaults(fn=orgs)
    args = ap.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()
