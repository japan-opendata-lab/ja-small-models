# 第2段階 辞書リンカー用辞書の入手と組み立てガイド

本モデル（`ja-address-ner-v1`）は、**2段階ハイブリッド設計**を採用しています。

- **第1段階（同梱モデル本体）**: 自由記述文から住所スパンを抽出し、5つの構成要素（`PREF` 都道府県 / `CITY` 市区町村 / `TOWN` 町域 / `BLOCK` 番地等 / `BUILDING` 建物名）へ分解します。
- **第2段階（同梱リンカー `linker/linker.py`）**: 分解された構成要素を市区町村マスターと照合し、5桁の全国地方公共団体コードを決定論的に付与します（一意に確定できない場合は `UNRESOLVABLE` を返します）。

行政区再編や市町村合併にモデル再学習なしで追随できるよう、**辞書データ本体はモデルリポジトリに同梱していません**。以下の公的オープンデータを入手し、同梱の `build_dictionary.py` で辞書を構築して `AddressLinker` に渡してください。

---

## 1. 公的データの入手先とライセンス

1. **日本郵便「郵便番号データ（住所の郵便番号・UTF-8 1レコード1行形式）」【必須】**
   - 入手先: https://www.post.japanpost.jp/zipcode/dl/utf-zip.html （`utf_ken_all.zip` を展開して `utf_ken_all.csv` を取得）
   - ライセンス: 日本郵便株式会社は同データの説明ページにて「郵便番号データに限っては著作権を主張しません」と明記しています（商用利用可）。

2. **デジタル庁「アドレス・ベース・レジストリ（ABR）町字マスター」【任意・推奨】**
   - 入手先: https://catalog.registries.digital.go.jp/ （全国版または都道府県別の町字マスター CSV / Parquet）
   - ライセンス: 公共データ利用規約（第1.0版）PDL1.0（商用利用可・出典記載が必要）。
   - ※ `utf_ken_all.csv` のみでも 1,892 市区町村辞書を構築できますが、ABR 町字マスターを併用すると `lg_code`（6桁検査数字付きコード）および郵便番号未付与の小字・町字まで網羅できます。

---

## 2. 辞書の組み立て手順

### 日本郵便データのみで構築する場合

```bash
python linker/build_dictionary.py \
  --ken-all /path/to/utf_ken_all.csv \
  --out ./dict_master
```

### 日本郵便データ ＋ デジタル庁 ABR 町字マスターで構築する場合

```bash
python linker/build_dictionary.py \
  --ken-all /path/to/utf_ken_all.csv \
  --abr /path/to/japan_address_registry_town_master.parquet \
  --out ./dict_master
```

実行後、`./dict_master` に以下の JSON ファイルが生成されます。

- `municipalities.json`: 現行市区町村コード（5桁・`lg_code`）・都道府県・郡・市区町村・政令市名・行政区名・カナ読み
- `town_index.json`: コード別の正規化町域キーおよびカナ読みキー
- `old_municipalities.json`: 旧市町村・旧行政区名と現行コードの対応表
- `master-stats.json`: 構築件数サマリ

---

## 3. 推論（抽出 → 分解 → コード付与）の実行例

```bash
# 第1段階のみ（抽出・5要素分解）
python examples/infer_address.py --model-dir . --backend onnx \
  --text "お届け先は東京都渋谷区神南1-2-3 パークハイツ201号室です。"

# 第1段階 ＋ 第2段階（市区町村コード付与／判定不能の返却）
python examples/infer_address.py --model-dir . --backend onnx --master-dir ./dict_master \
  --text "旧住所は浜松市北区根洗町1-2、新住所は府中市3-4-5です。"
```

Python コードからは `examples/infer_address.py` の `JaAddressExtractor(model_dir, backend="onnx", master_dir="./dict_master")` または `linker/linker.py` の `AddressLinker("./dict_master")` を呼び出してください。最新版の自治体コードや独自の事業所・地域マスターへ差し替える場合も、上記 JSON スキーマに合わせて `master_dir` を差し替えるだけでモデル再学習なしに更新できます。
