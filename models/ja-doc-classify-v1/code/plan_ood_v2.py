#!/usr/bin/env python3
"""第4号 ja-doc-classify-v1 OOD v2（全43種・200件・5文体非均等配分）割当表作成。

OOD v1（4文体均等・25種別）とは異なる配分・別ライター向けに、
全43種別×5文体（60/45/40/30/25件）の厳密な割当表 data/ood_v2/v2_assignment.json を生成する。
"""
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ACTUAL_DOMAINS = {
    "クレジットカード利用明細": "金融",
    "ローン審査書類": "金融",
    "口座開設申込書": "金融",
    "振込依頼書": "金融",
    "本人確認記録": "金融",
    "海外送金依頼書": "金融",
    "国際業務委託契約書": "法務",
    "契約書": "法務",
    "委任状": "法務",
    "本人確認書類一覧": "法務",
    "業務委託契約書": "法務",
    "申立関係書類": "法務",
    "相談記録票": "法務",
    "入院同意書": "医療",
    "処方箋": "医療",
    "検査結果報告書": "医療",
    "診断書": "医療",
    "診療情報提供書": "医療",
    "保護者連絡票": "教育",
    "入学願書": "教育",
    "奨学金申請書": "教育",
    "学生記録票": "教育",
    "成績通知書": "教育",
    "会員登録情報": "EC",
    "注文確認書": "EC",
    "購入者問い合わせ記録": "EC",
    "返金申請書": "EC",
    "配送伝票": "EC",
    "従業員台帳": "雇用",
    "給与連絡票": "雇用",
    "身上変更届": "雇用",
    "退職手続書類": "雇用",
    "雇用契約書": "雇用",
    "アカウント登録票": "通信・IT",
    "アクセス申請書": "通信・IT",
    "セキュリティインシデント報告書": "通信・IT",
    "パスワード再設定案内": "通信・IT",
    "端末登録台帳": "通信・IT",
    "住民異動届": "行政",
    "問い合わせ受付票": "行政",
    "本人確認台帳": "行政",
    "税務申告関連書類": "行政",
    "給付申請書": "行政",
}

COUNT_5_TYPES = [
    # OOD v1 未カバー 18種
    "口座開設申込書", "ローン審査書類", "本人確認記録",
    "契約書", "本人確認書類一覧", "申立関係書類", "相談記録票",
    "入院同意書",
    "入学願書", "学生記録票",
    "会員登録情報", "購入者問い合わせ記録",
    "従業員台帳", "身上変更届",
    "アカウント登録票", "端末登録台帳",
    "問い合わせ受付票", "本人確認台帳",
    # OOD v1 主要/弱点 10種
    "振込依頼書", "委任状", "処方箋", "検査結果報告書", "成績通知書",
    "注文確認書", "返金申請書", "アクセス申請書", "住民異動届", "業務委託契約書",
]

COUNT_4_TYPES = [
    "クレジットカード利用明細", "海外送金依頼書", "国際業務委託契約書",
    "診断書", "診療情報提供書", "奨学金申請書", "保護者連絡票",
    "配送伝票", "雇用契約書", "給与連絡票", "退職手続書類",
    "セキュリティインシデント報告書", "パスワード再設定案内", "給付申請書",
    "税務申告関連書類",
]

STYLE_TARGETS = {
    "business_email": 60,
    "chat_memo": 45,
    "ocr_form": 40,
    "narrative_long": 30,
    "portal_auto": 25,
}

EXACT_ALLOW_TARGETS = {
    "ocr_form": 18,
    "narrative_long": 14,
    "business_email": 11,
    "portal_auto": 7,
    "chat_memo": 6,
}  # 合計 56 / 200 (28.0% 含有, 72.0% 非含有)


def main() -> int:
    assert len(ACTUAL_DOMAINS) == 43
    assert set(COUNT_5_TYPES) | set(COUNT_4_TYPES) == set(ACTUAL_DOMAINS)
    assert not (set(COUNT_5_TYPES) & set(COUNT_4_TYPES))
    assert len(COUNT_5_TYPES) * 5 + len(COUNT_4_TYPES) * 4 == 200

    tiers = {}
    with open("risk_tiers_v0.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tiers[row["document_type"]] = row["risk_tier"]

    rng = random.Random(20261119)
    # 各種別の必要スロットを作成（200個）
    type_slots = []
    for ty in COUNT_5_TYPES:
        type_slots.extend([ty] * 5)
    for ty in COUNT_4_TYPES:
        type_slots.extend([ty] * 4)

    # 各種別に重複が最小になるよう文体を割り当て
    styles_order = ["business_email", "chat_memo", "ocr_form", "narrative_long", "portal_auto"]
    rem = dict(STYLE_TARGETS)
    per_type_styles = defaultdict(list)

    # ラウンド1..4: 各ラウンドで残り枠が大きい文体を優先しつつ種別内重複を避ける
    all_43 = COUNT_5_TYPES + COUNT_4_TYPES
    for rnd in range(5):
        order = list(COUNT_5_TYPES) if rnd == 4 else list(all_43)
        rng.shuffle(order)
        for ty in order:
            # まだその種別に割り当てられていない文体を優先
            avail = [st for st in styles_order if rem[st] > 0]
            fresh = [st for st in avail if st not in per_type_styles[ty]]
            pool = fresh if fresh else avail
            chosen = max(pool, key=lambda st: (rem[st], rng.random()))
            per_type_styles[ty].append(chosen)
            rem[chosen] -= 1

    assert all(v == 0 for v in rem.values()), rem

    # 各種別・文体ペアリストを構築
    items_by_style = defaultdict(list)
    for ty in sorted(ACTUAL_DOMAINS):
        for st in per_type_styles[ty]:
            items_by_style[st].append(ty)

    assignments = []
    for st in styles_order:
        tys = list(items_by_style[st])
        rng.shuffle(tys)
        n_allow = EXACT_ALLOW_TARGETS[st]
        # 同一種別が連続しないよう、かつ allow_exact_name が偏らないよう決定
        flags = [True] * n_allow + [False] * (len(tys) - n_allow)
        rng.shuffle(flags)
        for idx, (ty, allow) in enumerate(zip(tys, flags), 1):
            assignments.append({
                "id": f"v2_{st}_{idx:02d}",
                "style": st,
                "document_type": ty,
                "domain": ACTUAL_DOMAINS[ty],
                "risk_tier": tiers[ty],
                "allow_exact_name": allow,
            })

    # 各種別で少なくとも2件以上 allow_exact_name=False を保証（違反があれば同一文体内でフラグスワップ）
    by_ty = defaultdict(list)
    for a in assignments:
        by_ty[a["document_type"]].append(a)
    for ty, lst in by_ty.items():
        while sum(not x["allow_exact_name"] for x in lst) < 2:
            t_item = next(x for x in lst if x["allow_exact_name"])
            # 同文体でFalseが余っている別種別とスワップ
            cand = next(
                y for y in assignments
                if y["style"] == t_item["style"] and not y["allow_exact_name"]
                and sum(not z["allow_exact_name"] for z in by_ty[y["document_type"]]) > 2
            )
            t_item["allow_exact_name"], cand["allow_exact_name"] = False, True

    out_dir = Path("data/ood_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v2_assignment.json").write_text(
        json.dumps(assignments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Total assignments:", len(assignments))
    print("By style:", dict(Counter(a["style"] for a in assignments)))
    print("Allow exact name:", sum(a["allow_exact_name"] for a in assignments), "/ 200")
    print("By domain:", dict(Counter(a["domain"] for a in assignments)))
    print("By risk_tier:", dict(Counter(a["risk_tier"] for a in assignments)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
