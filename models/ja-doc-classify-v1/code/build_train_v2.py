#!/usr/bin/env python3
"""第4号 ja-doc-classify-v1 文体拡張データ v2 構築スクリプト。

元の data/pilot/train.jsonl（85,042件）のみから各種別 50% を均等抽出し、
4 種類のプログラム的文体変換（email / short / bullet / memo）を適用した変種を作成、
元の train と結合・シャッフルして data/train_v2/train_v2.jsonl を生成する。
※ dev.jsonl / sealed-test.jsonl / data/ood/ は一切参照しない（dev 非重複のみ検査）。
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

GENERIC_SUBJECTS = [
    "ご提出書類の件",
    "お手続き内容のご確認",
    "照会・連絡事項について",
    "送付書類のご案内",
    "確認依頼の件",
    "書類ご提出のお知らせ",
    "事務手続のご連絡",
    "関係書類共有の件",
    "対応状況ご報告",
    "登録・変更内容のご確認",
    "社内事務連絡",
    "お客様窓口からのご案内",
    "ご照会事項への回答",
    "処理状況のお知らせ",
    "書類写しのご送付",
    "ご確認のお願い",
]

EMAIL_OPENINGS = [
    "お世話になっております。下記の通りご連絡いたします。",
    "いつも大変お世話になっております。以下の内容をご確認ください。",
    "お疲れ様です。本件につきまして共有いたします。",
    "ご連絡ありがとうございます。関連内容を下記の通りお送りします。",
    "平素は格別のご高配を賜り厚く御礼申し上げます。詳細を記載いたします。",
    "ご担当者様 いつもお世話になっております。ご確認のほどお願いいたします。",
    "お疲れ様です。担当より下記の件について連絡申し上げます。",
    "本件の要点を以下にまとめましたのでご査収ください。",
    "窓口担当でございます。ご案内事項を以下の通りお伝えいたします。",
    "関係各位 お手数ですが下記内容をご確認願います。",
    "先ほどお話しいたしました件の詳細をお送りいたします。",
    "手続き内容の確認のため、以下にご案内申し上げます。",
]

EMAIL_CLOSINGS = [
    "よろしくお願いいたします。",
    "お手数をおかけいたしますが、ご確認のほどお願い申し上げます。",
    "何かご不明点がございましたら返信にてお知らせください。",
    "以上、引き続きよろしくお願いいたします。",
    "ご確認の上、必要なご対応をお願いいたします。",
    "どうぞよろしくお願い申し上げます。\n---\n事務管理担当 (info@example.jp)",
    "ご査収のほどよろしくお願いいたします。\n窓口サポート係",
    "お忙しいところ恐れ入りますが、ご対応願います。",
    "以上、よろしくお願いいたします。\n担当者：総務・業務課 (03-0000-1100)",
    "万一内容に相違がございましたら至急ご連絡ください。",
    "何卒よろしくお願い申し上げます。",
    "本メールはお知らせ専用となります。問い合わせは窓口までお願いいたします。",
]

FROM_ADDRS = [
    "support@example.jp",
    "desk@example.com",
    "jimukyoku@example.jp",
    "contact@example.com",
    "madoguchi@example.jp",
    "staff@example.com",
]

TO_ADDRS = [
    "user@example.jp",
    "client@example.com",
    "member@example.jp",
    "tantou@example.jp",
    "apply@example.com",
]

BULLET_MARKERS = ["・", "- ", "■ ", "▶ ", "num", "paren"]
BULLET_HEADERS = ["【詳細】", "■ 記載内容", "● 要項", "（内容一覧）", ""]
MEMO_PREFIXES = ["メモ：", "受付控：", "共有：", "【伝達】", "伝言：", ""]

META_ONLY_RE = re.compile(
    r"^(受付日|審査受付日|作成日|発行日|日付|氏名|名前|申込者|依頼人|担当者|住所|生年月日|年齢|電話番号|番号|ID)[:：]"
)


def strip_first_heading(text: str, doc_type: str, rng: random.Random, drop_prob: float) -> str:
    """1行目が短い見出し行または種別名を含む場合、確率 drop_prob で除去する。"""
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    first = lines[0].strip()
    is_heading = (doc_type in first) or (len(first) <= 30 and ":" not in first and "：" not in first)
    rest = "\n".join(lines[1:]).lstrip("\n")
    if is_heading and len(rest) >= 60:
        if rng.random() < drop_prob:
            return rest
        if rng.random() < 0.5:
            return f"（{first}）\n" + rest
    return text


def is_meta_block(para: str) -> bool:
    """日付・氏名・住所等の短いメタ情報だけの段落かを判定する。"""
    lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
    if not lines or len(para) > 120:
        return False
    if len(lines) <= 2 and all(len(ln) <= 35 for ln in lines):
        return True
    meta_hits = sum(1 for ln in lines if META_ONLY_RE.match(ln))
    return meta_hits >= max(1, len(lines) - 1) and len(para) <= 85


def select_core_text(body: str, min_chars: int = 100, max_chars: int = 200) -> str:
    """メタ情報だけの先頭段落を避けつつ、種別の手がかりが残る範囲で短文を抽出する。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paras:
        paras = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not paras:
        return body[:max_chars]

    # 先頭がメタ段落のみなら、後続段落を優先して手がかりを落とさない
    start_idx = 0
    while start_idx + 1 < len(paras) and is_meta_block(paras[start_idx]):
        # 次の段落が短ければ結合を試し、十分長ければ先頭メタをスキップするか1行だけ残す
        combined = paras[start_idx] + "\n" + paras[start_idx + 1]
        if len(combined) <= max_chars:
            paras[start_idx + 1] = combined
            start_idx += 1
            break
        start_idx += 1

    joined = "\n".join(paras[start_idx:])
    if len(joined) <= max_chars:
        if len(joined) >= min_chars or start_idx == 0:
            return joined
        # 足りなければ先頭から取り直す
        joined = "\n".join(paras)
        if len(joined) <= max_chars:
            return joined

    # 文・改行境界で max_chars 以内に収める（最低 min_chars 以上を確保）
    cut = joined[:max_chars]
    best = -1
    for sep in ("。", "\n", "、"):
        pos = cut.rfind(sep)
        if pos >= min_chars - 1:
            best = max(best, pos + (1 if sep == "。" else 0))
    if best >= min_chars:
        return cut[:best].strip()
    # 次の句点まで少しだけ伸ばせるなら許容（上限 220）
    ext = joined[:min(len(joined), max_chars + 25)]
    pos = ext.find("。", min_chars)
    if pos != -1:
        return ext[:pos + 1].strip()
    return cut.strip()


def to_email(text: str, doc_type: str, rng: random.Random) -> str:
    body = strip_first_heading(text, doc_type, rng, drop_prob=0.80)
    # 50% の確率で主要段落 200〜450 文字程度に簡潔化
    if rng.random() < 0.50 and len(body) > 260:
        target = rng.randint(210, 420)
        cut = body[:target]
        pos = max(cut.rfind("。"), cut.rfind("\n"))
        if pos >= 140:
            body = cut[:pos + 1].strip()
        else:
            body = cut.strip()

    subj = rng.choice(GENERIC_SUBJECTS)
    opening = rng.choice(EMAIL_OPENINGS)
    closing = rng.choice(EMAIL_CLOSINGS)
    mode = rng.random()
    if mode < 0.35:
        hdr = f"From: {rng.choice(FROM_ADDRS)}\nTo: {rng.choice(TO_ADDRS)}\n件名: {subj}\n\n"
    elif mode < 0.70:
        hdr = f"件名: {subj}\n\n"
    else:
        hdr = ""
    return f"{hdr}{opening}\n\n{body}\n\n{closing}"


def to_short(text: str, doc_type: str, rng: random.Random) -> str:
    body = strip_first_heading(text, doc_type, rng, drop_prob=0.75)
    target_max = rng.randint(145, 200)
    target_min = rng.randint(100, 130)
    return select_core_text(body, min_chars=target_min, max_chars=target_max)


def to_bullet(text: str, doc_type: str, rng: random.Random) -> str:
    body = strip_first_heading(text, doc_type, rng, drop_prob=0.75)
    raw_lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    items = []
    for ln in raw_lines:
        clean = re.sub(r"^[・\-■▶●◆□\d\.\(\)（）]+\s*", "", ln) if re.match(r"^[・\-■▶●◆□]\s*", ln) else ln
        # 長い散文は句点で分割
        if len(clean) > 65 and "。" in clean:
            parts = [p.strip() + "。" for p in clean.split("。") if p.strip()]
            items.extend(parts)
        else:
            items.append(clean)
    if not items:
        items = [body]

    marker = rng.choice(BULLET_MARKERS)
    header = rng.choice(BULLET_HEADERS)
    out_lines = []
    if header:
        out_lines.append(header)
    for idx, it in enumerate(items, 1):
        if marker == "num":
            m = f"{idx}. "
        elif marker == "paren":
            m = f"({idx}) "
        else:
            m = marker
        out_lines.append(f"{m}{it}")
    sep = "\n" if rng.random() < 0.75 else " / "
    return sep.join(out_lines)


def compress_sentence(s: str) -> str:
    s = s.strip()
    for tail in (
        "をお願いいたします。",
        "お願いいたします。",
        "いたします。",
        "しております。",
        "しています。",
        "ございます。",
        "であります。",
        "となります。",
        "おります。",
        "ください。",
        "ます。",
        "です。",
    ):
        if s.endswith(tail) and len(s) > len(tail) + 4:
            s = s[:-len(tail)]
            break
    return s.rstrip("。")


def to_memo(text: str, doc_type: str, rng: random.Random) -> str:
    body = strip_first_heading(text, doc_type, rng, drop_prob=0.85)
    core = select_core_text(body, min_chars=120, max_chars=240)
    # 行または句点単位で断片化して体言止め化
    raw_frags = []
    for ln in core.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if "。" in ln:
            for p in ln.split("。"):
                if p.strip():
                    raw_frags.append(compress_sentence(p))
        else:
            raw_frags.append(compress_sentence(ln))
    joiner = rng.choice([" / ", "、", "・", "\n"])
    memo_body = joiner.join(raw_frags)
    if len(memo_body) > 240:
        memo_body = memo_body[:240].rstrip(" /、・")
    prefix = rng.choice(MEMO_PREFIXES)
    return f"{prefix}{memo_body}"


def len_stats(lengths: list) -> dict:
    arr = np.array(lengths, dtype=int)
    return {
        "count": int(len(arr)),
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "mean": round(float(arr.mean()), 2),
        "p90": float(np.percentile(arr, 90)),
        "max": int(arr.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/pilot/train.jsonl")
    ap.add_argument("--dev", default="data/pilot/dev.jsonl")
    ap.add_argument("--out-dir", default="data/train_v2")
    ap.add_argument("--ratio", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_rows = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    dev_texts = {json.loads(l)["text"] for l in open(args.dev, encoding="utf-8") if l.strip()}

    rng = random.Random(args.seed)
    by_type = defaultdict(list)
    for r in train_rows:
        by_type[r["document_type"]].append(r)

    aug_types = ["email", "short", "bullet", "memo"]
    converters = {
        "email": to_email,
        "short": to_short,
        "bullet": to_bullet,
        "memo": to_memo,
    }

    aug_rows = []
    aug_per_type = {}
    orig_per_type = {}
    samples = {}

    for ty in sorted(by_type):
        part = list(by_type[ty])
        orig_per_type[ty] = len(part)
        rng.shuffle(part)
        n_aug = round(len(part) * args.ratio)
        aug_per_type[ty] = n_aug
        selected = part[:n_aug]
        for idx, r in enumerate(selected):
            mode = aug_types[idx % len(aug_types)]
            new_text = converters[mode](r["text"], ty, rng).strip()
            assert len(new_text) >= 40, f"Too short ({len(new_text)}) for {ty} mode={mode}: {new_text!r}"
            assert new_text not in dev_texts, f"Dev overlap found for {ty}"
            row_out = {
                "text": new_text,
                "domain": r["domain"],
                "document_type": r["document_type"],
                "document_format": r["document_format"],
                "aug_type": mode,
            }
            aug_rows.append(row_out)
            if mode not in samples:
                samples[mode] = {"orig_type": ty, "domain": r["domain"], "text": new_text}

    orig_tagged = [{**r, "aug_type": "orig"} for r in train_rows]
    all_rows = orig_tagged + aug_rows
    rng.shuffle(all_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "train_v2.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 統計算出
    by_aug = defaultdict(list)
    title_in_text = Counter()
    title_in_1st = Counter()
    for r in all_rows:
        at = r["aug_type"]
        by_aug[at].append(len(r["text"]))
        if r["document_type"] in r["text"]:
            title_in_text[at] += 1
        fl = r["text"].splitlines()[0] if r["text"].splitlines() else ""
        if r["document_type"] in fl:
            title_in_1st[at] += 1

    aug_counts = {k: len(v) for k, v in sorted(by_aug.items())}
    tot_aug = len(aug_rows)
    tot_all = len(all_rows)
    aug_title_text = sum(title_in_text[k] for k in aug_types)
    aug_title_1st = sum(title_in_1st[k] for k in aug_types)
    all_title_text = sum(title_in_text.values())
    all_title_1st = sum(title_in_1st.values())

    stats = {
        "orig_rows": len(train_rows),
        "aug_rows": tot_aug,
        "aug_ratio_vs_orig": round(tot_aug / len(train_rows), 4),
        "total_rows": tot_all,
        "n_types": len(by_type),
        "orig_type_min": min(orig_per_type.values()),
        "orig_type_max": max(orig_per_type.values()),
        "aug_type_min": min(aug_per_type.values()),
        "aug_type_max": max(aug_per_type.values()),
        "counts_by_aug_type": aug_counts,
        "char_len_by_aug_type": {k: len_stats(v) for k, v in sorted(by_aug.items())},
        "lt_40_chars": sum(1 for r in all_rows if len(r["text"]) < 40),
        "dev_exact_overlap": sum(1 for r in all_rows if r["text"] in dev_texts),
        "title_in_text_rate": {
            "orig": round(title_in_text["orig"] / len(train_rows), 4),
            "aug_only": round(aug_title_text / tot_aug, 4),
            "v2_total": round(all_title_text / tot_all, 4),
            **{k: round(title_in_text[k] / len(by_aug[k]), 4) for k in aug_types},
        },
        "title_in_1st_line_rate": {
            "orig": round(title_in_1st["orig"] / len(train_rows), 4),
            "aug_only": round(aug_title_1st / tot_aug, 4),
            "v2_total": round(all_title_1st / tot_all, 4),
            **{k: round(title_in_1st[k] / len(by_aug[k]), 4) for k in aug_types},
        },
        "train_v2_sha256": hashlib.sha256(out_jsonl.read_bytes()).hexdigest(),
        "train_v2_bytes": out_jsonl.stat().st_size,
        "samples": samples,
    }
    (out_dir / "build_v2_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in stats.items() if k != "samples"}, ensure_ascii=False, indent=2))
    print("\n=== SAMPLES ===")
    for k, v in samples.items():
        print(f"[{k}] ({v['orig_type']} | {v['domain']} | len={len(v['text'])})\n{v['text']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
