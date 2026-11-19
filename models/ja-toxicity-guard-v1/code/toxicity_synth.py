#!/usr/bin/env python3
"""第8号 ja-toxicity-guard-v1 合成データ生成・LSH検査・安全監査モジュール。

- 分割固有テンプレート (train / dev / sealed-test) を用いて多ラベル有害文と無害文を合成する。
- 全カテゴリで同一のスタイル・文脈導入/結び確率を適用し、表層ショートカットを防止する。
- 文字 5-gram MinHash LSH (num_perm=64, bands=16, rows_per=4, seed=42) により
  分割間の近傍重複 (Jaccard >= 0.80) が 0 件であることを保証する。
- 実在固有名詞・連絡先・自傷具体手段の混入がないことを自動監査する。
"""
import hashlib
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import toxicity_templates as tt

# =====================================================================
# MinHash LSH (第6号・第7号と統一パラメータ)
# =====================================================================
LSH_NUM_PERM = 64
LSH_BANDS = 16
LSH_ROWS_PER = 4
LSH_SEED = 42


def normalize_text_for_dedup(text: str) -> str:
    """空白・改行・句読点・一部記号を除去した正規化文字列を返す。"""
    s = re.sub(r"\s+", "", text)
    s = re.sub(r"[。、．，！？!?,.\-―〜「」『』（）()【】:：/]", "", s)
    return s.lower()


def char_shingles(text: str, n: int = 5) -> Set[str]:
    """文字 n-gram 集合を返す。"""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def build_minhash_coeffs(num_perm: int = LSH_NUM_PERM, seed: int = LSH_SEED):
    rng = random.Random(seed)
    m = (1 << 61) - 1
    coeffs = [(rng.randrange(1, m), rng.randrange(0, m)) for _ in range(num_perm)]
    return m, coeffs


_M_PRIME, _MINHASH_COEFFS = build_minhash_coeffs()


def compute_minhash(shingles: Set[str]) -> List[int]:
    sig = [_M_PRIME] * LSH_NUM_PERM
    for g in shingles:
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:16], 16)
        for i, (a, b) in enumerate(_MINHASH_COEFFS):
            v = (a * h + b) % _M_PRIME
            if v < sig[i]:
                sig[i] = v
    return sig


class LSHIndex:
    """分割間の近傍重複検査・回避用の MinHash LSH インデックス。"""

    def __init__(self, bands: int = LSH_BANDS, rows_per: int = LSH_ROWS_PER) -> None:
        self.bands = bands
        self.rows_per = rows_per
        self.buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
        self.shingles_list: List[Set[str]] = []

    def add(self, shingles: Set[str], sig: List[int]) -> int:
        idx = len(self.shingles_list)
        self.shingles_list.append(shingles)
        for b in range(self.bands):
            key = (b, tuple(sig[b * self.rows_per : (b + 1) * self.rows_per]))
            self.buckets[key].append(idx)
        return idx

    def has_near_dup(self, shingles: Set[str], sig: List[int], thr: float = 0.80) -> bool:
        cand: Set[int] = set()
        for b in range(self.bands):
            key = (b, tuple(sig[b * self.rows_per : (b + 1) * self.rows_per]))
            cand.update(self.buckets.get(key, ()))
        if not cand or not shingles:
            return False
        for idx in cand:
            other = self.shingles_list[idx]
            union = len(shingles | other)
            if union > 0 and len(shingles & other) / union >= thr:
                return True
        return False


def count_lsh_leaks(
    rows_a: Sequence[dict],
    rows_b: Sequence[dict],
    thr: float = 0.80,
) -> int:
    """2分割間で Jaccard >= thr となる近傍重複件数（rows_a 側の件数）を返す。"""
    index = LSHIndex()
    for r in rows_b:
        sh = char_shingles(r["text"])
        sig = compute_minhash(sh)
        index.add(sh, sig)
    hits = 0
    for r in rows_a:
        sh = char_shingles(r["text"])
        sig = compute_minhash(sh)
        if index.has_near_dup(sh, sig, thr=thr):
            hits += 1
    return hits


# =====================================================================
# 安全・倫理自動監査ルール
# =====================================================================
PREFECTURES = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県",
    "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県",
    "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県",
    "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県",
]

REAL_CITIES_AND_COUNTRIES = [
    "札幌市", "仙台市", "さいたま市", "千葉市", "横浜市", "川崎市", "相模原市",
    "新潟市", "静岡市", "浜松市", "名古屋市", "京都市", "大阪市", "堺市",
    "神戸市", "岡山市", "広島市", "北九州市", "福岡市", "熊本市",
    "新宿区", "渋谷区", "港区", "千代田区", "中央区",
    "アメリカ", "中国", "韓国", "ロシア", "イギリス", "フランス", "ドイツ",
]

REAL_SERVICES = [
    "Twitter", "YouTube", "TikTok", "Instagram", "Facebook", "LINE",
    "Google", "Amazon", "Apple", "Microsoft", "Yahoo", "ヤフー",
    "2ch", "5ch", "ニコニコ", "メルカリ",
]

SELF_HARM_FORBIDDEN_MEANS = [
    "首吊り", "首つり", "縊死", "練炭", "硫化水素", "飛び降り", "飛び込み",
    "農薬", "毒物", "大量服薬", "オーバードーズ", "手首を切", "カッター",
    "剃刀", "カミソリ", "包丁", "ロープ", "服毒", "ガス栓",
]

RE_PHONE = re.compile(r"0\d{1,4}-\d{1,4}-\d{3,4}")
RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def audit_safety(rows: Sequence[dict]) -> dict:
    """実在固有名詞・実在連絡先・自傷具体手段の混入がないか監査する。"""
    violations: List[dict] = []
    for r in rows:
        txt = r["text"]
        rid = r.get("id", "")
        for p in PREFECTURES:
            if p in txt:
                violations.append({"id": rid, "rule": "real_prefecture", "token": p})
        for c in REAL_CITIES_AND_COUNTRIES:
            if c in txt:
                violations.append({"id": rid, "rule": "real_city_or_country", "token": c})
        txt_lower = txt.lower()
        for svc in REAL_SERVICES:
            if svc.lower() in txt_lower:
                violations.append({"id": rid, "rule": "real_service", "token": svc})
        if RE_PHONE.search(txt):
            violations.append({"id": rid, "rule": "phone_pattern", "token": "phone"})
        if RE_EMAIL.search(txt):
            violations.append({"id": rid, "rule": "email_pattern", "token": "email"})
        for m in SELF_HARM_FORBIDDEN_MEANS:
            if m in txt:
                violations.append({"id": rid, "rule": "self_harm_means", "token": m})
        # self_harm 陽性文では脅迫向けの武器語（刃物等）も混入禁止とする
        if r.get("labels", {}).get("self_harm", 0) == 1:
            for extra in ("刃物", "火をつけ"):
                if extra in txt:
                    violations.append({"id": rid, "rule": "self_harm_weapon", "token": extra})
    return {
        "checked_rows": len(rows),
        "violation_count": len(violations),
        "violations_sample": violations[:10],
    }


# =====================================================================
# スロット展開・文合成ロジック
# =====================================================================
def _sample_context_slots(rng: random.Random) -> Dict[str, str]:
    """1文書内で一貫させる架空コンテキストスロットをサンプルする。"""
    return {
        "target": rng.choice(tt.TARGETS_PERSON),
        "short": rng.choice(tt.TARGETS_SHORT),
        "work_target": rng.choice(tt.TARGETS_WORK),
        "org": rng.choice(tt.FAKE_ORGS),
        "place": rng.choice(tt.FAKE_PLACES),
        "attr": rng.choice(tt.ATTR_GROUPS),
        "topic": rng.choice(tt.CRIT_TOPICS),
        "crit_org": rng.choice(tt.CRIT_ORGS),
        "scene": rng.choice(tt.WORK_SCENES),
        "time": rng.choice(tt.TIME_HINTS),
        "handle": rng.choice(tt.FAKE_HANDLES),
        "media": rng.choice(tt.MEDIA_SOURCES),
        "item": rng.choice(tt.GENERAL_ITEMS),
    }


def _join_two_sentences(rng: random.Random, s1: str, s2: str) -> str:
    """2文を自然な接続方式で連結する。"""
    r = rng.random()
    if r < 0.42:
        return f"{s1}{s2}"
    if r < 0.72:
        return f"{s1} {s2}"
    if r < 0.90:
        return f"{s1}\n{s2}"
    if not s1.endswith("？"):
        conn = rng.choice(["実際、", "しかも、", "なお、"])
        return f"{s1}{conn}{s2}"
    return f"{s1} {s2}"


def _apply_orthogonal_style(
    rng: random.Random,
    split: str,
    body: str,
    has_lead_allowed: bool = True,
) -> Tuple[str, str]:
    """有害・無害で完全に直交な文体バリエーションを付与する。"""
    text = body
    style_tags: List[str] = []

    # 先頭導入フレーズ（約 45%）
    if has_lead_allowed and rng.random() < 0.45:
        lead = rng.choice(tt.LEAD_PHRASES[split])
        if lead:
            text = f"{lead}{text}"
            style_tags.append("lead")

    # 末尾補足フレーズ（約 32%）
    if rng.random() < 0.32:
        tail = rng.choice(tt.TAIL_PHRASES[split])
        if tail:
            sep = rng.choice(["", " ", "\n"])
            text = f"{text}{sep}{tail}"
            style_tags.append("tail")

    # メンション・掲示板タグ接頭辞（約 16%、本文先頭に自然に付与）
    if rng.random() < 0.16:
        prefix = rng.choice([
            f"{rng.choice(tt.FAKE_HANDLES)} ",
            f"返信先 {rng.choice(tt.FAKE_HANDLES)}：",
            "【メモ】",
            "【投稿】",
            ">>1 ",
            ">>12 ",
        ])
        text = f"{prefix}{text}"
        style_tags.append("prefix")

    # 句点・読点の軽い表記ゆれ（約 15% で末尾句点落とし、約 8% で読点半角空白化）
    if text.endswith("。") and rng.random() < 0.15:
        text = text[:-1]
        style_tags.append("drop_period")
    if "、" in text and rng.random() < 0.08:
        text = text.replace("、", " ", 1)
        style_tags.append("comma_space")

    style_name = "+".join(style_tags) if style_tags else "plain"
    return text, style_name


def _build_split_plan(total_n: int) -> List[Tuple[str, Tuple[str, ...], str, int]]:
    """指定件数 total_n に対するカテゴリ別目標件数プランを作成する。

    戻り値リスト要素: (kind, active_labels, subtype, count)
      - kind: "single_toxic" | "multi_toxic" | "benign"
    """
    plan: List[Tuple[str, Tuple[str, ...], str, int]] = []

    # 1. 単一有害 7 ラベル (計 35.0% -> 各 5.0%)
    single_each = int(round(total_n * 0.05))
    single_total = single_each * len(tt.LABELS)
    for lb in tt.LABELS:
        plan.append(("single_toxic", (lb,), "single_" + lb, single_each))

    # 2. 無害 5 サブカテゴリ (計 45.0%)
    benign_ratios = [
        ("criticism", 0.12),
        ("news_quote", 0.12),
        ("debate", 0.09),
        ("support_legal", 0.06),
        ("general", 0.06),
    ]
    benign_total = 0
    for sub, ratio in benign_ratios:
        cnt = int(round(total_n * ratio))
        benign_total += cnt
        plan.append(("benign", (), sub, cnt))

    # 3. 多ラベル共起 14 パターン (残り 20.0%)
    cooccur_target = total_n - single_total - benign_total
    raw_counts = [cooccur_target * w for _pat, w in tt.COOCCUR_PATTERNS]
    int_counts = [int(x) for x in raw_counts]
    rem = cooccur_target - sum(int_counts)
    # 小数部が大きいパターンへ余りを配分
    order = sorted(
        range(len(tt.COOCCUR_PATTERNS)),
        key=lambda i: raw_counts[i] - int_counts[i],
        reverse=True,
    )
    for idx in order[:rem]:
        int_counts[idx] += 1

    for (pat, _w), cnt in zip(tt.COOCCUR_PATTERNS, int_counts):
        sub = "cooccur_" + "+".join(pat)
        plan.append(("multi_toxic", pat, sub, cnt))

    assert sum(x[3] for x in plan) == total_n
    return plan


def _render_candidate(
    rng: random.Random,
    split: str,
    kind: str,
    active_labels: Tuple[str, ...],
    subtype: str,
) -> Tuple[str, List[str], str]:
    """1件分のテキスト・使用テンプレートID群・文体タグを生成する。"""
    slots = _sample_context_slots(rng)
    tpl_ids: List[str] = []

    if kind == "single_toxic":
        lb = active_labels[0]
        pool = tt.SINGLE_TEMPLATES[lb][split]
        if rng.random() < 0.65:
            i1, i2 = rng.sample(range(len(pool)), 2)
            s1 = pool[i1].format(**slots)
            s2 = pool[i2].format(**slots)
            body = _join_two_sentences(rng, s1, s2)
            tpl_ids = [f"{split}_single_{lb}_{i1:02d}", f"{split}_single_{lb}_{i2:02d}"]
        else:
            i1 = rng.randrange(len(pool))
            body = pool[i1].format(**slots)
            tpl_ids = [f"{split}_single_{lb}_{i1:02d}"]

    elif kind == "benign":
        pool = tt.BENIGN_TEMPLATES[subtype][split]
        if rng.random() < 0.68:
            i1, i2 = rng.sample(range(len(pool)), 2)
            s1 = pool[i1].format(**slots)
            s2 = pool[i2].format(**slots)
            body = _join_two_sentences(rng, s1, s2)
            tpl_ids = [f"{split}_benign_{subtype}_{i1:02d}", f"{split}_benign_{subtype}_{i2:02d}"]
        else:
            i1 = rng.randrange(len(pool))
            body = pool[i1].format(**slots)
            tpl_ids = [f"{split}_benign_{subtype}_{i1:02d}"]

    elif kind == "multi_toxic":
        co_pool = tt.COOCCUR_TEMPLATES[active_labels][split]
        allow_compose = (active_labels != ("insult", "self_harm")) and (rng.random() < 0.35)
        if allow_compose:
            # 構成ラベルの単一テンプレート同士を同一コンテキストで結合
            lb_order = list(active_labels)
            rng.shuffle(lb_order)
            parts: List[str] = []
            for lb in lb_order:
                s_pool = tt.SINGLE_TEMPLATES[lb][split]
                idx = rng.randrange(len(s_pool))
                parts.append(s_pool[idx].format(**slots))
                tpl_ids.append(f"{split}_single_{lb}_{idx:02d}")
            body = parts[0]
            for p in parts[1:]:
                body = _join_two_sentences(rng, body, p)
        else:
            i1 = rng.randrange(len(co_pool))
            body = co_pool[i1].format(**slots)
            pat_tag = "+".join(active_labels)
            tpl_ids = [f"{split}_co_{pat_tag}_{i1:02d}"]
            # 確率 45% で同パターンまたは構成主ラベルの補強文を1文追加（insult+self_harm以外）
            if rng.random() < 0.45:
                if len(co_pool) >= 2 and rng.random() < 0.55:
                    other_idxs = [j for j in range(len(co_pool)) if j != i1]
                    i2 = rng.choice(other_idxs)
                    s2 = co_pool[i2].format(**slots)
                    body = _join_two_sentences(rng, body, s2)
                    tpl_ids.append(f"{split}_co_{pat_tag}_{i2:02d}")
                elif active_labels != ("insult", "self_harm"):
                    aux_lb = rng.choice(active_labels)
                    s_pool = tt.SINGLE_TEMPLATES[aux_lb][split]
                    idx = rng.randrange(len(s_pool))
                    s2 = s_pool[idx].format(**slots)
                    body = _join_two_sentences(rng, body, s2)
                    tpl_ids.append(f"{split}_single_{aux_lb}_{idx:02d}")
    else:
        raise ValueError(f"未定義の kind: {kind}")

    text, style_name = _apply_orthogonal_style(rng, split, body)
    return text, tpl_ids, style_name


def generate_split(
    split_name: str,
    total_n: int,
    seed: int,
    forbidden_lsh: Optional[LSHIndex] = None,
    forbidden_exact: Optional[Set[str]] = None,
    forbidden_norm: Optional[Set[str]] = None,
) -> Tuple[List[dict], dict]:
    """指定分割の合成データを重複・LSHリークなしで生成する。"""
    rng = random.Random(seed)
    plan = _build_split_plan(total_n)

    seen_exact: Set[str] = set(forbidden_exact) if forbidden_exact else set()
    seen_norm: Set[str] = set(forbidden_norm) if forbidden_norm else set()
    rows: List[dict] = []
    rejected_len = 0
    rejected_dup = 0
    rejected_lsh = 0

    for kind, active_labels, subtype, target_cnt in plan:
        got = 0
        attempts = 0
        max_attempts = max(target_cnt * 80, 2000)
        label_dict = {lb: (1 if lb in active_labels else 0) for lb in tt.LABELS}

        while got < target_cnt and attempts < max_attempts:
            attempts += 1
            text, tpl_ids, style_name = _render_candidate(
                rng, split_name, kind, active_labels, subtype
            )
            if not (15 <= len(text) <= 230):
                rejected_len += 1
                continue

            eh = hashlib.md5(text.encode("utf-8")).hexdigest()
            nh = hashlib.md5(normalize_text_for_dedup(text).encode("utf-8")).hexdigest()
            if eh in seen_exact or nh in seen_norm:
                rejected_dup += 1
                continue

            if forbidden_lsh is not None:
                sh = char_shingles(text)
                sig = compute_minhash(sh)
                if forbidden_lsh.has_near_dup(sh, sig, thr=0.80):
                    rejected_lsh += 1
                    continue

            seen_exact.add(eh)
            seen_norm.add(nh)
            rows.append(
                {
                    "id": "",  # シャッフル後に連番付与
                    "text": text,
                    "labels": dict(label_dict),
                    "active_labels": list(active_labels),
                    "is_toxic": len(active_labels) > 0,
                    "benign_subtype": None if active_labels else subtype,
                    "source_kind": kind,
                    "pattern": subtype,
                    "template_ids": tpl_ids,
                    "style": style_name,
                    "split": split_name,
                }
            )
            got += 1

        if got < target_cnt:
            raise RuntimeError(
                f"{split_name}/{subtype} の必要件数 {target_cnt} に対し {got} 件しか生成できませんでした "
                f"(attempts={attempts}, len_rej={rejected_len}, dup_rej={rejected_dup}, lsh_rej={rejected_lsh})"
            )

    rng.shuffle(rows)
    prefix = {"train": "tr", "dev": "dv", "sealed-test": "st"}[split_name]
    for idx, r in enumerate(rows):
        r["id"] = f"{prefix}-{idx:05d}"

    gen_meta = {
        "rejected_len": rejected_len,
        "rejected_dup": rejected_dup,
        "rejected_lsh": rejected_lsh,
    }
    return rows, gen_meta
