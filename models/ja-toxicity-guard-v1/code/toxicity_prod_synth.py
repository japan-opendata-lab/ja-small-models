#!/usr/bin/env python3
"""第8号 ja-toxicity-guard-v1 本番データ合成・高速MinHash LSH・安全監査モジュール。

- 本番用テンプレート (toxicity_prod_templates) のみを使用し、パイロット材料とは完全非重複。
- NumPy ベクトル化 MinHash LSH (num_perm=64, bands=16, rows_per=4, seed=42) により
  110,000件規模の分割間・対パイロット近傍重複検査 (Jaccard >= 0.80) を高速かつ厳密に実行する。
- 全件に対して実在固有名詞・連絡先・自傷具体手段 0 件の安全監査を実施する。
"""
import hashlib
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

import toxicity_prod_templates as pt

# =====================================================================
# 高速 NumPy ベクトル化 MinHash LSH (2^31 - 1 法により int64 オーバーフロー防止)
# =====================================================================
LSH_NUM_PERM = 64
LSH_BANDS = 16
LSH_ROWS_PER = 4
LSH_SEED = 42
M_PRIME = 2147483647  # 2^31 - 1

_rng_lsh = random.Random(LSH_SEED)
_LSH_A = np.array([_rng_lsh.randrange(1, M_PRIME) for _ in range(LSH_NUM_PERM)], dtype=np.int64)
_LSH_B = np.array([_rng_lsh.randrange(0, M_PRIME) for _ in range(LSH_NUM_PERM)], dtype=np.int64)
_SHINGLE_HASH_CACHE: Dict[str, int] = {}


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


def _shingle_id(g: str) -> int:
    hid = _SHINGLE_HASH_CACHE.get(g)
    if hid is None:
        hid = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16) % M_PRIME
        _SHINGLE_HASH_CACHE[g] = hid
    return hid


def compute_minhash(shingles: Set[str]) -> Tuple[int, ...]:
    """文字 n-gram 集合から 64 次元 MinHash シグネチャを算出する。"""
    if not shingles:
        return (M_PRIME,) * LSH_NUM_PERM
    h = np.array([_shingle_id(g) for g in shingles], dtype=np.int64)
    sig = ((h[:, None] * _LSH_A + _LSH_B) % M_PRIME).min(axis=0)
    return tuple(int(x) for x in sig)


class LSHIndex:
    """高速 MinHash LSH インデックス。"""

    def __init__(self, bands: int = LSH_BANDS, rows_per: int = LSH_ROWS_PER) -> None:
        self.bands = bands
        self.rows_per = rows_per
        self.buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
        self.shingles_list: List[Set[str]] = []

    def add(self, shingles: Set[str], sig: Tuple[int, ...]) -> int:
        idx = len(self.shingles_list)
        self.shingles_list.append(shingles)
        for b in range(self.bands):
            key = (b, sig[b * self.rows_per : (b + 1) * self.rows_per])
            self.buckets[key].append(idx)
        return idx

    def has_near_dup(self, shingles: Set[str], sig: Tuple[int, ...], thr: float = 0.80) -> bool:
        if not shingles:
            return False
        cand: Set[int] = set()
        for b in range(self.bands):
            key = (b, sig[b * self.rows_per : (b + 1) * self.rows_per])
            lst = self.buckets.get(key)
            if lst:
                cand.update(lst)
        if not cand:
            return False
        for idx in cand:
            other = self.shingles_list[idx]
            inter = len(shingles & other)
            if inter == 0:
                continue
            union = len(shingles) + len(other) - inter
            if union > 0 and inter / union >= thr:
                return True
        return False


def build_shingle_sigs(rows: Sequence[dict]) -> List[Tuple[Set[str], Tuple[int, ...]]]:
    """行リストに対し (shingles, minhash_sig) を一括計算して返す。"""
    out: List[Tuple[Set[str], Tuple[int, ...]]] = []
    for r in rows:
        sh = char_shingles(r["text"])
        sig = compute_minhash(sh)
        out.append((sh, sig))
    return out


def count_lsh_leaks_from_sigs(
    sigs_a: Sequence[Tuple[Set[str], Tuple[int, ...]]],
    sigs_b: Sequence[Tuple[Set[str], Tuple[int, ...]]],
    thr: float = 0.80,
) -> int:
    """事前計算済み (shingles, sig) を用いて2群間の Jaccard >= thr 件数を返す。"""
    index = LSHIndex()
    for sh, sig in sigs_b:
        index.add(sh, sig)
    hits = 0
    for sh, sig in sigs_a:
        if index.has_near_dup(sh, sig, thr=thr):
            hits += 1
    return hits


# =====================================================================
# 安全・倫理自動監査ルール（全110,000件対象）
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
    """実在固有名詞・実在連絡先・自傷具体手段の混入がないか全件監査する。"""
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
# 本番データ生成ロジック
# =====================================================================
def _sample_prod_slots(rng: random.Random) -> Dict[str, str]:
    """1文書内で一貫させる架空コンテキストスロットをサンプルする。"""
    return {
        "target": rng.choice(pt.PROD_TARGETS_PERSON),
        "short": rng.choice(pt.PROD_TARGETS_SHORT),
        "work_target": rng.choice(pt.PROD_TARGETS_WORK),
        "org": rng.choice(pt.PROD_FAKE_ORGS),
        "place": rng.choice(pt.PROD_FAKE_PLACES),
        "attr": rng.choice(pt.PROD_ATTR_GROUPS),
        "topic": rng.choice(pt.PROD_CRIT_TOPICS),
        "crit_org": rng.choice(pt.PROD_CRIT_ORGS),
        "scene": rng.choice(pt.PROD_WORK_SCENES),
        "time": rng.choice(pt.PROD_TIME_HINTS),
        "handle": rng.choice(pt.PROD_FAKE_HANDLES),
        "media": rng.choice(pt.PROD_MEDIA_SOURCES),
        "item": rng.choice(pt.PROD_GENERAL_ITEMS),
    }


def _join_two_sentences(rng: random.Random, s1: str, s2: str) -> str:
    r = rng.random()
    if r < 0.42:
        return f"{s1}{s2}"
    if r < 0.72:
        return f"{s1} {s2}"
    if r < 0.90:
        return f"{s1}\n{s2}"
    if not s1.endswith("？"):
        conn = rng.choice(["実際、", "しかも、", "なお、", "その上、"])
        return f"{s1}{conn}{s2}"
    return f"{s1} {s2}"


def _apply_orthogonal_style(
    rng: random.Random,
    split: str,
    body: str,
) -> Tuple[str, str]:
    """有害・無害で完全に直交な文体バリエーションを付与する。"""
    text = body
    style_tags: List[str] = []

    if rng.random() < 0.46:
        lead = rng.choice(pt.PROD_LEAD_PHRASES[split])
        if lead:
            text = f"{lead}{text}"
            style_tags.append("lead")

    if rng.random() < 0.34:
        tail = rng.choice(pt.PROD_TAIL_PHRASES[split])
        if tail:
            sep = rng.choice(["", " ", "\n"])
            text = f"{text}{sep}{tail}"
            style_tags.append("tail")

    if rng.random() < 0.16:
        prefix = rng.choice([
            f"{rng.choice(pt.PROD_FAKE_HANDLES)} ",
            f"宛先 {rng.choice(pt.PROD_FAKE_HANDLES)}：",
            "【記録】",
            "【コメント】",
            ">>3 ",
            ">>24 ",
        ])
        text = f"{prefix}{text}"
        style_tags.append("prefix")

    if text.endswith("。") and rng.random() < 0.15:
        text = text[:-1]
        style_tags.append("drop_period")
    if "、" in text and rng.random() < 0.08:
        text = text.replace("、", " ", 1)
        style_tags.append("comma_space")

    style_name = "+".join(style_tags) if style_tags else "plain"
    return text, style_name


def _build_prod_plan(total_n: int) -> List[Tuple[str, Tuple[str, ...], str, int]]:
    """本番構成比（debate を 12.0% へ強化）に基づく目標件数プランを作成する。"""
    plan: List[Tuple[str, Tuple[str, ...], str, int]] = []

    # 1. 単一有害 7 ラベル (計 35.0% -> 各 5.0%)
    single_each = int(round(total_n * 0.05))
    single_total = single_each * len(pt.LABELS)
    for lb in pt.LABELS:
        plan.append(("single_toxic", (lb,), "single_" + lb, single_each))

    # 2. 無害 5 サブカテゴリ (計 45.0%、debate を 12.0% に増強)
    benign_ratios = [
        ("criticism", 0.12),
        ("news_quote", 0.12),
        ("debate", 0.12),
        ("support_legal", 0.05),
        ("general", 0.04),
    ]
    benign_total = 0
    for sub, ratio in benign_ratios:
        cnt = int(round(total_n * ratio))
        benign_total += cnt
        plan.append(("benign", (), sub, cnt))

    # 3. 多ラベル共起 14 パターン (計 20.0%)
    cooccur_target = total_n - single_total - benign_total
    raw_counts = [cooccur_target * w for _pat, w in pt.COOCCUR_PATTERNS]
    int_counts = [int(x) for x in raw_counts]
    rem = cooccur_target - sum(int_counts)
    order = sorted(
        range(len(pt.COOCCUR_PATTERNS)),
        key=lambda i: raw_counts[i] - int_counts[i],
        reverse=True,
    )
    for idx in order[:rem]:
        int_counts[idx] += 1

    for (pat, _w), cnt in zip(pt.COOCCUR_PATTERNS, int_counts):
        sub = "cooccur_" + "+".join(pat)
        plan.append(("multi_toxic", pat, sub, cnt))

    assert sum(x[3] for x in plan) == total_n
    return plan


def _render_prod_candidate(
    rng: random.Random,
    split: str,
    kind: str,
    active_labels: Tuple[str, ...],
    subtype: str,
) -> Tuple[str, List[str], str]:
    slots = _sample_prod_slots(rng)
    tpl_ids: List[str] = []

    if kind == "single_toxic":
        lb = active_labels[0]
        pool = pt.PROD_SINGLE_TEMPLATES[lb][split]
        if rng.random() < 0.65:
            i1, i2 = rng.sample(range(len(pool)), 2)
            s1 = pool[i1].format(**slots)
            s2 = pool[i2].format(**slots)
            body = _join_two_sentences(rng, s1, s2)
            tpl_ids = [f"prod_{split[:2]}_single_{lb}_{i1:02d}", f"prod_{split[:2]}_single_{lb}_{i2:02d}"]
        else:
            i1 = rng.randrange(len(pool))
            body = pool[i1].format(**slots)
            tpl_ids = [f"prod_{split[:2]}_single_{lb}_{i1:02d}"]

    elif kind == "benign":
        pool = pt.PROD_BENIGN_TEMPLATES[subtype][split]
        if rng.random() < 0.68:
            i1, i2 = rng.sample(range(len(pool)), 2)
            s1 = pool[i1].format(**slots)
            s2 = pool[i2].format(**slots)
            body = _join_two_sentences(rng, s1, s2)
            tpl_ids = [f"prod_{split[:2]}_benign_{subtype}_{i1:02d}", f"prod_{split[:2]}_benign_{subtype}_{i2:02d}"]
        else:
            i1 = rng.randrange(len(pool))
            body = pool[i1].format(**slots)
            tpl_ids = [f"prod_{split[:2]}_benign_{subtype}_{i1:02d}"]

    elif kind == "multi_toxic":
        co_pool = pt.PROD_COOCCUR_TEMPLATES[active_labels][split]
        allow_compose = (active_labels != ("insult", "self_harm")) and (rng.random() < 0.35)
        if allow_compose:
            lb_order = list(active_labels)
            rng.shuffle(lb_order)
            parts: List[str] = []
            for lb in lb_order:
                s_pool = pt.PROD_SINGLE_TEMPLATES[lb][split]
                idx = rng.randrange(len(s_pool))
                parts.append(s_pool[idx].format(**slots))
                tpl_ids.append(f"prod_{split[:2]}_single_{lb}_{idx:02d}")
            body = parts[0]
            for p in parts[1:]:
                body = _join_two_sentences(rng, body, p)
        else:
            i1 = rng.randrange(len(co_pool))
            body = co_pool[i1].format(**slots)
            pat_tag = "+".join(active_labels)
            tpl_ids = [f"prod_{split[:2]}_co_{pat_tag}_{i1:02d}"]
            if rng.random() < 0.45:
                if len(co_pool) >= 2 and rng.random() < 0.55:
                    other_idxs = [j for j in range(len(co_pool)) if j != i1]
                    i2 = rng.choice(other_idxs)
                    s2 = co_pool[i2].format(**slots)
                    body = _join_two_sentences(rng, body, s2)
                    tpl_ids.append(f"prod_{split[:2]}_co_{pat_tag}_{i2:02d}")
                elif active_labels != ("insult", "self_harm"):
                    aux_lb = rng.choice(active_labels)
                    s_pool = pt.PROD_SINGLE_TEMPLATES[aux_lb][split]
                    idx = rng.randrange(len(s_pool))
                    s2 = s_pool[idx].format(**slots)
                    body = _join_two_sentences(rng, body, s2)
                    tpl_ids.append(f"prod_{split[:2]}_single_{aux_lb}_{idx:02d}")
    else:
        raise ValueError(f"未定義の kind: {kind}")

    text, style_name = _apply_orthogonal_style(rng, split, body)
    return text, tpl_ids, style_name


def generate_prod_split(
    split_name: str,
    total_n: int,
    seed: int,
    forbidden_lsh: Optional[LSHIndex] = None,
    forbidden_exact: Optional[Set[str]] = None,
    forbidden_norm: Optional[Set[str]] = None,
) -> Tuple[List[dict], dict]:
    """指定分割の本番データを重複・LSHリークなしで生成する。"""
    rng = random.Random(seed)
    plan = _build_prod_plan(total_n)

    seen_exact: Set[str] = set(forbidden_exact) if forbidden_exact else set()
    seen_norm: Set[str] = set(forbidden_norm) if forbidden_norm else set()
    rows: List[dict] = []
    rejected_len = 0
    rejected_dup = 0
    rejected_lsh = 0

    for kind, active_labels, subtype, target_cnt in plan:
        got = 0
        attempts = 0
        max_attempts = max(target_cnt * 120, 5000)
        label_dict = {lb: (1 if lb in active_labels else 0) for lb in pt.LABELS}

        while got < target_cnt and attempts < max_attempts:
            attempts += 1
            text, tpl_ids, style_name = _render_prod_candidate(
                rng, split_name, kind, active_labels, subtype
            )
            if not (15 <= len(text) <= 235):
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
                    "id": "",
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
    prefix = {"train": "ptr", "dev": "pdv", "sealed-test": "pst"}[split_name]
    for idx, r in enumerate(rows):
        r["id"] = f"{prefix}-{idx:06d}"

    gen_meta = {
        "rejected_len": rejected_len,
        "rejected_dup": rejected_dup,
        "rejected_lsh": rejected_lsh,
    }
    return rows, gen_meta
