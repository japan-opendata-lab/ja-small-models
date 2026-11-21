#!/usr/bin/env python3
"""ja-sentiment-v1 共通規則フィルタ・テキスト正規化・MinHash LSH 検査モジュール。

生成直後・独立検証後・分割構築後・リーク検査の各工程で共通使用する。
"""
import hashlib
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

VALID_LABELS = ("negative", "neutral", "positive")
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

_RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]")
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_RE_PHONE = re.compile(r"(?<!\d)0\d{1,4}-\d{1,4}-\d{3,4}(?!\d)")
_RE_MARKUP = re.compile(r"```|`|https?://|www\.|<[A-Za-z/!][^>]*>|^\s*(?:#{1,6}\s|[-*+]\s|\*\*)")
_RE_BOLD = re.compile(r"\*\*[^*]+\*\*|__[^_]+__")
# 同一文字5連続以上を異常反復とする（「ーーー」「…」等の軽微な3-4連続は許容しスパムを除外）
_RE_REPEAT5 = re.compile(r"(.)\1{4,}")
_RE_KANA = re.compile(r"[ぁ-んァ-ヶー]")

# プロンプト指示・ラベル定義の漏洩（自然な感想中の「ポジティブな気分」等を誤爆しないよう明確なメタ構文を検査）
_META_PATTERNS = [
    r"感情極性",
    r"サブタイプ",
    r"皮肉めいた表現",
    r"両面評価",
    r"否定形による肯定",
    r"控えめな不満",
    r"感情の薄い事実",
    r"^(?:ポジティブ|ネガティブ|中立|positive|negative|neutral)\s*[：:]",
    r"^(?:出力|レビュー文|投稿文|テキスト|回答|判定|ラベル)\s*[：:]",
    r"(?:この|上記の|以下の)(?:レビュー|文章|投稿|テキスト)は(?:ポジティブ|ネガティブ|中立)",
    r"ラベル(?:は|が|として)(?:ポジティブ|ネガティブ|中立)",
    r"【(?:ポジティブ|ネガティブ|中立|出力|制約|条件)】",
]
_RE_META = re.compile("|".join(_META_PATTERNS), re.IGNORECASE)

# 暴言・自傷・危険表現の抑止（合成レビュー内の過度な攻撃を除外）
_RE_TOXIC = re.compile(r"死ね|殺す|ぶっ殺|死んでくれ|クソ野郎|ゴミクズ|キモい死")

_RE_NORM_STRIP = re.compile(r"[\s、。．，,\.「」『』（）\(\)\[\]【】・：:；;！？!\?\-―ー〜~]+")


def normalize_for_dedup(text: str) -> str:
    """重複・完全一致比較用に NFKC 正規化＋空白/記号除去を行う。"""
    t = unicodedata.normalize("NFKC", text or "").strip().lower()
    return _RE_NORM_STRIP.sub("", t)


def clean_surface(text: str) -> str:
    """基本的な文字種整理（連続空白の整理・改行圧縮）。"""
    t = unicodedata.normalize("NFKC", text or "")
    t = re.sub(r"[ \t\u3000]+", " ", t)
    t = re.sub(r"\n+", " ", t)
    t = t.strip().strip('"').strip("'")
    if len(t) >= 2 and t[0] == "「" and t[-1] == "」" and t.count("「") == 1 and t.count("」") == 1:
        # 文全体が不要なカギ括弧で囲まれている場合は外す
        t = t[1:-1].strip()
    return t


def is_jp_char(ch: str) -> bool:
    c = ord(ch)
    return (
        0x3040 <= c <= 0x309F  # ひらがな
        or 0x30A0 <= c <= 0x30FF  # カタカナ
        or 0x4E00 <= c <= 0x9FFF  # CJK統合漢字
        or 0x3400 <= c <= 0x4DBF  # CJK拡張A
        or ch in "、。？！「」『』（）・ー〜々〆〇ヵヶ"
    )


def japanese_char_ratio(text: str) -> float:
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    jp = sum(1 for c in non_ws if is_jp_char(c))
    return jp / len(non_ws)


def brackets_balanced(text: str) -> bool:
    pairs = [("「", "」"), ("『", "』"), ("（", "）"), ("(", ")"), ("【", "】")]
    for op, cl in pairs:
        if text.count(op) != text.count(cl):
            return False
    return True


def char_ngrams(text: str, n: int = 4) -> Set[str]:
    t = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def ngram_jaccard(a: str, b: str, n: int = 4) -> float:
    sa = char_ngrams(a, n)
    sb = char_ngrams(b, n)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / (len(sa) + len(sb) - inter)


def check_text_rules(
    text: str,
    label: str,
    domain: str = "",
    length_band: str = "",
    *,
    min_len: int = 12,
    max_len: int = 380,
) -> Tuple[bool, List[str]]:
    """感情極性テキスト・ラベルに対する決定論的規則フィルタ。"""
    reasons: List[str] = []

    if not isinstance(text, str) or label not in VALID_LABELS:
        return False, ["bad_schema_or_label"]

    t = text.strip()
    if not t:
        return False, ["empty_text"]

    # 1. 長さ制限（全体 + 長さ帯との緩やかな整合確認）
    if not (min_len <= len(t) <= max_len):
        reasons.append("len_out_of_range")
    elif length_band == "short" and len(t) > 75:
        reasons.append("len_band_mismatch_short")
    elif length_band == "medium" and not (28 <= len(t) <= 185):
        reasons.append("len_band_mismatch_medium")
    elif length_band == "long" and len(t) < 55:
        reasons.append("len_band_mismatch_long")

    # 2. 制御文字・文字化け・PII・マークアップ
    if _RE_CTRL.search(t):
        reasons.append("ctrl_or_mojibake")
    if _RE_EMAIL.search(t) or _RE_PHONE.search(t):
        reasons.append("pii_pattern")
    if _RE_MARKUP.search(t) or _RE_BOLD.search(t) or "\n" in t:
        reasons.append("markup_or_newline")

    # 3. 日本語自然性・括弧・反復
    if not _RE_KANA.search(t):
        reasons.append("no_kana")
    if japanese_char_ratio(t) < 0.50:
        reasons.append("low_jp_ratio")
    if not brackets_balanced(t):
        reasons.append("unbalanced_brackets")
    if _RE_REPEAT5.search(t):
        reasons.append("char_repeat")

    # 4. メタ漏洩・過激表現
    if _RE_META.search(t):
        reasons.append("meta_leak")
    if _RE_TOXIC.search(t):
        reasons.append("toxic_word")

    return (len(reasons) == 0), reasons


def filter_intra_batch_similar(
    items: List[dict], text_key: str = "text", jaccard_thresh: float = 0.65
) -> Tuple[List[dict], int]:
    """同一レシピ／バッチ内の生成文間で高類似重複を除外する。"""
    kept: List[dict] = []
    dropped = 0
    for it in items:
        txt = it.get(text_key, "")
        dup = False
        for k in kept:
            if ngram_jaccard(txt, k.get(text_key, ""), n=4) >= jaccard_thresh:
                dup = True
                break
        if dup:
            dropped += 1
        else:
            kept.append(it)
    return kept, dropped


# --- 決定的 MinHash LSH（重複除去＆分割間リーク検査用） ---

def _mix_hash(val: int, seed_a: int, seed_b: int) -> int:
    x = (val * seed_a + seed_b) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 33
    return x


def _make_perm_params(num_perm: int = 64, seed: int = 42) -> List[Tuple[int, int]]:
    params = []
    for i in range(num_perm):
        h1 = hashlib.blake2b(f"a:{seed}:{i}".encode(), digest_size=8).digest()
        h2 = hashlib.blake2b(f"b:{seed}:{i}".encode(), digest_size=8).digest()
        a = int.from_bytes(h1, "little") | 1
        b = int.from_bytes(h2, "little")
        params.append((a, b))
    return params


_PERM_CACHE: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}


def compute_minhash_sig(
    text: str, ngram_n: int = 5, num_perm: int = 64, seed: int = 42
) -> Tuple[int, ...]:
    """テキストの文字 n-gram MinHash シグネチャを返す。"""
    key = (num_perm, seed)
    if key not in _PERM_CACHE:
        _PERM_CACHE[key] = _make_perm_params(num_perm, seed)
    params = _PERM_CACHE[key]

    grams = char_ngrams(normalize_for_dedup(text) or text, n=ngram_n)
    if not grams:
        return tuple(0 for _ in range(num_perm))
    base_hashes = [
        int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little")
        for g in grams
    ]
    sig = []
    for a, b in params:
        mv = min(_mix_hash(hv, a, b) for hv in base_hashes)
        sig.append(mv)
    return tuple(sig)


def lsh_find_near_duplicates(
    texts: Sequence[str],
    *,
    ngram_n: int = 5,
    num_perm: int = 64,
    bands: int = 16,
    seed: int = 42,
    jaccard_thresh: float = 0.75,
) -> List[Tuple[int, int, float]]:
    """全テキスト中の近傍重複ペア (i, j, jaccard) を検出する。"""
    rows = num_perm // bands
    assert bands * rows == num_perm
    sigs = [compute_minhash_sig(t, ngram_n=ngram_n, num_perm=num_perm, seed=seed) for t in texts]
    grams_list = [char_ngrams(normalize_for_dedup(t) or t, n=ngram_n) for t in texts]

    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for b in range(bands):
            st = b * rows
            buckets[(b, sig[st : st + rows])].append(idx)

    cand_pairs: Set[Tuple[int, int]] = set()
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        if len(idxs) > 250:
            idxs = idxs[:250]
        for a_i in range(len(idxs)):
            u = idxs[a_i]
            for b_i in range(a_i + 1, len(idxs)):
                v = idxs[b_i]
                cand_pairs.add((u, v) if u < v else (v, u))

    hits: List[Tuple[int, int, float]] = []
    for u, v in sorted(cand_pairs):
        su = grams_list[u]
        sv = grams_list[v]
        if not su or not sv:
            continue
        inter = len(su & sv)
        jac = inter / (len(su) + len(sv) - inter)
        if jac >= jaccard_thresh:
            hits.append((u, v, jac))
    return hits


def lsh_cross_split_leaks(
    texts_a: Sequence[str],
    texts_b: Sequence[str],
    *,
    ngram_n: int = 5,
    num_perm: int = 64,
    bands: int = 16,
    seed: int = 42,
    jaccard_thresh: float = 0.70,
) -> int:
    """2つの分割 A と B 間の近傍重複ペア数を数える。"""
    rows = num_perm // bands
    assert bands * rows == num_perm
    grams_a = [char_ngrams(normalize_for_dedup(t) or t, n=ngram_n) for t in texts_a]
    grams_b = [char_ngrams(normalize_for_dedup(t) or t, n=ngram_n) for t in texts_b]

    buckets_a: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
    for idx, t in enumerate(texts_a):
        sig = compute_minhash_sig(t, ngram_n=ngram_n, num_perm=num_perm, seed=seed)
        for b in range(bands):
            st = b * rows
            buckets_a[(b, sig[st : st + rows])].append(idx)

    seen_pairs: Set[Tuple[int, int]] = set()
    for idx_b, t in enumerate(texts_b):
        sig = compute_minhash_sig(t, ngram_n=ngram_n, num_perm=num_perm, seed=seed)
        sb = grams_b[idx_b]
        if not sb:
            continue
        for b in range(bands):
            st = b * rows
            for idx_a in buckets_a.get((b, sig[st : st + rows]), ()):
                pair = (idx_a, idx_b)
                if pair in seen_pairs:
                    continue
                sa = grams_a[idx_a]
                if not sa:
                    continue
                inter = len(sa & sb)
                jac = inter / (len(sa) + len(sb) - inter)
                if jac >= jaccard_thresh:
                    seen_pairs.add(pair)
    return len(seen_pairs)
