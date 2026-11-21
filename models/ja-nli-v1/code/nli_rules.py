#!/usr/bin/env python3
"""ja-nli-v1 共通規則フィルタ・テキスト正規化・MinHash LSH 検査モジュール。

生成直後・独立検証後・分割構築後・リーク検査の各工程で共通使用する。
"""
import hashlib
import re
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

VALID_LABELS = ("entailment", "neutral", "contradiction")

_RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]")
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_RE_PHONE = re.compile(r"(?<!\d)0\d{1,4}-\d{1,4}-\d{3,4}(?!\d)")
_RE_MARKUP = re.compile(r"```|`|https?://|www\.|<[A-Za-z/!][^>]*>|^\s*(?:#{1,6}\s|[-*+]\s|\*\*)")
_RE_BOLD = re.compile(r"\*\*[^*]+\*\*|__[^_]+__")
_RE_REPEAT4 = re.compile(r"(.)\1{3,}")
_RE_KANA = re.compile(r"[ぁ-んァ-ヶー]")

# メタ言及・データセット用語・文脈欠落表現の禁止リスト（仮説側に適用）
_META_EXACT_PATTERNS = [
    r"前提",
    r"仮説",
    r"含意",
    r"矛盾",
    r"中立",
    r"テキスト",
    r"与えられた",
    r"問題文",
    r"本文(?:に|で|の|から|は|を|と)",
    r"文章(?:では|に|の|は|から|を)",
    r"(?:この|当該)(?:文書|文章|テキスト|節|段落)",
    r"上記",
    r"(?:述べ|記載さ|記述さ|書か|記さ|示さ)れて(?:いる|いない|おり|おらず|いました|いません)",
    r"(?:^|[、。])以下(?:の(?:通り|とおり|表|図|点|内容|項目)|(?:では|において|に示))",
    r"本(?:文書|文章|白書|資料|セクション|章|節|ページ)",
]
_RE_META = re.compile("|".join(_META_EXACT_PATTERNS))

# 明らかな文頭指示語による独立文不成立（例：「これにより〜」「それらは〜」）のチェック
_RE_UNANCHORED_START = re.compile(
    r"^(?:これ(?:は|が|を|に|ら|により|によって)|それ(?:は|が|を|に|ら|により|によって)|"
    r"この結果|その結果|この点|その点|したがって、|よって、|なお、|また、|さらに、)"
)

# 第2工程強化用：文中・名詞句内の文脈欠落指示語および不自然な数詞（例：「10つの」）
_RE_UNANCHORED_NP = re.compile(
    r"(?:(?<![そど])(?:"
    r"この(?:レッスン|コース|チュートリアル|アプリ|アプリケーション|コード|スクリプト|"
    r"パイプライン|ガイドライン|プログラム|プラットフォーム|プロジェクト|ドキュメント|"
    r"記事|翻訳|ページ|セクション|演習|課題|パラメータ|メソッド|関数|クラス|オブジェクト|"
    r"メッセージ|アノテーション|トークナイザ|オプション|機能|仕組み|手法|アプローチ|"
    r"システム|モデル|法律|制度|事業|施策|取組|取り組み|支援策|会議|会食|調査|計画|方針|"
    r"規制|措置|キャンペーン)|"
    r"これらの(?:機能|演算|地域|協力|支援策|データ|手法|ツール|ファイル|パラメータ|項目|要素|設定|措置))"
    r"|(?<![ぁ-んァ-ヶ一-龥A-Za-z0-9々〆])本(?:プログラム|事業|制度|ガイドライン|プロジェクト|レッスン|コース|ツール|システム|法律|計画))"
)
_RE_BAD_COUNTER = re.compile(r"(?:^|[^\d])(?:[1-9]\d+)つの")

_RE_NORM_STRIP = re.compile(r"[\s、。．，,\.「」『』（）\(\)\[\]【】・：:；;！？!\?\-―ー]+")


def normalize_for_dedup(text: str) -> str:
    """重複・完全一致比較用に NFKC 正規化＋空白/記号除去を行う。"""
    t = unicodedata.normalize("NFKC", text or "").strip().lower()
    return _RE_NORM_STRIP.sub("", t)


def clean_surface(text: str) -> str:
    """基本的な文字種整理（制御文字なし前提、空白の連続圧縮）。"""
    t = unicodedata.normalize("NFKC", text or "")
    t = re.sub(r"[ \t\u3000]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


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
    pairs = [("「", "」"), ("『", "』"), ("（", "）"), ("(", ")")]
    for op, cl in pairs:
        if text.count(op) != text.count(cl):
            return False
    return True


def char_ngrams(text: str, n: int = 4) -> Set[str]:
    t = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def ngram_containment(hyp: str, prem: str, n: int = 4) -> float:
    """仮説の文字 n-gram のうち前提に含まれる割合を返す。"""
    h_ng = char_ngrams(hyp, n)
    if not h_ng:
        return 0.0
    p_ng = char_ngrams(prem, n)
    return len(h_ng & p_ng) / len(h_ng)


def ngram_jaccard(a: str, b: str, n: int = 4) -> float:
    sa = char_ngrams(a, n)
    sb = char_ngrams(b, n)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / (len(sa) + len(sb) - inter)


def check_pair_rules(
    premise: str,
    hypothesis: str,
    label: str,
    *,
    min_premise_len: int = 50,
    max_premise_len: int = 1600,
    min_hyp_len: int = 12,
    max_hyp_len: int = 160,
    max_copy_containment: float = 0.85,
    strict_demonstratives: bool = False,
) -> Tuple[bool, List[str]]:
    """前提・仮説・ラベルに対する決定論的規則フィルタ。

    Returns:
        (passed, failure_reasons)
    """
    reasons: List[str] = []

    if not isinstance(premise, str) or not isinstance(hypothesis, str) or label not in VALID_LABELS:
        return False, ["bad_schema_or_label"]

    p = premise.strip()
    h = hypothesis.strip()
    if not p or not h:
        return False, ["empty_text"]

    # 1. 長さ制限
    if not (min_premise_len <= len(p) <= max_premise_len):
        reasons.append("premise_len_out_of_range")
    if not (min_hyp_len <= len(h) <= max_hyp_len):
        reasons.append("hypothesis_len_out_of_range")

    # 2. 制御文字・文字化け・PII パターン
    if _RE_CTRL.search(p) or _RE_CTRL.search(h):
        reasons.append("ctrl_or_mojibake")
    if _RE_EMAIL.search(p) or _RE_EMAIL.search(h) or _RE_PHONE.search(p) or _RE_PHONE.search(h):
        reasons.append("pii_pattern")

    # 3. 日本語自然性・マークアップ混入・括弧・文末・反復
    if not _RE_KANA.search(h):
        reasons.append("hypothesis_no_kana")
    if japanese_char_ratio(h) < 0.55:
        reasons.append("hypothesis_low_jp_ratio")
    if japanese_char_ratio(p) < 0.50:
        reasons.append("premise_low_jp_ratio")
    if _RE_MARKUP.search(h) or _RE_BOLD.search(h) or "\n" in h:
        reasons.append("hypothesis_markup_or_newline")
    if not brackets_balanced(h):
        reasons.append("hypothesis_unbalanced_brackets")
    if h[-1] not in ("。", "？", "！"):
        reasons.append("hypothesis_bad_ending")
    # 仮説は原則1文（終端以外の句点・複数文連結を制限、ただし引用「〜。」を除く簡易判定）
    if "。" in h[:-1]:
        # 括弧外の句点があれば複数文とみなす
        depth = 0
        multi_sent = False
        for idx, ch in enumerate(h[:-1]):
            if ch in ("「", "『", "（"):
                depth += 1
            elif ch in ("」", "』", "）") and depth > 0:
                depth -= 1
            elif ch == "。" and depth == 0:
                multi_sent = True
                break
        if multi_sent:
            reasons.append("hypothesis_multi_sentence")
    if _RE_REPEAT4.search(h):
        reasons.append("hypothesis_char_repeat")

    # 4. メタ言及・文頭指示語・強化検査
    if _RE_META.search(h):
        reasons.append("hypothesis_meta_leak")
    if _RE_UNANCHORED_START.search(h):
        reasons.append("hypothesis_unanchored_start")
    if strict_demonstratives:
        if _RE_UNANCHORED_NP.search(h):
            reasons.append("hypothesis_unanchored_np")
        if _RE_BAD_COUNTER.search(h):
            reasons.append("hypothesis_bad_counter")

    # 5. 丸コピー・過度抜粋
    norm_p = normalize_for_dedup(p)
    norm_h = normalize_for_dedup(h)
    if norm_h and norm_h in norm_p:
        reasons.append("exact_substring_copy")
    elif ngram_containment(h, p, n=4) >= max_copy_containment:
        reasons.append("high_copy_containment")

    return (len(reasons) == 0), reasons


def filter_intra_premise_similar(
    items: List[dict], jaccard_thresh: float = 0.75
) -> Tuple[List[dict], int]:
    """同一前提内の候補間で高類似仮説ペアを除外する（E/N/C 各1件想定）。"""
    kept: List[dict] = []
    dropped = 0
    for it in items:
        h = it.get("hypothesis", "")
        dup = False
        for k in kept:
            if ngram_jaccard(h, k.get("hypothesis", ""), n=4) >= jaccard_thresh:
                dup = True
                break
        if dup:
            dropped += 1
        else:
            kept.append(it)
    return kept, dropped


def filter_intra_chunk_hypotheses(
    items: List[dict],
    *,
    same_class_thresh: float = 0.65,
    cross_class_thresh: float = 0.88,
) -> Tuple[List[dict], int]:
    """同一チャンク内の複数パス候補を検査し、正規化重複・同クラス近似・異クラス過度近似を除外する。"""
    kept: List[dict] = []
    dropped = 0
    for it in items:
        h = it.get("hypothesis", "")
        nh = normalize_for_dedup(h)
        lbl = it.get("intended_label")
        dup = False
        for k in kept:
            kh = k.get("hypothesis", "")
            if nh and nh == normalize_for_dedup(kh):
                dup = True
                break
            jac = ngram_jaccard(h, kh, n=4)
            thresh = same_class_thresh if lbl == k.get("intended_label") else cross_class_thresh
            if jac >= thresh:
                dup = True
                break
        if dup:
            dropped += 1
        else:
            kept.append(it)
    return kept, dropped


# --- 決定的 MinHash LSH（分割前クラスタリング＆リーク検査用） ---

def _mix_hash(val: int, seed_a: int, seed_b: int) -> int:
    # 64-bit universal hash over 2^61-1 prime approx via bit masking
    x = ((val * seed_a + seed_b) & 0xFFFFFFFFFFFFFFFF)
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
_PERM_NP_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
_MIX_MUL_C = np.uint64(0xFF51AFD7ED558CCD)


def compute_minhash_sig(
    text: str, ngram_n: int = 5, num_perm: int = 64, seed: int = 42
) -> Tuple[int, ...]:
    """テキストの文字 n-gram MinHash シグネチャを返す（numpy uint64 ベクトル化版）。"""
    key = (num_perm, seed)
    if key not in _PERM_CACHE:
        _PERM_CACHE[key] = _make_perm_params(num_perm, seed)
    if key not in _PERM_NP_CACHE:
        params = _PERM_CACHE[key]
        a_arr = np.array([a for a, _ in params], dtype=np.uint64)[:, None]
        b_arr = np.array([b for _, b in params], dtype=np.uint64)[:, None]
        _PERM_NP_CACHE[key] = (a_arr, b_arr)
    a_arr, b_arr = _PERM_NP_CACHE[key]

    grams = char_ngrams(normalize_for_dedup(text) or text, n=ngram_n)
    if not grams:
        return tuple(0 for _ in range(num_perm))
    hv = np.fromiter(
        (int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little") for g in grams),
        dtype=np.uint64,
        count=len(grams),
    )[None, :]
    with np.errstate(over="ignore"):
        x = hv * a_arr + b_arr
        x ^= x >> np.uint64(33)
        x *= _MIX_MUL_C
        x ^= x >> np.uint64(33)
    return tuple(int(v) for v in x.min(axis=1))


def lsh_find_near_duplicates(
    texts: Sequence[str],
    *,
    ngram_n: int = 5,
    num_perm: int = 64,
    bands: int = 16,
    seed: int = 42,
    jaccard_thresh: float = 0.70,
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
        # バケットが極端に大きい場合を安全制限
        if len(idxs) > 200:
            idxs = idxs[:200]
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
    jaccard_thresh: float = 0.80,
) -> int:
    """2つの分割 A と B 間の近傍重複ペア数を数える。"""
    rows = num_perm // bands
    assert bands * rows == num_perm
    n_a = len(texts_a)
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
