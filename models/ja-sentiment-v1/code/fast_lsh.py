#!/usr/bin/env python3
"""第12号 ja-sentiment-v1 本番10万件規模向け 高速ベクトル化 MinHash LSH モジュール。

sentiment_rules.py と同一の正規化・文字n-gram・blake2b ベース MinHash (64 perm, 16 bands) を
NumPy uint64 ベクトル演算＋ベースハッシュキャッシュで高速化する。
"""
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sentiment_rules as sr  # noqa: E402

_CONST_MIX = np.uint64(0xFF51AFD7ED558CCD)
_SHIFT33 = np.uint64(33)
_PARAM_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}


def _get_np_perm_params(num_perm: int = 64, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    key = (num_perm, seed)
    if key not in _PARAM_CACHE:
        pairs = sr._make_perm_params(num_perm=num_perm, seed=seed)
        a = np.array([p[0] for p in pairs], dtype=np.uint64)
        b = np.array([p[1] for p in pairs], dtype=np.uint64)
        _PARAM_CACHE[key] = (a, b)
    return _PARAM_CACHE[key]


def precompute_lsh_features(
    texts: Sequence[str],
    *,
    ngram_n: int = 5,
    num_perm: int = 64,
    bands: int = 16,
    seed: int = 42,
) -> Tuple[List[Set[int]], List[List[bytes]]]:
    """各テキストの n-gram 64-bit ハッシュ集合と 16 バンドキー列を 1 パスで事前計算する。"""
    assert num_perm % bands == 0
    rows = num_perm // bands
    a, b = _get_np_perm_params(num_perm=num_perm, seed=seed)
    a_mat = a[None, :]
    b_mat = b[None, :]
    ngram_hash_cache: Dict[str, int] = {}

    gram_sets: List[Set[int]] = []
    band_keys_list: List[List[bytes]] = []

    zero_keys = [bytes(rows * 8) for _ in range(bands)]
    for t in texts:
        grams = sr.char_ngrams(sr.normalize_for_dedup(t) or t, n=ngram_n)
        if not grams:
            gram_sets.append(set())
            band_keys_list.append(zero_keys)
            continue

        hlist: List[int] = []
        for g in grams:
            hv = ngram_hash_cache.get(g)
            if hv is None:
                hv = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little")
                ngram_hash_cache[g] = hv
            hlist.append(hv)

        gram_sets.append(set(hlist))
        h_arr = np.array(hlist, dtype=np.uint64)[:, None]
        with np.errstate(over="ignore"):
            x = h_arr * a_mat + b_mat
            x ^= x >> _SHIFT33
            x *= _CONST_MIX
            x ^= x >> _SHIFT33
        sig = x.min(axis=0)
        sig_bytes = sig.tobytes()
        step = rows * 8
        band_keys_list.append([sig_bytes[bi * step : (bi + 1) * step] for bi in range(bands)])

    return gram_sets, band_keys_list


def find_near_duplicates_fast(
    texts: Sequence[str],
    *,
    ngram_n: int = 5,
    num_perm: int = 64,
    bands: int = 16,
    seed: int = 42,
    jaccard_thresh: float = 0.75,
    max_bucket: int = 500,
) -> List[Tuple[int, int, float]]:
    """単一配列内の近傍重複ペア (i, j, jaccard) を検出する。"""
    gram_sets, band_keys_list = precompute_lsh_features(
        texts, ngram_n=ngram_n, num_perm=num_perm, bands=bands, seed=seed
    )
    buckets: List[Dict[bytes, List[int]]] = [defaultdict(list) for _ in range(bands)]
    for idx, bkeys in enumerate(band_keys_list):
        if not gram_sets[idx]:
            continue
        for bi, bk in enumerate(bkeys):
            buckets[bi][bk].append(idx)

    cand_pairs: Set[Tuple[int, int]] = set()
    for bmap in buckets:
        for idxs in bmap.values():
            if len(idxs) < 2:
                continue
            if len(idxs) > max_bucket:
                idxs = idxs[:max_bucket]
            for i in range(len(idxs)):
                u = idxs[i]
                for j in range(i + 1, len(idxs)):
                    v = idxs[j]
                    cand_pairs.add((u, v) if u < v else (v, u))

    hits: List[Tuple[int, int, float]] = []
    for u, v in sorted(cand_pairs):
        su = gram_sets[u]
        sv = gram_sets[v]
        inter = len(su & sv)
        jac = inter / (len(su) + len(sv) - inter)
        if jac >= jaccard_thresh:
            hits.append((u, v, jac))
    return hits


class LSHIndex:
    """増分追加可能な LSH バケットインデックス（プール間重複フィルタ用）。"""

    def __init__(self, *, ngram_n: int = 5, num_perm: int = 64, bands: int = 16, seed: int = 42):
        self.ngram_n = ngram_n
        self.num_perm = num_perm
        self.bands = bands
        self.seed = seed
        self.gram_sets: List[Set[int]] = []
        self.buckets: List[Dict[bytes, List[int]]] = [defaultdict(list) for _ in range(bands)]

    def add_texts(self, texts: Sequence[str]) -> None:
        gsets, bkeys_list = precompute_lsh_features(
            texts, ngram_n=self.ngram_n, num_perm=self.num_perm, bands=self.bands, seed=self.seed
        )
        base_idx = len(self.gram_sets)
        self.gram_sets.extend(gsets)
        for off, bkeys in enumerate(bkeys_list):
            if not gsets[off]:
                continue
            idx = base_idx + off
            for bi, bk in enumerate(bkeys):
                self.buckets[bi][bk].append(idx)

    def query_matches(self, texts: Sequence[str], jaccard_thresh: float = 0.70) -> Tuple[List[bool], int]:
        """各テキストがインデックス内いずれかの文と jaccard >= thresh かを返す。"""
        if not self.gram_sets or not texts:
            return [False] * len(texts), 0
        gsets, bkeys_list = precompute_lsh_features(
            texts, ngram_n=self.ngram_n, num_perm=self.num_perm, bands=self.bands, seed=self.seed
        )
        matched = [False] * len(texts)
        pair_count = 0
        for q_idx, (sg, bkeys) in enumerate(zip(gsets, bkeys_list)):
            if not sg:
                continue
            cand_ids: Set[int] = set()
            for bi, bk in enumerate(bkeys):
                cand_ids.update(self.buckets[bi].get(bk, ()))
            for cid in cand_ids:
                sc_set = self.gram_sets[cid]
                if not sc_set:
                    continue
                inter = len(sg & sc_set)
                jac = inter / (len(sg) + len(sc_set) - inter)
                if jac >= jaccard_thresh:
                    matched[q_idx] = True
                    pair_count += 1
                    break
        return matched, pair_count

    def count_cross_overlap_pairs(self, texts: Sequence[str], jaccard_thresh: float = 0.70) -> int:
        """リーク検査用: インデックス内全件と texts の全重複ペア総数を厳密に数える。"""
        if not self.gram_sets or not texts:
            return 0
        gsets, bkeys_list = precompute_lsh_features(
            texts, ngram_n=self.ngram_n, num_perm=self.num_perm, bands=self.bands, seed=self.seed
        )
        seen_pairs: Set[Tuple[int, int]] = set()
        for q_idx, (sg, bkeys) in enumerate(zip(gsets, bkeys_list)):
            if not sg:
                continue
            cand_ids: Set[int] = set()
            for bi, bk in enumerate(bkeys):
                cand_ids.update(self.buckets[bi].get(bk, ()))
            for cid in cand_ids:
                pair = (cid, q_idx)
                if pair in seen_pairs:
                    continue
                sc_set = self.gram_sets[cid]
                if not sc_set:
                    continue
                inter = len(sg & sc_set)
                jac = inter / (len(sg) + len(sc_set) - inter)
                if jac >= jaccard_thresh:
                    seen_pairs.add(pair)
        return len(seen_pairs)
