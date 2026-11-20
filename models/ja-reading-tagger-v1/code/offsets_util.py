#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 用 文字オフセット整列ユーティリティ。

Fast トークナイザ（ModernBERT）および Slow トークナイザ（BERT-Japanese MeCab+WordPiece）の
双方に対して元テキスト上の文字オフセット (start, end) を正確に付与し、
文字スパンとトークン列インデックスの対応付けを提供する。
"""
import re
import unicodedata
from typing import List, Tuple


def _clusters(text: str) -> List[str]:
    """書記素クラスタの近似分割（基底文字＋結合文字・半角濁点）。"""
    clusters: List[str] = []
    cur = ""
    for ch in text:
        if cur and (
            unicodedata.combining(ch)
            or unicodedata.category(ch) in ("Mn", "Me", "Mc")
            or ch in ("\uff9e", "\uff9f")
        ):
            cur += ch
        else:
            if cur:
                clusters.append(cur)
            cur = ch
    if cur:
        clusters.append(cur)
    return clusters


def nfkc_with_map(text: str) -> Tuple[str, List[int], List[int]]:
    """書記素クラスタ単位で NFKC 正規化を行い、元テキストへの逆対応表を返す。"""
    norm_parts: List[str] = []
    start_map: List[int] = []
    end_map: List[int] = []
    pos = 0
    for cl in _clusters(text):
        n = unicodedata.normalize("NFKC", cl)
        if not n:
            pos += len(cl)
            continue
        for _ in n:
            start_map.append(pos)
            end_map.append(pos + len(cl))
        norm_parts.append(n)
        pos += len(cl)
    return "".join(norm_parts), start_map, end_map


def _encode_slow_bert_japanese(tok, text: str, max_len: int) -> Tuple[dict, List[Tuple[int, int]]]:
    """BertJapaneseTokenizer (is_fast=False) 向けのオフセット付きエンコード。"""
    norm, start_map, end_map = nfkc_with_map(text)
    specials = sorted(tok.all_special_tokens, key=len, reverse=True)
    sp_re = (
        re.compile("(" + "|".join(re.escape(s) for s in specials) + ")")
        if specials
        else None
    )
    segments = sp_re.split(norm) if sp_re else [norm]
    sp_set = set(specials)
    unk = tok.unk_token

    tokens: List[str] = []
    offsets: List[Tuple[int, int]] = []
    pos = 0

    for seg in segments:
        if not seg:
            continue
        if seg in sp_set:
            idx = norm.find(seg, pos)
            if idx >= 0:
                pos = idx + len(seg)
            tokens.append(seg)
            offsets.append((0, 0))
            continue

        words = tok.word_tokenizer.tokenize(seg, never_split=tok.all_special_tokens)
        for word in words:
            idx = norm.find(word, pos)
            if idx < 0:
                subwords = tok.subword_tokenizer.tokenize(word)
                for sw in subwords:
                    tokens.append(sw)
                    offsets.append((0, 0))
                continue

            word_start_orig = start_map[idx]
            word_end_orig = end_map[idx + len(word) - 1]
            subwords = tok.subword_tokenizer.tokenize(word)

            if subwords == [unk]:
                tokens.append(unk)
                offsets.append((word_start_orig, word_end_orig))
            else:
                wpos = 0
                for sw in subwords:
                    surf = sw[2:] if sw.startswith("##") else sw
                    sub_pos = word.find(surf, wpos)
                    if sub_pos < 0:
                        tokens.append(sw)
                        offsets.append((word_start_orig, word_end_orig))
                    else:
                        s_ni = idx + sub_pos
                        e_ni = s_ni + len(surf) - 1
                        tokens.append(sw)
                        offsets.append((start_map[s_ni], end_map[e_ni]))
                        wpos = sub_pos + len(surf)

            pos = idx + len(word)

    max_body = max_len - tok.num_special_tokens_to_add(pair=False)
    if max_body < 0:
        max_body = 0
    tokens = tokens[:max_body]
    offsets = offsets[:max_body]

    ids = tok.build_inputs_with_special_tokens(tok.convert_tokens_to_ids(tokens))
    n_special = len(ids) - len(tokens)
    if n_special == 2:
        offsets = [(0, 0)] + offsets + [(0, 0)]
    elif n_special == 1:
        offsets = offsets + [(0, 0)]
    elif n_special > 2:
        offsets = [(0, 0)] + offsets + [(0, 0)] * (n_special - 1)

    assert len(ids) == len(offsets), f"オフセット長不一致: {len(ids)} != {len(offsets)}"

    enc = {"input_ids": ids, "attention_mask": [1] * len(ids)}
    if "token_type_ids" in tok.model_input_names:
        enc["token_type_ids"] = [0] * len(ids)
    return enc, offsets


def encode_with_offsets(tok, text: str, max_len: int) -> Tuple[dict, List[Tuple[int, int]]]:
    """Fast / Slow 両トークナイザで (encoding, offset_mapping) を返す。"""
    if tok.is_fast:
        enc = tok(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_len,
        )
        raw_offsets = enc.pop("offset_mapping")
        offsets = [(int(a), int(b)) for a, b in raw_offsets]
        return dict(enc), offsets
    return _encode_slow_bert_japanese(tok, text, max_len)


def span_to_token_indices(
    offsets: List[Tuple[int, int]], start: int, end: int
) -> List[int]:
    """文字スパン [start, end) に対応するトークンインデックス列を返す。

    まずスパンに厳密内包される (a >= start and b <= end) トークンを優先し、
    無ければ 1 文字以上重なりを持つトークンを返す。
    """
    inside: List[int] = []
    overlap: List[int] = []
    for i, (a, b) in enumerate(offsets):
        if a >= b:
            continue
        if a >= start and b <= end:
            inside.append(i)
        if min(b, end) > max(a, start):
            overlap.append(i)
    return inside if inside else overlap
