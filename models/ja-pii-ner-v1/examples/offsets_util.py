#!/usr/bin/env python3
"""トークナイザのオフセット取得の共通部品。

bert-base-japanese-v3 のようなスロートークナイザ (MeCab 系) は offset_mapping を
返さないため、NFKC 正規化したテキスト上でトークン表層を順に照合して補完する。
オフセットは書記素クラスタ単位の位置マップで元の文字列の座標へ逆変換して返すため、
全角記号・全角英字・半角カナ結合など NFKC で文字数が変わる箇所でもずれない。
"""
import unicodedata


def _clusters(text):
    """書記素クラスタの近似分割 (基底文字＋結合文字)。"""
    clusters, cur = [], ""
    for ch in text:
        if cur and (unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Mc") or ch in "\uff9e\uff9f"):
            cur += ch
        else:
            if cur:
                clusters.append(cur)
            cur = ch
    if cur:
        clusters.append(cur)
    return clusters


def nfkc_with_map(text):
    """NFKC 正規化と、正規化後の各文字位置から元の位置への対応表を返す。

    戻り値: (norm, start_map, end_map)
    """
    norm_parts, start_map, end_map = [], [], []
    pos = 0
    for cl in _clusters(text):
        n = unicodedata.normalize("NFKC", cl)
        for _ in n:
            start_map.append(pos)
            end_map.append(pos + len(cl))
        norm_parts.append(n)
        pos += len(cl)
    return "".join(norm_parts), start_map, end_map


def get_enc(tok, text, max_len=None):
    """(enc, offsets, base_text) を返す。offsets は元の text に対する座標。"""
    if tok.is_fast:
        enc = tok(text, truncation=max_len is not None, max_length=max_len,
                  return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        return enc, offsets, text
    norm, start_map, end_map = nfkc_with_map(text)
    enc = tok(norm, truncation=max_len is not None, max_length=max_len)
    tokens = tok.convert_ids_to_tokens(enc["input_ids"])
    offsets, pos = [], 0
    for t in tokens:
        if t in tok.all_special_tokens:
            offsets.append((0, 0))
            continue
        surface = t[2:] if t.startswith("##") else t
        idx = norm.find(surface, pos)
        if idx < 0:
            offsets.append((0, 0))
            continue
        offsets.append((start_map[idx], end_map[idx + len(surface) - 1]))
        pos = idx + len(surface)
    return enc, offsets, text
