#!/usr/bin/env python3
"""第6号 ja-error-detect-v1 用 文字オフセット整列・復号ユーティリティ。

Fast トークナイザ（ModernBERT）および Slow トークナイザ（BERT-Japanese MeCab+WordPiece）の
双方に対して元テキスト上の文字オフセット (start, end) を正確に付与し、
文字単位の誤りスパンとトークン列上の BIO ラベル（O / B-ERR / I-ERR）を相互変換する。
"""
import re
import unicodedata
from typing import Dict, List, Tuple

LABELS = ["O", "B-ERR", "I-ERR"]
LABEL2ID: Dict[str, int] = {"O": 0, "B-ERR": 1, "I-ERR": 2}
ID2LABEL: Dict[int, str] = {0: "O", 1: "B-ERR", 2: "I-ERR"}


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
    """書記素クラスタ単位で NFKC 正規化を行い、元テキストへの逆対応表を返す。

    1文字が複数文字に展開される場合（例: '㈱' -> '(株)'）も、
    すべての展開後文字に対して元のクラスタ範囲 [pos, pos + len(cl)) を割り当てる。
    """
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
                # 正規化差異等で完全一致できない場合のフォールバック
                subwords = tok.subword_tokenizer.tokenize(word)
                for sw in subwords:
                    tokens.append(sw)
                    offsets.append((0, 0))
                continue

            word_start_orig = start_map[idx]
            word_end_orig = end_map[idx + len(word) - 1]
            subwords = tok.subword_tokenizer.tokenize(word)

            if subwords == [unk]:
                # 希少漢字等で [UNK] になった場合も -100 に落とさず元テキスト上の実オフセットを保持する
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
    # 先頭・末尾特殊トークン（[CLS], [SEP]）に合わせて (0, 0) を付与
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


def align_spans_to_tokens(
    text: str,
    offsets: List[Tuple[int, int]],
    spans: List[dict],
) -> List[int]:
    """文字単位の誤りスパンをトークン列の BIO ラベル（O/B-ERR/I-ERR, 特殊は -100）へ整列する。

    サブワード境界が 1 文字誤りを内包するケースでラベルが脱落しないよう、
    厳密包含ではなく重なり判定 (a < end and b > start) で整列する。
    """
    labels = [0] * len(offsets)
    for i, (a, b) in enumerate(offsets):
        if a >= b:
            labels[i] = -100

    for s in spans:
        st, ed = int(s["start"]), int(s["end"])
        if s.get("corrupted") is not None:
            assert text[st:ed] == s["corrupted"], (
                f"スパン文字列不一致: expected={s['corrupted']!r}, actual={text[st:ed]!r}"
            )
        first = True
        for i, (a, b) in enumerate(offsets):
            if a >= b:
                continue
            if a < ed and b > st:
                labels[i] = LABEL2ID["B-ERR"] if first else LABEL2ID["I-ERR"]
                first = False
    return labels


def decode_token_preds_to_spans(
    text: str,
    offsets: List[Tuple[int, int]],
    pred_ids: List[int],
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """トークン単位の予測ラベル列から文字単位スパン一覧と文字単位 BIO 配列を復元する。"""
    raw_spans: List[List[int]] = []
    cur: List[int] | None = None

    for (a, b), tid in zip(offsets, pred_ids):
        if a >= b:
            continue
        if tid == 1:  # B-ERR
            if cur is not None and a == cur[0] and b == cur[1]:
                # byte-fallback 等で同一文字が複数トークンに分割された場合の連続 B-ERR を統合
                cur[1] = max(cur[1], b)
            else:
                if cur is not None:
                    raw_spans.append(cur)
                cur = [a, b]
        elif tid == 2:  # I-ERR
            if cur is not None:
                cur[1] = max(cur[1], b)
            else:
                # 孤立した I-ERR は新規スパン開始として回復
                cur = [a, b]
        else:  # O (または -100)
            if cur is not None:
                raw_spans.append(cur)
                cur = None

    if cur is not None:
        raw_spans.append(cur)

    # byte-fallback 途中での断片化等により同一文字範囲が重複した場合のみ統合
    merged: List[List[int]] = []
    for s, e in raw_spans:
        if merged and s < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # スパン端の純空白文字のみトリム
    clean_spans: List[Tuple[int, int]] = []
    n_text = len(text)
    for s, e in merged:
        s = max(0, min(s, n_text))
        e = max(0, min(e, n_text))
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            clean_spans.append((s, e))

    char_preds = [0] * n_text
    for s, e in clean_spans:
        char_preds[s] = 1  # B-ERR
        for k in range(s + 1, e):
            char_preds[k] = 2  # I-ERR

    return clean_spans, char_preds
