#!/usr/bin/env python3
"""第7号 ja-address-ner-v1 用 文字オフセット整列・復号・住所グルーピングユーティリティ。

Fast トークナイザ（ModernBERT）および Slow トークナイザ（BERT-Japanese MeCab+WordPiece）の
双方に対して元テキスト上の文字オフセット (start, end) を正確に付与し、
5構成要素（PREF, CITY, TOWN, BLOCK, BUILDING）の11クラス BIO ラベル相互変換と
構成要素スパンから住所単位スパンへのグルーピングを提供する。
"""
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

COMPONENTS: List[str] = ["PREF", "CITY", "TOWN", "BLOCK", "BUILDING"]
LABELS: List[str] = ["O"]
for _comp in COMPONENTS:
    LABELS.append(f"B-{_comp}")
    LABELS.append(f"I-{_comp}")

LABEL2ID: Dict[str, int] = {lb: i for i, lb in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {i: lb for i, lb in enumerate(LABELS)}

_PURE_WS_RE = re.compile(r"[ \t\u3000]*")


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


def align_spans_to_tokens(
    text: str,
    offsets: List[Tuple[int, int]],
    comp_spans: List[dict],
) -> List[int]:
    """文字単位の5構成要素スパンをトークン列の 11クラス BIO ラベルへ整列する。

    トークンの前後空白を除いた実質範囲 (a_t, b_t) に対し、重なり幅最大の
    正解要素スパンを割り当てて直結要素の境界ズレを最小化する。
    """
    n_text = len(text)
    for s in comp_spans:
        st, ed = int(s["start"]), int(s["end"])
        if s.get("text") is not None:
            assert text[st:ed] == s["text"], (
                f"スパン文字列不一致: expected={s['text']!r}, actual={text[st:ed]!r}"
            )

    labels = [0] * len(offsets)
    prev_span_idx: Optional[int] = None

    for i, (a, b) in enumerate(offsets):
        if a >= b:
            labels[i] = -100
            prev_span_idx = None
            continue

        a_t = max(0, min(a, n_text))
        b_t = max(0, min(b, n_text))
        while a_t < b_t and text[a_t].isspace():
            a_t += 1
        while b_t > a_t and text[b_t - 1].isspace():
            b_t -= 1
        if a_t >= b_t:
            # 純空白トークン：単一正解スパンの厳密内部にある場合は継続（建物名等の空白分裂防止）
            inside_idx: Optional[int] = None
            for si, s in enumerate(comp_spans):
                if int(s["start"]) < a and b <= int(s["end"]):
                    inside_idx = si
                    break
            if inside_idx is not None and prev_span_idx == inside_idx:
                labels[i] = LABEL2ID[f"I-{comp_spans[inside_idx]['label']}"]
            else:
                labels[i] = 0
                prev_span_idx = None
            continue

        best_idx: Optional[int] = None
        best_ov = 0
        for si, s in enumerate(comp_spans):
            st, ed = int(s["start"]), int(s["end"])
            ov = min(b_t, ed) - max(a_t, st)
            if ov > best_ov:
                best_ov = ov
                best_idx = si

        if best_idx is None:
            labels[i] = 0
            prev_span_idx = None
        else:
            lb_name = comp_spans[best_idx]["label"]
            prefix = "I-" if prev_span_idx == best_idx else "B-"
            labels[i] = LABEL2ID[f"{prefix}{lb_name}"]
            prev_span_idx = best_idx

    return labels


def decode_token_preds_to_components(
    text: str,
    offsets: List[Tuple[int, int]],
    pred_ids: List[int],
) -> List[dict]:
    """トークン単位の11クラス予測ラベル列から文字単位の構成要素スパン一覧を復元する。"""
    n_text = len(text)
    raw: List[dict] = []
    cur: Optional[dict] = None

    for (a, b), tid in zip(offsets, pred_ids):
        if a >= b:
            continue
        lb_str = ID2LABEL.get(int(tid), "O")
        if lb_str == "O":
            if cur is not None:
                raw.append(cur)
                cur = None
            continue

        bio, comp = lb_str.split("-", 1)
        if bio == "B":
            if (
                cur is not None
                and cur["label"] == comp
                and a == cur["last_a"]
                and b == cur["last_b"]
            ):
                cur["end"] = max(cur["end"], b)
            else:
                if cur is not None:
                    raw.append(cur)
                cur = {"start": a, "end": b, "label": comp, "last_a": a, "last_b": b}
        else:  # I-X
            if cur is not None and cur["label"] == comp:
                cur["end"] = max(cur["end"], b)
                cur["last_a"] = a
                cur["last_b"] = b
            else:
                if cur is not None:
                    raw.append(cur)
                # 孤立または異種別からの切替 I-X は新スパンとして回復
                cur = {"start": a, "end": b, "label": comp, "last_a": a, "last_b": b}

    if cur is not None:
        raw.append(cur)

    clean: List[dict] = []
    for item in raw:
        s = max(0, min(int(item["start"]), n_text))
        e = max(0, min(int(item["end"]), n_text))
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            clean.append(
                {
                    "start": s,
                    "end": e,
                    "label": item["label"],
                    "text": text[s:e],
                }
            )
    return clean


def group_components_to_addresses(text: str, comp_spans: List[dict]) -> List[dict]:
    """出現順の構成要素スパンを統一規則で住所単位スパンへ統合する。

    連続要素間のギャップが純空白（スペース/タブ/全角空白）のみ、かつ
    同一グループ内に同じ要素種別がまだ含まれない場合にのみ1つの住所へまとめる。
    """
    if not comp_spans:
        return []
    ordered = sorted(comp_spans, key=lambda x: (int(x["start"]), int(x["end"])))

    groups: List[List[dict]] = []
    cur_grp: List[dict] = [ordered[0]]
    seen_labels = {ordered[0]["label"]}
    cur_end = int(ordered[0]["end"])

    for sp in ordered[1:]:
        st = int(sp["start"])
        ed = int(sp["end"])
        gap = text[cur_end:st] if st >= cur_end else ""
        if (
            st >= cur_end
            and _PURE_WS_RE.fullmatch(gap)
            and sp["label"] not in seen_labels
        ):
            cur_grp.append(sp)
            seen_labels.add(sp["label"])
            cur_end = max(cur_end, ed)
        else:
            groups.append(cur_grp)
            cur_grp = [sp]
            seen_labels = {sp["label"]}
            cur_end = ed
    groups.append(cur_grp)

    addresses: List[dict] = []
    for grp in groups:
        st = min(int(c["start"]) for c in grp)
        ed = max(int(c["end"]) for c in grp)
        comp_map = {c["label"]: c["text"] for c in grp}
        addresses.append(
            {
                "start": st,
                "end": ed,
                "text": text[st:ed],
                "components": comp_map,
                "component_spans": grp,
            }
        )
    return addresses
