#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 自己完結型 構造化読み・発音推定モジュール。

PyTorch (model.safetensors) および ONNX Runtime FP32 (onnx/model.onnx) の
両バックエンドに対応し、外部プロジェクトモジュールに一切依存せず
release-staging ディレクトリ単体で動作する参照推論実装。

入力テキストを一切改変せず、漢字スパンごとの
[{"start", "end", "surface", "reading", "pron"}] を返す。
（人名・算用数字・英字は仕様により対象外）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sudachipy import dictionary, tokenizer
from transformers import AutoTokenizer

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


# ---------------------------------------------------------------------------
# 日本語表記・音韻ユーティリティ
# ---------------------------------------------------------------------------

KANJI_CHARS = "々ヶヵ〆〇"
KANJI_RE = re.compile(r"[一-龥々ヶヵ〆〇]")

_O_ROW = frozenset("オコソトノホモヨロヲゴゾドボポォョ")
_E_ROW = frozenset("エケセテネヘメエレゲゼデベペェ")
_U_ROW = frozenset("ウクスツヌフムユルグズヅブプゥュヴ")

_NUM_KANJI = frozenset("一二三四五六七八九十百千万億兆〇零")
_NUM_SENSITIVE_COMPOUNDS = frozenset({"一日", "一分", "十分", "三月", "五月"})
_ADJ_CATALOG_KEYS = frozenset({"辛+い", "良+い", "強+い"})


def is_kanji_char(ch: str) -> bool:
    """1文字が対象漢字（CJK統合漢字および々ヶヵ〆〇）か判定する。"""
    if not ch:
        return False
    return ("\u4e00" <= ch <= "\u9fff") or (ch in KANJI_CHARS)


def has_kanji(text: str) -> bool:
    """文字列に1文字以上の対象漢字が含まれるか判定する。"""
    return bool(KANJI_RE.search(text))


def is_all_kanji(text: str) -> bool:
    """文字列が空でなく全文字が対象漢字か判定する。"""
    return bool(text) and all(is_kanji_char(c) for c in text)


def hira_to_kata(text: str) -> str:
    """ひらがなをカタカナに変換する（長音・記号はそのまま保持）。"""
    out: List[str] = []
    for ch in text:
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:
            out.append(chr(code + 0x60))
        elif ch == "\u309d":  # ゝ
            out.append("\u30fd")
        elif ch == "\u309e":  # ゞ
            out.append("\u30fe")
        else:
            out.append(ch)
    return "".join(out)


def kata_to_hira(text: str) -> str:
    """カタカナをひらがなに変換する（長音・ヴ等はそのまま保持）。"""
    out: List[str] = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        elif ch == "\u30fd":  # ヽ
            out.append("\u309d")
        elif ch == "\u30fe":  # ヾ
            out.append("\u309e")
        else:
            out.append(ch)
    return "".join(out)


def normalize_kana(text: str) -> str:
    """読みを標準の全角カタカナへ正規化する。"""
    t = text.strip().replace(" ", "").replace("\u3000", "")
    return hira_to_kata(t)


def is_valid_reading_str(text: str) -> bool:
    """読み文字列が正当な仮名列のみで構成されるか判定する。"""
    if not text:
        return False
    for ch in text:
        code = ord(ch)
        if not (
            (0x3041 <= code <= 0x3096)
            or (0x30A1 <= code <= 0x30FA)
            or ch in ("ー", "・")
        ):
            return False
    return True


def strip_okurigana(
    surface: str, reading_kata: str
) -> Optional[List[Tuple[int, int, str, str]]]:
    """表層文字列と全体の読みカタカナから漢字ブロック単位の (start, end, 漢字列, 読み) を抽出する。"""
    if not surface or not reading_kata:
        return None
    r_kata = normalize_kana(reading_kata)
    if not has_kanji(surface):
        return []

    segs: List[Tuple[bool, int, int, str]] = []
    n = len(surface)
    i = 0
    while i < n:
        flag = is_kanji_char(surface[i])
        j = i + 1
        while j < n and is_kanji_char(surface[j]) == flag:
            j += 1
        segs.append((flag, i, j, surface[i:j]))
        i = j

    if len(segs) == 1 and segs[0][0]:
        return [(0, n, surface, r_kata)]

    seg_info: List[Tuple[bool, int, int, str, str]] = []
    for flag, a, b, txt in segs:
        k = "" if flag else normalize_kana(txt)
        seg_info.append((flag, a, b, txt, k))

    r_len = len(r_kata)
    solutions: List[List[Tuple[int, int, str, str]]] = []

    def dfs(s_idx: int, r_pos: int, acc: List[Tuple[int, int, str, str]]) -> None:
        if s_idx == len(seg_info):
            if r_pos == r_len:
                solutions.append(list(acc))
            return

        is_k, a, b, txt, k_pat = seg_info[s_idx]
        if not is_k:
            if not k_pat:
                dfs(s_idx + 1, r_pos, acc)
                return
            if r_kata.startswith(k_pat, r_pos):
                dfs(s_idx + 1, r_pos + len(k_pat), acc)
            return

        min_len = 1
        if s_idx == len(seg_info) - 1:
            if r_len - r_pos >= min_len:
                acc.append((a, b, txt, r_kata[r_pos:]))
                dfs(s_idx + 1, r_len, acc)
                acc.pop()
            return

        next_k_pat = seg_info[s_idx + 1][4]
        if not next_k_pat:
            return
        rem_kana_len = sum(len(x[4]) for x in seg_info[s_idx + 1 :] if not x[0])
        rem_kanji_cnt = sum(1 for x in seg_info[s_idx + 1 :] if x[0])
        max_end = r_len - rem_kana_len - rem_kanji_cnt
        for end_pos in range(r_pos + min_len, max_end + 1):
            if r_kata.startswith(next_k_pat, end_pos):
                acc.append((a, b, txt, r_kata[r_pos:end_pos]))
                dfs(s_idx + 1, end_pos, acc)
                acc.pop()

    dfs(0, 0, [])
    if not solutions:
        return None
    if len(solutions) == 1:
        return solutions[0]

    def score_sol(sol: List[Tuple[int, int, str, str]]) -> Tuple[float, int]:
        penalty = 0.0
        for a, b, _, rk in sol:
            klen = b - a
            ratio = len(rk) / max(1, klen)
            penalty += abs(ratio - 1.8)
            if len(rk) == 0:
                penalty += 100.0
        return (penalty, -len(sol[0][3]))

    solutions.sort(key=score_sol)
    return solutions[0]


def rule_kana_to_pron(reading_kata: str) -> str:
    """表記読みカタカナを UniDic 準拠の発音読みカタカナへ規則変換する。"""
    k = normalize_kana(reading_kata).replace("ヂ", "ジ").replace("ヅ", "ズ")
    out: List[str] = []
    i = 0
    n = len(k)
    while i < n:
        ch = k[i]
        if i + 1 < n:
            nxt = k[i + 1]
            if ch in _O_ROW and (nxt == "ウ" or nxt == "オ"):
                out.append(ch)
                out.append("ー")
                i += 2
                continue
            if ch in _E_ROW and nxt == "イ":
                out.append(ch)
                out.append("ー")
                i += 2
                continue
            if ch in _U_ROW and nxt == "ウ":
                out.append(ch)
                out.append("ー")
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def kana_to_pron(
    reading_kata: str,
    surface: str = "",
    pron_map: Optional[Dict[str, str]] = None,
) -> str:
    """発音マップを最優先し、未登録語には規則変換を適用して発音読みカタカナを返す。"""
    k = normalize_kana(reading_kata)
    if pron_map:
        if surface:
            sk_key = f"{surface}\t{k}"
            if sk_key in pron_map:
                return pron_map[sk_key]
        if k in pron_map:
            return pron_map[k]
    return rule_kana_to_pron(k)


# ---------------------------------------------------------------------------
# 文字オフセット整列ユーティリティ
# ---------------------------------------------------------------------------

def _clusters(text: str) -> List[str]:
    """書記素クラスタの近似分割。"""
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


def _nfkc_with_map(text: str) -> Tuple[str, List[int], List[int]]:
    """NFKC 正規化と元文字インデックス対応表を返す。"""
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


def _encode_slow(tok, text: str, max_len: int) -> Tuple[dict, List[Tuple[int, int]]]:
    """Slow トークナイザ向けフォールバック（Fast 時は return_offsets_mapping を使用）。"""
    norm, start_map, end_map = _nfkc_with_map(text)
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
                for sw in tok.subword_tokenizer.tokenize(word):
                    tokens.append(sw)
                    offsets.append((0, 0))
                continue
            ws, we = start_map[idx], end_map[idx + len(word) - 1]
            subwords = tok.subword_tokenizer.tokenize(word)
            if subwords == [unk]:
                tokens.append(unk)
                offsets.append((ws, we))
            else:
                wpos = 0
                for sw in subwords:
                    surf = sw[2:] if sw.startswith("##") else sw
                    sub_pos = word.find(surf, wpos)
                    if sub_pos < 0:
                        tokens.append(sw)
                        offsets.append((ws, we))
                    else:
                        s_ni = idx + sub_pos
                        e_ni = s_ni + len(surf) - 1
                        tokens.append(sw)
                        offsets.append((start_map[s_ni], end_map[e_ni]))
                        wpos = sub_pos + len(surf)
            pos = idx + len(word)

    max_body = max(0, max_len - tok.num_special_tokens_to_add(pair=False))
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
    enc = {"input_ids": ids, "attention_mask": [1] * len(ids)}
    if "token_type_ids" in tok.model_input_names:
        enc["token_type_ids"] = [0] * len(ids)
    return enc, offsets


def encode_with_offsets(tok, text: str, max_len: int) -> Tuple[dict, List[Tuple[int, int]]]:
    """(encoding, offset_mapping) を返す。"""
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
    return _encode_slow(tok, text, max_len)


def span_to_token_indices(
    offsets: List[Tuple[int, int]], start: int, end: int
) -> List[int]:
    """文字スパン [start, end) に対応するトークンインデックス列を返す。"""
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


# ---------------------------------------------------------------------------
# 決定的スパン抽出器（inference_config.json + pron_map.json 使用）
# ---------------------------------------------------------------------------

class StandaloneSpanExtractor:
    """SudachiPy Mode C ＋ 複合語補正 ＋ カタログ候補列挙による決定的スパン抽出器。"""

    def __init__(self, model_dir: Path, inference_cfg: dict) -> None:
        self.model_dir = Path(model_dir)
        self.exclude_person_names = bool(inference_cfg.get("exclude_person_names", True))
        self.multi_readings: Dict[str, dict] = inference_cfg.get("catalog_candidates", {})
        self.compound_catalog: Set[str] = set(inference_cfg.get("compound_catalog", []))
        self.protected_surfaces: Set[str] = set(inference_cfg.get("protected_surfaces", []))

        pron_rel = inference_cfg.get("pron_map_path", "pron_map.json")
        pron_file = self.model_dir / pron_rel if pron_rel else None
        if pron_file is not None and pron_file.exists():
            with open(pron_file, encoding="utf-8") as f:
                self.pron_map: Dict[str, str] = json.load(f)
        else:
            self.pron_map = {}

        sudachi_dict_kind = str(inference_cfg.get("sudachi_dict", "core"))
        self.sudachi_dict = dictionary.Dictionary(dict=sudachi_dict_kind)
        self.tok = self.sudachi_dict.tokenizer()

    def to_pron(self, reading_kata: str, surface: str = "") -> str:
        """表記読みカタカナを発音読みカタカナ（UniDic準拠）へ変換する。"""
        return kana_to_pron(reading_kata, surface=surface, pron_map=self.pron_map)

    def _tokenize_with_compound_fix(
        self, text: str
    ) -> List[Tuple[int, int, str, str, Tuple[str, ...]]]:
        """Sudachi Mode C でトークナイズし、カタログ複合語の境界ズレを補正する。"""
        ms = list(self.tok.tokenize(text, tokenizer.Tokenizer.SplitMode.C))
        raw_tokens: List[Tuple[int, int, str, str, Tuple[str, ...]]] = [
            (m.begin(), m.end(), m.surface(), m.reading_form(), tuple(m.part_of_speech()[:3]))
            for m in ms
        ]
        fixed: List[Tuple[int, int, str, str, Tuple[str, ...]]] = []
        i = 0
        n_tok = len(raw_tokens)
        while i < n_tok:
            st, ed, surf, read_f, pos = raw_tokens[i]
            matched_comp: Optional[str] = None
            for L in (3, 2):
                if st + L <= len(text) and len(surf) < L:
                    cand = text[st : st + L]
                    if cand in self.compound_catalog:
                        if cand in _NUM_SENSITIVE_COMPOUNDS and st > 0 and text[st - 1] in _NUM_KANJI:
                            continue
                        matched_comp = cand
                        break

            if matched_comp is not None:
                comp_end = st + len(matched_comp)
                j = i
                while j < n_tok and raw_tokens[j][1] <= comp_end:
                    j += 1
                info = self.multi_readings[matched_comp]
                fixed.append(
                    (st, comp_end, matched_comp, info["default"], ("名詞", "普通名詞", "一般"))
                )
                if j < n_tok and raw_tokens[j][0] < comp_end < raw_tokens[j][1]:
                    chunk_end = len(text)
                    for p in range(comp_end, len(text)):
                        if text[p] in "、。！？,.!?\n\r \u3000":
                            chunk_end = p + 1
                            break
                    k = j
                    while k < n_tok and raw_tokens[k][1] <= chunk_end:
                        k += 1
                    chunk_end = max(chunk_end, raw_tokens[k - 1][1]) if k > j else raw_tokens[j][1]
                    rem_text = text[comp_end:chunk_end]
                    re_tokens: List[Tuple[int, int, str, str, Tuple[str, ...]]] = [
                        (
                            comp_end + rm.begin(),
                            comp_end + rm.end(),
                            rm.surface(),
                            rm.reading_form(),
                            tuple(rm.part_of_speech()[:3]),
                        )
                        for rm in self.tok.tokenize(rem_text, tokenizer.Tokenizer.SplitMode.C)
                    ]
                    raw_tokens = raw_tokens[:j] + re_tokens + raw_tokens[k:]
                    n_tok = len(raw_tokens)
                i = j
                continue

            fixed.append((st, ed, surf, read_f, pos))
            i += 1
        return fixed

    def _resolve_key_and_candidates(
        self,
        text: str,
        abs_s: int,
        abs_e: int,
        k_surf: str,
        k_read: str,
        morph_is_all_kanji: bool,
        pos: Tuple[str, ...],
    ) -> Tuple[str, List[str]]:
        """漢字ブロックに対する識別キーと候補リストを決定する。"""
        if pos[0] in ("接頭辞", "接尾辞") or pos[1] == "数詞":
            return k_surf, ([k_read] if k_read else [])

        for tlen in (2, 1):
            if abs_e + tlen <= len(text):
                trail = text[abs_e : abs_e + tlen]
                if all("\u3041" <= c <= "\u3096" for c in trail):
                    ck = f"{k_surf}+{trail}"
                    if ck in self.multi_readings:
                        if ck in _ADJ_CATALOG_KEYS and pos[0] != "形容詞":
                            continue
                        if ck not in _ADJ_CATALOG_KEYS and pos[0] != "動詞":
                            continue
                        cands = list(self.multi_readings[ck]["candidates"])
                        if k_read and k_read not in cands:
                            cands = [k_read] + cands
                        return ck, cands

        is_predicate = pos[0] in ("動詞", "形容詞") or (not morph_is_all_kanji)
        if not is_predicate and k_surf in self.multi_readings:
            cands = list(self.multi_readings[k_surf]["candidates"])
            if k_read and k_read not in cands:
                cands = [k_read] + cands
            return k_surf, cands

        return k_surf, ([k_read] if k_read else [])

    def extract_spans(self, text: str) -> List[dict]:
        """テキストから対象漢字スパン一覧と候補リストを抽出する。"""
        if not text:
            return []
        tokens = self._tokenize_with_compound_fix(text)
        spans: List[dict] = []

        for st, ed, surf, read_f, pos in tokens:
            if not has_kanji(surf):
                continue
            if self.exclude_person_names and pos[1] == "固有名詞" and pos[2] == "人名":
                if surf not in self.protected_surfaces:
                    continue

            rk = normalize_kana(read_f) if (read_f and read_f != "*") else ""
            if not rk or not is_valid_reading_str(rk):
                if surf in self.multi_readings:
                    rk = self.multi_readings[surf]["default"]
                else:
                    continue

            morph_all_kanji = is_all_kanji(surf)
            stripped = strip_okurigana(surf, rk)
            if stripped is None:
                stripped = [(0, len(surf), surf, rk)]

            for rel_s, rel_e, k_surf, k_read in stripped:
                abs_s = st + rel_s
                abs_e = st + rel_e
                key, cands = self._resolve_key_and_candidates(
                    text, abs_s, abs_e, k_surf, k_read, morph_all_kanji, pos
                )
                if not cands:
                    continue
                default_r = k_read if k_read in cands else cands[0]
                spans.append(
                    {
                        "start": abs_s,
                        "end": abs_e,
                        "surface": k_surf,
                        "key": key,
                        "candidates": cands,
                        "default_reading": default_r,
                    }
                )
        return spans


# ---------------------------------------------------------------------------
# メイン推論クラス（PyTorch / ONNX Runtime 両対応）
# ---------------------------------------------------------------------------

class JaReadingTagger:
    """日本語テキストから漢字スパンの表記読み・発音読みを文脈推定する構造化タガー。"""

    def __init__(
        self,
        model_dir: str | Path,
        backend: str = "pytorch",
        device: Optional[str] = None,
        max_len: Optional[int] = None,
        override_tau: Optional[float] = None,
        local_files_only: bool = False,
        ort_intra_threads: int = 0,
        ort_inter_threads: int = 0,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.backend = backend.lower().strip()
        if self.backend not in ("pytorch", "onnx"):
            raise ValueError(f"未対応の backend です: {backend} ('pytorch' または 'onnx' を指定)")

        inf_path = self.model_dir / "inference_config.json"
        if not inf_path.exists():
            raise FileNotFoundError(f"inference_config.json が見つかりません: {inf_path}")
        with open(inf_path, encoding="utf-8") as f:
            self.inf_cfg = json.load(f)

        self.catalog_version = str(self.inf_cfg.get("catalog_version", "unknown"))
        self.max_len = int(max_len if max_len is not None else self.inf_cfg.get("max_length", 256))
        self.override_tau = float(
            override_tau if override_tau is not None else self.inf_cfg.get("override_tau", 0.0)
        )
        self.label2id: Dict[str, int] = dict(self.inf_cfg["label2id"])
        self.extractor = StandaloneSpanExtractor(self.model_dir, self.inf_cfg)
        self.tok = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=local_files_only
        )

        self.pt_model = None
        self.ort_sess = None
        self.ort_input_names: List[str] = []
        self.device = "cpu"

        if self.backend == "pytorch":
            if torch is None:
                raise ImportError("backend='pytorch' には torch が必要です。")
            from transformers import AutoModelForTokenClassification

            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.pt_model = AutoModelForTokenClassification.from_pretrained(
                str(self.model_dir), local_files_only=local_files_only
            )
            self.pt_model.to(self.device)
            self.pt_model.eval()
        else:
            if ort is None:
                raise ImportError("backend='onnx' には onnxruntime が必要です。")
            onnx_rel = self.inf_cfg.get("onnx_path", "onnx/model.onnx")
            onnx_file = self.model_dir / onnx_rel
            if not onnx_file.exists():
                raise FileNotFoundError(f"ONNX ファイルが見つかりません: {onnx_file}")
            sess_opts = ort.SessionOptions()
            if ort_intra_threads > 0:
                sess_opts.intra_op_num_threads = int(ort_intra_threads)
            if ort_inter_threads > 0:
                sess_opts.inter_op_num_threads = int(ort_inter_threads)
            self.ort_sess = ort.InferenceSession(
                str(onnx_file),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self.ort_input_names = [inp.name for inp in self.ort_sess.get_inputs()]

    @staticmethod
    def _softmax_np(x: np.ndarray) -> np.ndarray:
        """数値安定な float32 softmax を計算する。"""
        shifted = x - np.max(x)
        ex = np.exp(shifted)
        return ex / np.sum(ex)

    def _forward_logits(
        self,
        input_ids: List[List[int]],
        attention_mask: List[List[int]],
        token_type_ids: Optional[List[List[int]]] = None,
    ) -> np.ndarray:
        """パディング済みミニバッチに対して [B, T, C] の FP32 ロジット (numpy) を返す。"""
        if self.backend == "pytorch":
            assert torch is not None and self.pt_model is not None
            model_in = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self.device),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=self.device),
            }
            if token_type_ids is not None and "token_type_ids" in self.tok.model_input_names:
                model_in["token_type_ids"] = torch.tensor(
                    token_type_ids, dtype=torch.long, device=self.device
                )
            with torch.no_grad():
                logits = self.pt_model(**model_in).logits
            return logits.detach().cpu().float().numpy()

        assert self.ort_sess is not None
        feed = {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(attention_mask, dtype=np.int64),
        }
        if "token_type_ids" in self.ort_input_names:
            if token_type_ids is None:
                feed["token_type_ids"] = np.zeros_like(feed["input_ids"], dtype=np.int64)
            else:
                feed["token_type_ids"] = np.asarray(token_type_ids, dtype=np.int64)
        logits = self.ort_sess.run(None, feed)[0]
        return np.asarray(logits, dtype=np.float32)

    def predict_batch_with_logits(
        self, texts: List[str], batch_size: int = 32
    ) -> Tuple[List[List[dict]], List[Tuple[np.ndarray, List[int]]]]:
        """デコード結果と各文の有効トークンロジット（パリティ検証用）を返す。"""
        if not texts:
            return [], []

        all_results: List[List[dict]] = []
        all_logits_meta: List[Tuple[np.ndarray, List[int]]] = []
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0

        for b_start in range(0, len(texts), batch_size):
            batch_texts = texts[b_start : b_start + batch_size]
            encs: List[dict] = []
            offsets_list: List[List[Tuple[int, int]]] = []
            spans_list: List[List[dict]] = []

            for txt in batch_texts:
                enc, offsets = encode_with_offsets(self.tok, txt, max_len=self.max_len)
                encs.append(enc)
                offsets_list.append(offsets)
                spans_list.append(self.extractor.extract_spans(txt))

            maxlen = max(len(e["input_ids"]) for e in encs)
            input_ids: List[List[int]] = []
            attn: List[List[int]] = []
            has_tti = "token_type_ids" in encs[0]
            ttids: Optional[List[List[int]]] = [] if has_tti else None

            for e in encs:
                pad_n = maxlen - len(e["input_ids"])
                input_ids.append(e["input_ids"] + [pad_id] * pad_n)
                attn.append(e["attention_mask"] + [0] * pad_n)
                if ttids is not None:
                    ttids.append(e["token_type_ids"] + [0] * pad_n)

            logits = self._forward_logits(input_ids, attn, ttids)  # [B, T, C]

            for bi, (offsets, sp_list, e) in enumerate(zip(offsets_list, spans_list, encs)):
                seq_len = len(e["input_ids"])
                sent_logits = logits[bi, :seq_len, :]
                all_logits_meta.append((sent_logits, e["attention_mask"]))

                out_spans: List[dict] = []
                for sp in sp_list:
                    cands = sp["candidates"]
                    default_rk = sp["default_reading"]
                    chosen_rk = default_rk

                    if len(cands) >= 2:
                        cand_label_ids = []
                        for rk in cands:
                            lid = self.label2id.get(f"{sp['key']}|{rk}")
                            if lid is not None:
                                cand_label_ids.append((rk, lid))

                        if len(cand_label_ids) >= 2:
                            tok_idxs = span_to_token_indices(offsets, sp["start"], sp["end"])
                            if tok_idxs:
                                span_vec = sent_logits[tok_idxs].mean(axis=0)
                                cand_vals = np.array(
                                    [float(span_vec[lid]) for _, lid in cand_label_ids],
                                    dtype=np.float32,
                                )
                                probs = self._softmax_np(cand_vals)
                                best_idx = int(np.argmax(probs))
                                best_rk = cand_label_ids[best_idx][0]
                                best_prob = float(probs[best_idx])
                                if best_rk != default_rk and best_prob >= self.override_tau:
                                    chosen_rk = best_rk

                    out_spans.append(
                        {
                            "start": sp["start"],
                            "end": sp["end"],
                            "surface": sp["surface"],
                            "reading": kata_to_hira(chosen_rk),
                            "pron": self.extractor.to_pron(chosen_rk, surface=sp["surface"]),
                        }
                    )
                all_results.append(out_spans)

        return all_results, all_logits_meta

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[List[dict]]:
        """複数文に対してバッチ推論し、各文のスパン予測リストを返す。"""
        results, _ = self.predict_batch_with_logits(texts, batch_size=batch_size)
        return results

    def predict(self, text: str) -> List[dict]:
        """1文から構造化読みリストを返す。"""
        return self.predict_batch([text], batch_size=1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="第9号 ja-reading-tagger-v1 構造化読み推定 CLI")
    parser.add_argument("--model-dir", default=".", help="config.json / inference_config.json 配置ディレクトリ")
    parser.add_argument("--backend", choices=["pytorch", "onnx"], default="onnx")
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    tagger = JaReadingTagger(
        Path(args.model_dir),
        backend=args.backend,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    res = tagger.predict(args.text)
    print(json.dumps({"text": args.text, "spans": res}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
