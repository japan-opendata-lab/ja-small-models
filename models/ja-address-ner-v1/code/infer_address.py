#!/usr/bin/env python3
"""第7号 ja-address-ner-v1 公開用 推論参照実装（PyTorch / ONNX 両対応・自己完結型）。

第1段階（自由記述からの住所スパン抽出・5構成要素分解）と、
任意の第2段階辞書リンカー（linker/linker.py）による全国地方公共団体コード（5桁）付与
または「判定不能（UNRESOLVABLE）」返却の一連の流れを実行する。

使い方:
  # 抽出・分解のみ
  python examples/infer_address.py --model-dir . --backend onnx \\
      --text "お届け先は東京都渋谷区神南1-2-3 パークハイツ201号室です。"

  # 抽出・分解 ＋ 辞書リンカーによるコード付与（辞書を事前構築して --master-dir に指定）
  python examples/infer_address.py --model-dir . --backend onnx --master-dir ./dict_master \\
      --text "旧住所は浜松市北区根洗町1-2、新住所は府中市3-4-5です。"
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

COMPONENTS: List[str] = ["PREF", "CITY", "TOWN", "BLOCK", "BUILDING"]
LABELS: List[str] = ["O"]
for _comp in COMPONENTS:
    LABELS.append(f"B-{_comp}")
    LABELS.append(f"I-{_comp}")
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


def _nfkc_with_map(text: str) -> Tuple[str, List[int], List[int]]:
    """書記素クラスタ単位の NFKC 正規化と元テキスト座標への逆対応表を返す。"""
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


def encode_with_offsets(tok, text: str, max_len: int) -> Tuple[dict, List[Tuple[int, int]]]:
    """Fast / Slow（BertJapaneseTokenizer）両対応でオフセット付きエンコードを返す。"""
    if getattr(tok, "is_fast", False):
        enc = tok(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_len,
        )
        raw_offsets = enc.pop("offset_mapping")
        offsets = [(int(a), int(b)) for a, b in raw_offsets]
        return dict(enc), offsets

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
        else:
            if cur is not None and cur["label"] == comp:
                cur["end"] = max(cur["end"], b)
                cur["last_a"] = a
                cur["last_b"] = b
            else:
                if cur is not None:
                    raw.append(cur)
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
    """構成要素スパン列から統一規則に従って住所単位スパンへ統合する。"""
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


def _load_address_linker_class(model_dir: Path):
    """同梱 linker/linker.py（または開発用配置）から AddressLinker を動的に読み込む。"""
    here = Path(__file__).resolve().parent
    cand_dirs = [
        model_dir / "linker",
        here.parent / "linker",
        here / "release_linker",
        here,
    ]
    for d in cand_dirs:
        lpath = d / "linker.py"
        if lpath.exists():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            spec = importlib.util.spec_from_file_location("ja_address_linker", lpath)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod.AddressLinker
    raise FileNotFoundError(
        "linker/linker.py が見つかりません。model_dir/linker/ に配置されているか確認してください。"
    )


class JaAddressExtractor:
    """2段階ハイブリッド日本語住所抽出・分解・コード付与パイプラインの参照実装。"""

    def __init__(
        self,
        model_dir: Union[str, Path],
        backend: str = "onnx",
        master_dir: Optional[Union[str, Path]] = None,
        max_length: int = 384,
        intra_op_num_threads: int = 0,
        local_files_only: bool = True,
    ) -> None:
        from transformers import AutoTokenizer

        self.model_dir = Path(model_dir)
        self.backend = backend.lower()
        if self.backend not in ("onnx", "pytorch"):
            raise ValueError(f"Unsupported backend: {backend}")
        self.max_length = int(max_length)
        self.tok = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=local_files_only
        )

        if self.backend == "onnx":
            import onnxruntime as ort

            onnx_cands = [
                self.model_dir / "onnx" / "model.onnx",
                self.model_dir / "model_fp32.onnx",
                self.model_dir / "model.onnx",
            ]
            onnx_path = next((p for p in onnx_cands if p.exists()), None)
            if onnx_path is None:
                raise FileNotFoundError(f"ONNX モデルが見つかりません: {onnx_cands}")
            opts = ort.SessionOptions()
            if intra_op_num_threads > 0:
                opts.intra_op_num_threads = int(intra_op_num_threads)
                opts.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(
                str(onnx_path), opts, providers=["CPUExecutionProvider"]
            )
            self.ort_inputs = {i.name for i in self.sess.get_inputs()}
            self.pt_model = None
        else:
            import torch
            from transformers import AutoModelForTokenClassification

            self.pt_model = AutoModelForTokenClassification.from_pretrained(
                str(self.model_dir), local_files_only=local_files_only
            )
            self.pt_model.eval()
            self._torch = torch
            self.sess = None

        resolved_master = master_dir or os.environ.get("JA_ADDRESS_MASTER_DIR")
        if resolved_master:
            linker_cls = _load_address_linker_class(self.model_dir)
            self.linker = linker_cls(resolved_master)
        else:
            self.linker = None

    def _predict_token_ids(self, enc: dict) -> List[int]:
        if self.backend == "onnx":
            import numpy as np

            feed = {}
            for k in ("input_ids", "attention_mask", "token_type_ids"):
                if k in self.ort_inputs:
                    vals = enc.get(k, [0] * len(enc["input_ids"]))
                    feed[k] = np.asarray([vals], dtype=np.int64)
            logits = self.sess.run(None, feed)[0][0]
            return [int(x) for x in logits.argmax(axis=-1)]

        torch = self._torch
        kw = {
            "input_ids": torch.tensor([enc["input_ids"]], dtype=torch.long),
            "attention_mask": torch.tensor([enc["attention_mask"]], dtype=torch.long),
        }
        if "token_type_ids" in enc:
            kw["token_type_ids"] = torch.tensor([enc["token_type_ids"]], dtype=torch.long)
        with torch.no_grad():
            logits = self.pt_model(**kw).logits[0]
        return [int(x) for x in logits.argmax(dim=-1).cpu().tolist()]

    def extract(self, text: str) -> List[dict]:
        """自由記述文から住所を抽出し、5要素へ分解、リンカー設定時はコード付与または UNRESOLVABLE を返す。"""
        if not text:
            return []
        enc, offsets = encode_with_offsets(self.tok, text, self.max_length)
        pred_ids = self._predict_token_ids(enc)
        comp_spans = decode_token_preds_to_components(text, offsets, pred_ids)
        addresses = group_components_to_addresses(text, comp_spans)
        if self.linker is not None:
            for addr in addresses:
                addr["link"] = self.linker.link_components(addr["components"])
        return addresses


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ja-address-ner-v1 推論CLI（抽出・分解・任意で辞書リンカー接続）"
    )
    ap.add_argument("--model-dir", default=".")
    ap.add_argument("--backend", choices=["onnx", "pytorch"], default="onnx")
    ap.add_argument("--master-dir", default=None, help="build_dictionary.py で作成した辞書ディレクトリ")
    ap.add_argument("--text", default=None)
    ap.add_argument("--file", default=None, help="1行1文書のテキストファイル")
    ap.add_argument("--max-length", type=int, default=384)
    args = ap.parse_args()

    if not args.text and not args.file:
        ap.error("--text または --file のいずれかを指定してください。")

    extractor = JaAddressExtractor(
        model_dir=args.model_dir,
        backend=args.backend,
        master_dir=args.master_dir,
        max_length=args.max_length,
        local_files_only=True,
    )

    texts = (
        [args.text]
        if args.text is not None
        else [l.rstrip("\r\n") for l in open(args.file, encoding="utf-8") if l.strip()]
    )
    for txt in texts:
        res = {"text": txt, "addresses": extractor.extract(txt)}
        print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
