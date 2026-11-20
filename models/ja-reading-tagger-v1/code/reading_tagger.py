#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 構造化読み推定 参照実装。

学習済みチェックポイント（ModernBERT / BERT-v3 互換の TokenClassification モデル）と
統合辞書・決定的スパン抽出器を用い、入力文を一切改変せずに
[{"start", "end", "surface", "reading", "pron"}] を返す非生成の汎用部品。
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offsets_util as ou  # noqa: E402
import phonology as ph  # noqa: E402
from span_extractor import SpanExtractor  # noqa: E402


class JaReadingTagger:
    """日本語テキストから漢字スパンの表記読み・発音読みを文脈推定するタグ付け器。"""

    def __init__(
        self,
        model_dir: Path,
        master_dir: Path,
        device: Optional[str] = None,
        max_len: int = 256,
        override_tau: Optional[float] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.master_dir = Path(master_dir)
        self.max_len = max_len
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

        with open(self.model_dir / "label_map.json", encoding="utf-8") as f:
            lmap = json.load(f)
        self.label2id: Dict[str, int] = lmap["label2id"]

        if override_tau is None:
            cal_path = self.model_dir / "calibration.json"
            if cal_path.exists():
                with open(cal_path, encoding="utf-8") as f:
                    cal = json.load(f)
                self.override_tau = float(cal.get("override_tau", 0.0))
            else:
                self.override_tau = 0.0
        else:
            self.override_tau = float(override_tau)

        self.extractor = SpanExtractor(self.master_dir)

    @torch.no_grad()
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[List[dict]]:
        """複数文に対してバッチ推論し、各文のスパン予測リストを返す（入力は text のみ）。"""
        all_results: List[List[dict]] = []
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0

        for b_start in range(0, len(texts), batch_size):
            batch_texts = texts[b_start : b_start + batch_size]
            encs: List[dict] = []
            offsets_list: List[List] = []
            spans_list: List[List[dict]] = []

            for txt in batch_texts:
                enc, offsets = ou.encode_with_offsets(self.tok, txt, max_len=self.max_len)
                encs.append(enc)
                offsets_list.append(offsets)
                spans_list.append(self.extractor.extract_spans(txt))

            maxlen = max(len(e["input_ids"]) for e in encs)
            input_ids = []
            attn = []
            ttids = []
            has_tti = "token_type_ids" in encs[0]
            for e in encs:
                pad_n = maxlen - len(e["input_ids"])
                input_ids.append(e["input_ids"] + [pad_id] * pad_n)
                attn.append(e["attention_mask"] + [0] * pad_n)
                if has_tti:
                    ttids.append(e["token_type_ids"] + [0] * pad_n)

            model_in = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self.device),
                "attention_mask": torch.tensor(attn, dtype=torch.long, device=self.device),
            }
            if has_tti:
                model_in["token_type_ids"] = torch.tensor(ttids, dtype=torch.long, device=self.device)

            logits = self.model(**model_in).logits.float()  # [B, T, C]

            for bi, (txt, offsets, sp_list) in enumerate(
                zip(batch_texts, offsets_list, spans_list)
            ):
                sent_logits = logits[bi]  # [T, C]
                out_spans: List[dict] = []
                for sp in sp_list:
                    cands = sp["candidates"]
                    default_rk = sp["default_reading"]
                    chosen_rk = default_rk

                    if len(cands) >= 2:
                        # 候補のうちモデルラベル空間に存在するクラスのみを競争させる
                        cand_label_ids = []
                        for rk in cands:
                            lid = self.label2id.get(f"{sp['key']}|{rk}")
                            if lid is not None:
                                cand_label_ids.append((rk, lid))

                        if len(cand_label_ids) >= 2:
                            tok_idxs = ou.span_to_token_indices(offsets, sp["start"], sp["end"])
                            if tok_idxs:
                                span_vec = sent_logits[tok_idxs].mean(dim=0)
                                cand_vals = torch.tensor(
                                    [float(span_vec[lid].item()) for _, lid in cand_label_ids],
                                    dtype=torch.float32,
                                )
                                probs = F.softmax(cand_vals, dim=0)
                                best_idx = int(torch.argmax(probs).item())
                                best_rk = cand_label_ids[best_idx][0]
                                best_prob = float(probs[best_idx].item())
                                if best_rk != default_rk and best_prob >= self.override_tau:
                                    chosen_rk = best_rk

                    out_spans.append(
                        {
                            "start": sp["start"],
                            "end": sp["end"],
                            "surface": sp["surface"],
                            "reading": ph.kata_to_hira(chosen_rk),
                            "pron": self.extractor.to_pron(chosen_rk, surface=sp["surface"]),
                        }
                    )
                all_results.append(out_spans)
        return all_results

    def predict(self, text: str) -> List[dict]:
        """1文から構造化読みリストを返す。"""
        return self.predict_batch([text], batch_size=1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--master-dir", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    tagger = JaReadingTagger(Path(args.model_dir), Path(args.master_dir))
    res = tagger.predict(args.text)
    print(json.dumps({"text": args.text, "spans": res}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
