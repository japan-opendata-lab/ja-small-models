#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 本番用 Qwen3 (Apache 2.0) 合成＆独立逆検証スクリプト。

特徴:
- パイロット公開分割 (pilot/train, pilot/dev, pilot/homograph-eval) および
  パイロット生成データ (qwen3_drafts, qwen3_verified) 全体をブラックリスト化し、
  完全一致および 5-gram MinHash LSH (Jaccard >= 0.80) の重複文を一切排除する。
- パイロットで未合成だった基礎単漢字・用言23語＋誤答上位語＋新規低頻度一般多読語20語に
  2バッチずつ集中し、全3層・約129グループを新規生成・独立逆検証する。
"""
import argparse
import json
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import fugashi
from sudachipy import dictionary

sys.path.insert(0, str(Path(__file__).resolve().parent))
import homograph_catalog as hc  # noqa: E402
from lexicon_util import LexiconValidator  # noqa: E402
import phonology as ph  # noqa: E402
from span_extractor import SpanExtractor  # noqa: E402
import synth_qwen3 as sq  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MASTER_DIR = ROOT / "data" / "master"
PILOT_DIR = ROOT / "data" / "pilot"
RAW_DIR = ROOT / "data" / "raw"
PROD_DRAFTS_PATH = RAW_DIR / "qwen3_prod_drafts.jsonl"
PROD_VERIFIED_PATH = RAW_DIR / "qwen3_prod_verified.jsonl"
PROD_STATS_PATH = RAW_DIR / "qwen3_prod_stats.json"

_NORM_STRIP_RE = re.compile(r"[、。「」『』（）！？…・\s　]")
_NUM_PERM = 64
_BANDS = 16
_ROWS = 4

# 2バッチ集中対象（パイロット除外23語＋パイロット誤答上位語＋新規低頻度20語）
_HIGH_PRIORITY_KEYS = {
    "前", "上", "下", "内", "外", "先", "本", "気", "力", "世",
    "代", "月", "日", "私", "家", "星", "体", "良+い", "強+い",
    "行+く", "小川", "三月", "五月",
    "音", "間", "明日", "今日", "開+く", "開+いた", "空+く", "空+いた",
    "何時", "何人", "一時", "表", "額", "角", "紅葉", "家内", "早急",
    "風車", "気質", "生", "行", "行+った", "何", "後",
    "他", "翁", "米", "市", "傍", "牧場", "念仏", "外面", "憤怒",
    "歩", "大金", "上物", "昨夜", "続柄", "豚肉", "物怪", "下男",
    "実事", "生々", "関西",
}

_PROD_STYLE_HINTS = [
    "気象・季節の移ろい・自然風景や災害対策の話題で作り、文の出だしを毎回変えてください。",
    "職場での業務連絡、会議、商談、技術説明の文脈で作り、同じ主語や動詞を繰り返さないでください。",
    "家庭の食卓、買い物、地域の行事、近所との会話の場面で自然な口語・敬語を混ぜて作ってください。",
    "駅やバス・旅行・街歩き・道案内の場面で具体的な情景を入れて作ってください。",
    "学校の授業、図書館、読書感想、スポーツや部活動の場面で作ってください。",
    "新聞記事、地域ニュース、解説コラムのような客観的で引き締まった文体で作ってください。",
]


def norm_for_dedup(text: str) -> str:
    return _NORM_STRIP_RE.sub("", text)


def char_5grams(norm_txt: str) -> Set[str]:
    if len(norm_txt) <= 5:
        return {norm_txt}
    return {norm_txt[i : i + 5] for i in range(len(norm_txt) - 4)}


def compute_minhash(grams: Set[str]) -> Tuple[int, ...]:
    vals = [zlib.crc32(g.encode("utf-8")) & 0xFFFFFFFF for g in grams]
    sig = []
    for seed in range(_NUM_PERM):
        a = (seed * 2654435761 + 1) & 0xFFFFFFFF
        b = (seed * 2246822519 + 13) & 0xFFFFFFFF
        sig.append(min(((v * a + b) & 0xFFFFFFFF) for v in vals))
    return tuple(sig)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


class LSHBlacklist:
    """パイロットデータ・既存合成文との完全一致および近接重複 (Jaccard >= 0.80) 遮断器。"""

    def __init__(self, paths: List[Path]) -> None:
        self.exact_norms: Set[str] = set()
        self.grams_list: List[Set[str]] = []
        self.buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)

        for p in paths:
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    txt = json.loads(line)["text"]
                    self.add(txt)
        print(f"Blacklist loaded: {len(self.exact_norms)} texts indexed in LSH.", flush=True)

    def is_duplicate_or_near(self, text: str) -> bool:
        nt = norm_for_dedup(text)
        if not nt or nt in self.exact_norms:
            return True
        grams = char_5grams(nt)
        sig = compute_minhash(grams)
        cands: Set[int] = set()
        for bi in range(_BANDS):
            cands.update(self.buckets.get((bi, sig[bi * _ROWS : (bi + 1) * _ROWS]), ()))
        for idx in cands:
            if jaccard(grams, self.grams_list[idx]) >= 0.80:
                return True
        return False

    def add(self, text: str) -> None:
        nt = norm_for_dedup(text)
        if not nt or nt in self.exact_norms:
            return
        self.exact_norms.add(nt)
        grams = char_5grams(nt)
        sig = compute_minhash(grams)
        idx = len(self.grams_list)
        self.grams_list.append(grams)
        for bi in range(_BANDS):
            self.buckets[(bi, sig[bi * _ROWS : (bi + 1) * _ROWS])].append(idx)


def build_prod_targets() -> List[dict]:
    groups: List[dict] = []
    for item in hc.HOMOGRAPH_CATALOG:
        valid_r = [
            r for r in item["readings"] if (item["surface"], r["reading"]) not in sq._SKIP_SENSES
        ]
        if len(valid_r) >= 2:
            cp = dict(item)
            cp["readings"] = valid_r
            cp["is_high_priority"] = item["key"] in _HIGH_PRIORITY_KEYS
            groups.append(cp)
    return groups


def run_phase_a_prod(
    groups: List[dict],
    blacklist: LSHBlacklist,
    draft_model: str,
) -> List[dict]:
    t0 = time.time()
    se = SpanExtractor(MASTER_DIR)
    sudachi_tok = dictionary.Dictionary(dict="core").tokenizer()
    unidic_tagger = fugashi.Tagger()
    protected: Set[str] = {item["surface"] for item in hc.HOMOGRAPH_CATALOG} | {
        item["span_surface"] for item in hc.HOMOGRAPH_CATALOG
    }

    existing: List[dict] = []
    done_targets: Counter = Counter()
    if PROD_DRAFTS_PATH.exists():
        with open(PROD_DRAFTS_PATH, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                existing.append(r)
                blacklist.add(r["text"])
                done_targets[(r["key"], r["target_reading"])] += 1
        print(f"Resuming Phase A prod from {len(existing)} drafts.", flush=True)

    counts = Counter()
    total_targets = sum(len(g["readings"]) for g in groups)
    t_idx = 0

    with open(PROD_DRAFTS_PATH, "a", encoding="utf-8") as fout:
        for g in groups:
            surf = g["surface"]
            span_surf = g["span_surface"]
            key = g["key"]
            cat = g["category"]
            n_batches = 2 if g["is_high_priority"] else 1
            sents_per_batch = 16 if g["is_high_priority"] else 18
            need_min = n_batches * (sents_per_batch - 4)

            for rinfo in g["readings"]:
                t_idx += 1
                t_read = rinfo["reading"]
                t_hira = ph.kata_to_hira(t_read)
                t_full_hira = ph.kata_to_hira(rinfo["full_reading"])

                if done_targets[(key, t_read)] >= need_min:
                    continue

                for bi in range(n_batches):
                    style_hint = _PROD_STYLE_HINTS[(t_idx * 3 + bi) % len(_PROD_STYLE_HINTS)]
                    other_desc = " / ".join(
                        f"「{ph.kata_to_hira(o['reading'])}（{o['sense']}）」"
                        for o in g["readings"]
                        if o["reading"] != t_read
                    )
                    prompt = f"""次の日本語の語について、指定された「読み・意味」でしか読めない自然な現代日本語の文を {sents_per_batch} 文作成し、JSON配列（文字列のみ）で出力してください。

【対象語】「{surf}」
【指定の読み】{t_full_hira}（漢字部分「{span_surf}」＝ {t_hira}）
【指定の意味】{rinfo['sense']}
【判別の手がかり例】{rinfo['cues']}
【避けるべき別読み】{other_desc}

【厳守ルール】
1. 各文には必ず「{surf}」をちょうど1回、文脈から確実に「{t_full_hira}」と読まれる形で使うこと。
2. 「{t_hira}」「{t_full_hira}」などの読み仮名や括弧（）《》は絶対に出力文に書かないこと。
3. 人名（姓・名）、英字、算用数字（0-9）は絶対に使わないこと。
4. 1文の長さは15〜65文字程度、文末は「。」「！」「？」のいずれかにすること。
5. {style_hint} 同じフレーズや同じ文型を3回以上繰り返さず、動詞・助詞・語順を多様にすること。

出力形式（JSON配列のみ、前置きや説明不要）:
["文1。", "文2。", ...]"""

                    try:
                        raw = sq.ollama_generate(
                            draft_model,
                            prompt,
                            temperature=0.86,
                            num_predict=1250,
                            keep_alive="10m",
                        )
                        cands = sq.parse_json_array(raw)
                    except Exception:
                        counts["ollama_err"] += 1
                        continue

                    for s in cands:
                        counts["raw_generated"] += 1
                        s = s.strip().replace("\n", "")
                        if not s:
                            continue
                        if not s.endswith(("。", "！", "？")):
                            s += "。"
                        if not (14 <= len(s) <= 80):
                            counts["bad_len"] += 1
                            continue
                        if sq._LEAK_BRACKET_RE.search(s) or ph.PERSON_EXCLUDE_RE.search(s):
                            counts["leak_or_alnum"] += 1
                            continue
                        if not sq._ALLOWED_SYNTH_RE.match(s):
                            counts["bad_chars"] += 1
                            continue
                        if len(t_full_hira) >= 3 and t_full_hira in s:
                            counts["reading_leak"] += 1
                            continue
                        if s.count(surf) != 1:
                            counts["surface_count_ne_1"] += 1
                            continue
                        if blacklist.is_duplicate_or_near(s):
                            counts["dup_or_near_pilot"] += 1
                            continue

                        pos_idx = s.find(surf)
                        sp_list = se.extract_spans(s)
                        matched_sp = [
                            sp
                            for sp in sp_list
                            if sp["start"] == pos_idx and sp["key"] == key
                        ]
                        if not matched_sp:
                            counts["span_mismatch"] += 1
                            continue

                        if sq.contains_person_name(s, sudachi_tok, unidic_tagger, protected):
                            counts["person_name"] += 1
                            continue

                        rec = {
                            "text": s,
                            "key": key,
                            "surface": surf,
                            "span_surface": span_surf,
                            "category": cat,
                            "target_start": matched_sp[0]["start"],
                            "target_end": matched_sp[0]["end"],
                            "target_reading": t_read,
                            "target_full_reading": rinfo["full_reading"],
                        }
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        blacklist.add(s)
                        existing.append(rec)
                        counts["draft_accepted"] += 1

            if t_idx % 20 == 0 or t_idx == total_targets:
                fout.flush()
                print(
                    f"[Prod Phase A] targets {t_idx}/{total_targets}, drafts={len(existing)}, "
                    f"elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )

    print("Prod Phase A done:", json.dumps(dict(counts), ensure_ascii=False), flush=True)
    return existing


def run_phase_b_prod(
    drafts: List[dict],
    groups: List[dict],
    verify_model: str,
    batch_size: int,
    seed: int,
) -> List[dict]:
    t0 = time.time()
    rng = random.Random(seed)
    val = LexiconValidator(MASTER_DIR)
    se = SpanExtractor(MASTER_DIR)
    unidic_tagger = fugashi.Tagger()

    group_by_key = {g["key"]: g for g in groups}
    drafts_by_key: Dict[str, List[dict]] = defaultdict(list)
    for d in drafts:
        drafts_by_key[d["key"]].append(d)

    already_verified: Set[str] = set()
    verified_records: List[dict] = []
    if PROD_VERIFIED_PATH.exists():
        with open(PROD_VERIFIED_PATH, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                verified_records.append(r)
                already_verified.add(r["text"])
        print(f"Resuming Phase B prod from {len(verified_records)} verified.", flush=True)

    counts = Counter()
    keys_list = sorted(drafts_by_key.keys())

    with open(PROD_VERIFIED_PATH, "a", encoding="utf-8") as fout:
        for ki, key in enumerate(keys_list, 1):
            g = group_by_key.get(key)
            if not g:
                continue
            items = [d for d in drafts_by_key[key] if d["text"] not in already_verified]
            if not items:
                continue
            rng.shuffle(items)

            letters = ["A", "B", "C", "D", "E"]
            read_to_letter = {r["reading"]: letters[idx] for idx, r in enumerate(g["readings"])}
            opt_lines = "\n".join(
                f"- {letters[idx]}: {ph.kata_to_hira(r['full_reading'])}（{r['sense']}）"
                for idx, r in enumerate(g["readings"])
            )
            valid_letters = {letters[idx] for idx in range(len(g["readings"]))}

            for b_start in range(0, len(items), batch_size):
                batch = items[b_start : b_start + batch_size]
                sent_lines = "\n".join(f"{i+1}. {it['text']}" for i, it in enumerate(batch))
                prompt = f"""次の各文に含まれる「{g['surface']}」の文脈上の正しい読みを、下の選択肢から1つずつ選んでください。
少しでも意味が通らない文や、どちらの読みとも解釈できて曖昧な文は必ず "X" にしてください。

選択肢:
{opt_lines}
- X: どちらとも取れる・曖昧・不自然・対象語が別の語の一部

出力ルール:
各文番号の順番通りに回答記号のみを並べた JSON 配列だけを出力してください（長さ {len(batch)} の配列）。例: ["A", "B", ...]

文リスト:
{sent_lines}"""

                try:
                    raw = sq.ollama_generate(
                        verify_model,
                        prompt,
                        temperature=0.0,
                        num_predict=220,
                        keep_alive="10m",
                    )
                    preds = sq.parse_json_array(raw)
                except Exception:
                    counts["verify_err"] += len(batch)
                    continue

                if len(preds) != len(batch):
                    counts["len_mismatch"] += len(batch)
                    continue

                for it, pred_char in zip(batch, preds):
                    counts["total_judged"] += 1
                    p_norm = pred_char.strip().upper()[:1]
                    expected_letter = read_to_letter.get(it["target_reading"])
                    if p_norm not in valid_letters or p_norm != expected_letter:
                        counts["rejected_by_qwen14b"] += 1
                        continue

                    text = it["text"]
                    t_st, t_ed = it["target_start"], it["target_end"]
                    t_read = it["target_reading"]
                    sp_list = se.extract_spans(text)

                    ud_map: Dict[Tuple[int, int], str] = {}
                    cur_p = 0
                    ud_ok = True
                    for w in unidic_tagger(text):
                        ws = w.surface
                        idx = text.find(ws, cur_p)
                        if idx < 0:
                            ud_ok = False
                            break
                        cur_p = idx + len(ws)
                        if not ph.has_kanji(ws):
                            continue
                        if not w.feature.kana:
                            ud_ok = False
                            break
                        wrk = ph.normalize_kana(w.feature.kana)
                        st_res = ph.strip_okurigana(ws, wrk)
                        if st_res is None:
                            ud_ok = False
                            break
                        for rs, re_, ks, kr in st_res:
                            ud_map[(idx + rs, idx + re_)] = kr
                    if not ud_ok:
                        counts["rejected_unidic_parse"] += 1
                        continue

                    all_spans: List[dict] = []
                    full_ok = True
                    target_found = False

                    for sp in sp_list:
                        s_st, s_ed, s_surf = sp["start"], sp["end"], sp["surface"]
                        if s_st == t_st and s_ed == t_ed:
                            chosen_rk = t_read
                            target_found = True
                        else:
                            chosen_rk = sp["default_reading"]
                            ud_r = ud_map.get((s_st, s_ed))
                            if ud_r is None:
                                sub_parts = [
                                    (us, ue, ur)
                                    for (us, ue), ur in sorted(ud_map.items())
                                    if us >= s_st and ue <= s_ed
                                ]
                                if sub_parts and sub_parts[0][0] == s_st and sub_parts[-1][1] == s_ed:
                                    contiguous = all(
                                        sub_parts[idx][1] == sub_parts[idx + 1][0]
                                        for idx in range(len(sub_parts) - 1)
                                    )
                                    if contiguous:
                                        ud_r = "".join(p[2] for p in sub_parts)
                            if ud_r is None or ud_r != chosen_rk:
                                full_ok = False
                                break

                        lead = text[max(0, s_st - 2) : s_st]
                        trail = text[s_ed : s_ed + 4]
                        if not val.verify_span_reading(
                            s_surf, chosen_rk, trailing_text=trail, leading_text=lead
                        ):
                            full_ok = False
                            break
                        all_spans.append(
                            {
                                "start": s_st,
                                "end": s_ed,
                                "surface": s_surf,
                                "reading": ph.kata_to_hira(chosen_rk),
                                "reading_kata": chosen_rk,
                                "pron": val.to_pron(chosen_rk, surface=s_surf),
                                "is_target": (s_st == t_st and s_ed == t_ed),
                            }
                        )

                    if not (full_ok and target_found):
                        counts["rejected_non_target_disagree"] += 1
                        continue

                    covered = [False] * len(text)
                    for sp in all_spans:
                        for p in range(sp["start"], sp["end"]):
                            covered[p] = True
                    if any(ph.is_kanji_char(text[p]) and not covered[p] for p in range(len(text))):
                        counts["rejected_uncovered_kanji"] += 1
                        continue

                    counts["verified_accepted"] += 1
                    already_verified.add(text)
                    target_sp_obj = [sp for sp in all_spans if sp["is_target"]][0]
                    rec = {
                        "id": f"qwen3_p_{len(verified_records):06d}",
                        "source": "qwen3_synth",
                        "license": "Apache-2.0",
                        "models": {"draft": "qwen3:8b", "verify": verify_model},
                        "text": text,
                        "spans": all_spans,
                        "target_key": key,
                        "target_span": target_sp_obj,
                        "category": it["category"],
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    verified_records.append(rec)

            if ki % 15 == 0 or ki == len(keys_list):
                fout.flush()
                print(
                    f"[Prod Phase B] keys {ki}/{len(keys_list)}, verified={len(verified_records)}, "
                    f"elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )

    val.close()
    print("Prod Phase B done:", json.dumps(dict(counts), ensure_ascii=False), flush=True)
    return verified_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-batch-size", type=int, default=14)
    parser.add_argument("--draft-model", type=str, default="qwen3:8b")
    parser.add_argument("--verify-model", type=str, default="qwen3:14b")
    parser.add_argument("--skip-phase-a", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    t0 = time.time()
    groups = build_prod_targets()
    print(
        f"Prod synth target groups: {len(groups)} "
        f"(high_priority={sum(1 for g in groups if g['is_high_priority'])}), "
        f"total_readings={sum(len(g['readings']) for g in groups)}",
        flush=True,
    )

    blacklist = LSHBlacklist(
        [
            PILOT_DIR / "train.jsonl",
            PILOT_DIR / "dev.jsonl",
            PILOT_DIR / "homograph-eval.jsonl",
            RAW_DIR / "qwen3_drafts.jsonl",
            RAW_DIR / "qwen3_verified.jsonl",
        ]
    )

    try:
        if not args.skip_phase_a:
            drafts = run_phase_a_prod(groups, blacklist, draft_model=args.draft_model)
            sq.unload_ollama_model(args.draft_model)
        else:
            drafts = [json.loads(l) for l in open(PROD_DRAFTS_PATH, encoding="utf-8") if l.strip()]

        verified = run_phase_b_prod(
            drafts,
            groups,
            verify_model=args.verify_model,
            batch_size=args.verify_batch_size,
            seed=args.seed,
        )
    finally:
        sq.unload_ollama_model(args.draft_model)
        sq.unload_ollama_model(args.verify_model)

    by_key_read = Counter((r["target_key"], r["target_span"]["reading_kata"]) for r in verified)
    stats = {
        "target_groups": len(groups),
        "total_drafts": len(drafts),
        "total_verified": len(verified),
        "pass_rate": round(len(verified) / max(1, len(drafts)), 4),
        "unique_keys_verified": len({r["target_key"] for r in verified}),
        "elapsed_sec": round(time.time() - t0, 2),
        "min_per_target": min(by_key_read.values()) if by_key_read else 0,
        "median_per_target": sorted(by_key_read.values())[len(by_key_read) // 2] if by_key_read else 0,
    }
    with open(PROD_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("Prod Qwen3 synthesis complete:", json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
