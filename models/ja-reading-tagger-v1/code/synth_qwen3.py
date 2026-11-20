#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 オープンウェイト LLM (Qwen3, Apache 2.0) 合成＆独立逆検証スクリプト。

2フェーズ分離設計（VRAM スワップ 1 回のみ）:
- Phase A (qwen3:8b, Apache 2.0):
  同形異音語カタログの各 (語, 読み) ターゲットに対し、意味定義＋判別手がかり (Cues) 付きで
  多様な現代日本語文を生成し、一次ルール・形態素境界検証を通過した候補を保存する。
- Phase B (qwen3:14b, Apache 2.0):
  同一見出し語の異なる読み候補をランダムシャッフル混合バッチにして独立判定させ、
  生成時の想定読みと qwen3:14b 判定が一致し、かつ全文の非ターゲット漢字スパンも
  Sudachi == UniDic かつ統合辞書検証を通過した文のみを最終採用する。
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import fugashi
from sudachipy import dictionary, tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import homograph_catalog as hc  # noqa: E402
from lexicon_util import LexiconValidator  # noqa: E402
import phonology as ph  # noqa: E402
from span_extractor import SpanExtractor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MASTER_DIR = ROOT / "data" / "master"
DRAFTS_PATH = ROOT / "data" / "raw" / "qwen3_drafts.jsonl"
VERIFIED_PATH = ROOT / "data" / "raw" / "qwen3_verified.jsonl"
STATS_PATH = ROOT / "data" / "raw" / "qwen3_stats.json"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

_ALLOWED_SYNTH_RE = re.compile(
    r"^[ぁ-んァ-ヶー一-龥々ヶヵ〆〇、。「」！？・　 ]+$"
)
_LEAK_BRACKET_RE = re.compile(r"[\(\)（）《》【】\[\]]")

# 一般文脈に極端に偏りにくく合成で読み分け訓練が有効な重点カタログ対象選定
# 115語のうち、漢字数1〜3文字で文脈読み分けが明確に定義できる全語（稀な古典専用語「追従=ツイショウ」等は残しつつ
# 「大風=タイフウ」のような漢詩専用や「五月=サツキ」単漢字重複などを除くか含める）
# 原則: 2読み以上が現代日本語または一般的表現で使い分け可能な全90語以上を対象にする。
_SKIP_SENSES = {
    ("大風", "タイフウ"),   # 古典漢詩「大風起りて…」専用で現代「台風」と衝突するため
    ("大水", "ダイスイ"),   # 極端に稀
    ("赤子", "セキシ"),     # 極端に稀
    ("人間", "ジンカン"),   # 古典専用
    ("行方", "ギョウホウ"), # 仏教専門で現代語ではほぼ使われない
    ("大人", "タイジン"),   # 古典専用
}

# 青空文庫＋ROHAN実文に大量出現する超基礎単漢字・日常語の一部は合成枠から外し、
# 熟語・用言・レア読み単漢字などの重点語に GPU 合成時間を集中する
_SKIP_GROUPS = {
    "前", "上", "下", "内", "外", "先", "本", "気", "力", "世",
    "代", "月", "日", "私", "家", "星", "体", "良+い", "強+い",
    "行+く", "小川", "三月", "五月",
}

_STYLE_HINTS = [
    "日常会話、家族や友人とのやりとり、暮らしの風景など多様な場面で作り、文の書き出し（主語）を毎回変えてください。",
    "ニュース記事、ビジネスメール、報告書、解説文、エッセイ調など硬め〜自然な文体で作り、文の書き出しを毎回変えてください。",
    "学校・趣味・旅行・買い物・季節の情景など具体的な状況を盛り込み、異なる語彙を使って作ってください。",
]


def ollama_generate(
    model: str,
    prompt: str,
    temperature: float = 0.8,
    num_predict: int = 900,
    keep_alive: str = "10m",
    timeout: int = 180,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "repeat_penalty": 1.08,
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data.get("response", "")


def unload_ollama_model(model: str) -> None:
    """Ollama モデルを VRAM から即座にアンロードする。"""
    try:
        payload = {"model": model, "prompt": "", "stream": False, "keep_alive": 0}
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
    except Exception:
        pass


def parse_json_array(raw: str) -> List[str]:
    """LLM 出力テキストから文字列の JSON 配列を頑健に抽出する。"""
    txt = raw.strip()
    # コードフェンス除去
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    st = txt.find("[")
    ed = txt.rfind("]")
    if st < 0 or ed <= st:
        return []
    sub = txt[st : ed + 1]
    try:
        obj = json.loads(sub)
        if isinstance(obj, list):
            out = []
            for x in obj:
                if isinstance(x, str):
                    out.append(x.strip())
                elif isinstance(x, dict):
                    for k in ("sentence", "text", "文"):
                        if k in x and isinstance(x[k], str):
                            out.append(x[k].strip())
                            break
            return out
    except Exception:
        return []
    return []


def build_target_catalog() -> List[dict]:
    """合成対象となる同形異音語グループを構築する。"""
    groups: List[dict] = []
    for item in hc.HOMOGRAPH_CATALOG:
        if item["key"] in _SKIP_GROUPS:
            continue
        valid_r = [
            r for r in item["readings"] if (item["surface"], r["reading"]) not in _SKIP_SENSES
        ]
        if len(valid_r) >= 2:
            cp = dict(item)
            cp["readings"] = valid_r
            groups.append(cp)
    return groups


def contains_person_name(
    text: str, sudachi_tok, unidic_tagger, protected: Set[str]
) -> bool:
    for m in sudachi_tok.tokenize(text, tokenizer.Tokenizer.SplitMode.C):
        s = m.surface()
        if ph.has_kanji(s):
            pos = m.part_of_speech()
            if pos[1] == "固有名詞" and pos[2] == "人名" and s not in protected:
                return True
    for w in unidic_tagger(text):
        s = w.surface
        if ph.has_kanji(s):
            if w.feature.pos2 == "固有名詞" and w.feature.pos3 == "人名" and s not in protected:
                return True
    return False


def run_phase_a(
    groups: List[dict],
    batches_per_reading: int,
    sents_per_batch: int,
    draft_model: str,
    seed: int,
) -> List[dict]:
    """Phase A: qwen3:8b で候補文を生成し、一次ルール＋SpanExtractor 境界検証を通過した文を保存。"""
    t0 = time.time()
    se = SpanExtractor(MASTER_DIR)
    sudachi_tok = dictionary.Dictionary(dict="core").tokenizer()
    unidic_tagger = fugashi.Tagger()
    protected: Set[str] = {item["surface"] for item in hc.HOMOGRAPH_CATALOG} | {
        item["span_surface"] for item in hc.HOMOGRAPH_CATALOG
    }

    # 再開用チェックポイント読み込み
    existing: List[dict] = []
    seen_texts: Set[str] = set()
    done_targets: Counter = Counter()
    if DRAFTS_PATH.exists():
        with open(DRAFTS_PATH, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                existing.append(r)
                seen_texts.add(r["text"])
                done_targets[(r["key"], r["target_reading"])] += 1
        print(f"Resuming Phase A from {len(existing)} existing drafts.")

    counts = Counter()
    total_targets = sum(len(g["readings"]) for g in groups)
    t_idx = 0

    with open(DRAFTS_PATH, "a", encoding="utf-8") as fout:
        for g in groups:
            surf = g["surface"]
            span_surf = g["span_surface"]
            key = g["key"]
            cat = g["category"]
            for r_idx, rinfo in enumerate(g["readings"]):
                t_idx += 1
                t_read = rinfo["reading"]
                t_hira = ph.kata_to_hira(t_read)
                t_full_hira = ph.kata_to_hira(rinfo["full_reading"])

                need_min = batches_per_reading * (sents_per_batch - 4)
                if done_targets[(key, t_read)] >= need_min:
                    continue

                for bi in range(batches_per_reading):
                    style_hint = _STYLE_HINTS[(t_idx + bi) % len(_STYLE_HINTS)]
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
1. 各文には必ず「{surf}」をちょうど1回、文脈から確実に「{t_full_hira}」と読まれる形で使うこと（他の漢字と熟語にして別の語にしないこと。例：「人気」を「大人気」にしない）。
2. 「{t_hira}」「{t_full_hira}」などの読み仮名や括弧（）《》は絶対に出力文に書かないこと。
3. 人名（姓・名）、英字、算用数字（0-9）は絶対に使わないこと。
4. 1文の長さは15〜65文字程度、文末は「。」「！」「？」のいずれかにすること。
5. {style_hint} 同じフレーズや同じ文型（例：「〜のない」「この〜は」）を3回以上繰り返さず、動詞・助詞・語順を多様にすること。

出力形式（JSON配列のみ、前置きや説明不要）:
["文1。", "文2。", ...]"""

                    try:
                        raw = ollama_generate(
                            draft_model,
                            prompt,
                            temperature=0.82,
                            num_predict=1100,
                            keep_alive="10m",
                        )
                        cands = parse_json_array(raw)
                    except Exception as e:
                        counts["ollama_err"] += 1
                        continue

                    for s in cands:
                        counts["raw_generated"] += 1
                        # 終端正規化
                        s = s.strip().replace("\n", "")
                        if not s:
                            continue
                        if not s.endswith(("。", "！", "？")):
                            s += "。"
                        if not (14 <= len(s) <= 80):
                            counts["bad_len"] += 1
                            continue
                        if _LEAK_BRACKET_RE.search(s) or ph.PERSON_EXCLUDE_RE.search(s):
                            counts["leak_or_alnum"] += 1
                            continue
                        if not _ALLOWED_SYNTH_RE.match(s):
                            counts["bad_chars"] += 1
                            continue
                        # 読み仮名の漏洩チェック（2文字以上の読みが平仮名で本文に出現するのを防止、ただし「あ」「い」「き」等の1〜2文字送り仮名一般語との偶然一致は除く）
                        if len(t_full_hira) >= 3 and t_full_hira in s:
                            counts["reading_leak"] += 1
                            continue
                        if s.count(surf) != 1:
                            counts["surface_count_ne_1"] += 1
                            continue
                        if s in seen_texts:
                            counts["dup"] += 1
                            continue

                        # SpanExtractor で対象語が独立スパン (start == pos_idx, key == key) として抽出されるか確認
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

                        if contains_person_name(s, sudachi_tok, unidic_tagger, protected):
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
                        seen_texts.add(s)
                        existing.append(rec)
                        counts["draft_accepted"] += 1

            if t_idx % 20 == 0 or t_idx == total_targets:
                fout.flush()
                print(
                    f"[Phase A] targets {t_idx}/{total_targets}, total_drafts={len(existing)}, "
                    f"elapsed={time.time()-t0:.1f}s"
                )

    print("Phase A done:", json.dumps(dict(counts), ensure_ascii=False))
    return existing


def run_phase_b(
    drafts: List[dict],
    groups: List[dict],
    verify_model: str,
    batch_size: int,
    seed: int,
) -> List[dict]:
    """Phase B: qwen3:14b によるシャッフル混合バッチ独立逆検証＋全文辞書検証。"""
    t0 = time.time()
    rng = random.Random(seed)
    val = LexiconValidator(MASTER_DIR)
    se = SpanExtractor(MASTER_DIR)
    sudachi_tok = dictionary.Dictionary(dict="core").tokenizer()
    unidic_tagger = fugashi.Tagger()

    group_by_key = {g["key"]: g for g in groups}
    drafts_by_key: Dict[str, List[dict]] = defaultdict(list)
    for d in drafts:
        drafts_by_key[d["key"]].append(d)

    # 検証済みチェックポイントがあれば読込
    already_verified: Set[str] = set()
    verified_records: List[dict] = []
    if VERIFIED_PATH.exists():
        with open(VERIFIED_PATH, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                verified_records.append(r)
                already_verified.add(r["text"])
        print(f"Resuming Phase B from {len(verified_records)} already verified records.")

    counts = Counter()
    keys_list = sorted(drafts_by_key.keys())

    with open(VERIFIED_PATH, "a", encoding="utf-8") as fout:
        for ki, key in enumerate(keys_list, 1):
            g = group_by_key.get(key)
            if not g:
                continue
            items = [d for d in drafts_by_key[key] if d["text"] not in already_verified]
            if not items:
                continue
            # 必ずシャッフルして異なる読みを混合バッチにする（多数派慣性バイアス対策）
            rng.shuffle(items)

            # 選択肢ラベル A, B, C, ... と X (両義・不自然)
            letters = ["A", "B", "C", "D", "E"]
            read_to_letter = {r["reading"]: letters[idx] for idx, r in enumerate(g["readings"])}
            opt_lines = "\n".join(
                f"- {letters[idx]}: {ph.kata_to_hira(r['full_reading'])}（{r['sense']}）"
                for idx, r in enumerate(g["readings"])
            )
            valid_letters = {letters[idx] for idx in range(len(g["readings"]))}

            for b_start in range(0, len(items), batch_size):
                batch = items[b_start : b_start + batch_size]
                sent_lines = "\n".join(
                    f"{i+1}. {it['text']}" for i, it in enumerate(batch)
                )
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
                    raw = ollama_generate(
                        verify_model,
                        prompt,
                        temperature=0.0,
                        num_predict=200,
                        keep_alive="10m",
                    )
                    preds = parse_json_array(raw)
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

                    # 逆検証一致 -> 全文の他スパンが Sudachi == UniDic かつ辞書検証を通るか厳密検査
                    text = it["text"]
                    t_st, t_ed = it["target_start"], it["target_end"]
                    t_read = it["target_reading"]

                    # SpanExtractor で全漢字スパン抽出
                    sp_list = se.extract_spans(text)
                    # UniDic で全漢字位置の読み確認
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

                    # 全スパン検証
                    all_spans: List[dict] = []
                    full_ok = True
                    target_found = False

                    for sp in sp_list:
                        s_st, s_ed, s_surf = sp["start"], sp["end"], sp["surface"]
                        if s_st == t_st and s_ed == t_ed:
                            # ターゲットスパン: qwen3:14b 逆検証済み読みを採用（かつ辞書検証済み）
                            chosen_rk = t_read
                            target_found = True
                        else:
                            # 非ターゲットスパン: UniDic 側と一致するか確認（分割粒度差も連結で照合）
                            chosen_rk = sp["default_reading"]
                            ud_r = ud_map.get((s_st, s_ed))
                            if ud_r is None:
                                # 分割粒度差（例: 報告書 vs 報告+書）を確認
                                sub_parts = [
                                    (us, ue, ur)
                                    for (us, ue), ur in sorted(ud_map.items())
                                    if us >= s_st and ue <= s_ed
                                ]
                                if sub_parts and sub_parts[0][0] == s_st and sub_parts[-1][1] == s_ed:
                                    # 隙間がないか確認
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

                    # 原文上の全漢字カバー確認
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
                        "id": f"qwen3_{len(verified_records):06d}",
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
                    f"[Phase B] keys {ki}/{len(keys_list)}, verified={len(verified_records)}, "
                    f"elapsed={time.time()-t0:.1f}s"
                )

    val.close()
    print("Phase B done:", json.dumps(dict(counts), ensure_ascii=False))
    return verified_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches-per-reading", type=int, default=2)
    parser.add_argument("--sents-per-batch", type=int, default=16)
    parser.add_argument("--verify-batch-size", type=int, default=14)
    parser.add_argument("--draft-model", type=str, default="qwen3:8b")
    parser.add_argument("--verify-model", type=str, default="qwen3:14b")
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--skip-phase-a", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    groups = build_target_catalog()
    if args.limit_groups > 0:
        groups = groups[: args.limit_groups]
    print(f"Synth target groups: {len(groups)}, total readings: {sum(len(g['readings']) for g in groups)}")

    try:
        if not args.skip_phase_a:
            drafts = run_phase_a(
                groups,
                batches_per_reading=args.batches_per_reading,
                sents_per_batch=args.sents_per_batch,
                draft_model=args.draft_model,
                seed=args.seed,
            )
            # Phase A モデルを完全アンロード
            unload_ollama_model(args.draft_model)
        else:
            drafts = [json.loads(l) for l in open(DRAFTS_PATH, encoding="utf-8") if l.strip()]

        verified = run_phase_b(
            drafts,
            groups,
            verify_model=args.verify_model,
            batch_size=args.verify_batch_size,
            seed=args.seed,
        )
    finally:
        # 学習で GPU VRAM を空けるため、必ず両モデルをアンロード
        unload_ollama_model(args.draft_model)
        unload_ollama_model(args.verify_model)

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
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("Qwen3 synthesis complete:", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
