#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 本番用 Qwen3 欠落・極少ターゲット補完スクリプト。

特徴:
- data/raw/qwen3_prod_verified.jsonl で検証済み件数が 6 件未満の全カタログターゲットを対象に、
  qwen3:14b (Apache-2.0) 1 モデル完結（VRAM スワップ 0 回）で Phase A 生成＆Phase B 独立逆検証を行う。
- data/prod/sealed-test.jsonl および data/prod/dev.jsonl には一切触れず、
  母集団ファイル（aozora_prod_clean / qwen3_prod_verified / pilot 各ファイル等）全件を
  LSH ブラックリスト（完全一致＋5-gram MinHash Jaccard >= 0.80）に登録して完全非重複を保証する。
"""
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

import fugashi
from sudachipy import dictionary

sys.path.insert(0, str(Path(__file__).resolve().parent))
import homograph_catalog as hc  # noqa: E402
from lexicon_util import LexiconValidator  # noqa: E402
import phonology as ph  # noqa: E402
from span_extractor import SpanExtractor  # noqa: E402
import synth_qwen3 as sq  # noqa: E402
import synth_qwen3_prod as sqp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MASTER_DIR = ROOT / "data" / "master"
PILOT_DIR = ROOT / "data" / "pilot"
RAW_DIR = ROOT / "data" / "raw"

TOPUP_DRAFTS_PATH = RAW_DIR / "qwen3_prod_topup_drafts.jsonl"
TOPUP_VERIFIED_PATH = RAW_DIR / "qwen3_prod_topup_verified.jsonl"
TOPUP_STATS_PATH = RAW_DIR / "qwen3_prod_topup_stats.json"


def count_existing_verified() -> Counter:
    """qwen3_prod_verified.jsonl における既存の (target_key, reading_kata) 件数を集計する。"""
    cnt: Counter = Counter()
    p = RAW_DIR / "qwen3_prod_verified.jsonl"
    if not p.exists():
        return cnt
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cnt[(r["target_key"], r["target_span"]["reading_kata"])] += 1
    return cnt


def load_existing_anchors(max_per_key_read: int = 4) -> Dict[str, Dict[str, List[dict]]]:
    """Phase B 混合シャッフル用に qwen3_prod_verified.jsonl からキー×読み別のアンカー文を読み込む。"""
    anchors: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    p = RAW_DIR / "qwen3_prod_verified.jsonl"
    if not p.exists():
        return anchors
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            k = r["target_key"]
            rk = r["target_span"]["reading_kata"]
            if len(anchors[k][rk]) < max_per_key_read:
                anchors[k][rk].append(
                    {
                        "text": r["text"],
                        "key": k,
                        "target_reading": rk,
                        "is_anchor": True,
                    }
                )
    return anchors


def collect_topup_targets(existing_cnt: Counter, min_verified: int = 6) -> List[dict]:
    """検証済み件数が min_verified 未満のターゲットのみを抽出してグループ化する。"""
    groups: List[dict] = []
    for item in hc.HOMOGRAPH_CATALOG:
        key = item["key"]
        valid_r = [
            r for r in item["readings"] if (item["surface"], r["reading"]) not in sq._SKIP_SENSES
        ]
        if len(valid_r) < 2:
            continue
        need_r = []
        for r in valid_r:
            cur = existing_cnt[(key, r["reading"])]
            if cur < min_verified:
                rcp = dict(r)
                rcp["existing_verified"] = cur
                need_r.append(rcp)
        if need_r:
            cp = dict(item)
            cp["all_valid_readings"] = valid_r
            cp["need_readings"] = need_r
            groups.append(cp)
    return groups


def build_extra_constraint(key: str, t_read: str) -> str:
    """特定語（時間表記・一日・単漢字単独用法）に対する誤結合防止の補助制約を返す。"""
    lines: List[str] = []
    if key in {"十分", "一分"} and t_read in {"ジュップン", "イップン"}:
        lines.append(
            "6. 【重要制約】対象語の直後に「間」「前」「後」などの漢字を続けず、"
            "「で」「ほど」「も」「だけ」「すぎ」「遅れ」などのひらがな・助詞を続けること"
            "（例：「十分ほど待つ」「一分も遅れない」）。"
        )
    elif key == "一日" and t_read == "ツイタチ":
        lines.append(
            "6. 【重要制約】直前に漢数字（二十・三十・十月 等）を置かず、"
            "「毎月一日」「今月の一日」「来月の一日」「新年度の一日」のように月の初日の意味で使うこと。"
        )
    elif key in {"日", "月", "中", "何"}:
        lines.append(
            "6. 【重要制約】対象語が他の漢字熟語の一部（例：「今日」「今月」「日中」「何か」以外の熟語連結等）"
            "にならないよう、対象語の直前と直後をひらがな（助詞・用言）や句読点で自然に挟んで単独で使うこと。"
        )
    else:
        lines.append(
            "6. 日常会話・暮らし・仕事・社会・学校・自然など多様な場面で作り、同じ書き出しや文型を繰り返さないこと。"
        )
    return "\n".join(lines)


def generate_batch_and_filter(
    g: dict,
    rinfo: dict,
    n_sents: int,
    model_name: str,
    blacklist: sqp.LSHBlacklist,
    se: SpanExtractor,
    sudachi_tok,
    unidic_tagger,
    protected: Set[str],
    fout,
    drafts_out: List[dict],
    counts: Counter,
) -> int:
    """1バッチ分を qwen3:14b で生成し、一次フィルタ通過分を書き込む。戻り値は新規採択数。"""
    surf = g["surface"]
    span_surf = g["span_surface"]
    key = g["key"]
    cat = g["category"]
    t_read = rinfo["reading"]
    t_hira = ph.kata_to_hira(t_read)
    t_full_hira = ph.kata_to_hira(rinfo["full_reading"])

    other_desc = " / ".join(
        f"「{ph.kata_to_hira(o['reading'])}（{o['sense']}）」"
        for o in g["all_valid_readings"]
        if o["reading"] != t_read
    )
    extra_rule = build_extra_constraint(key, t_read)

    prompt = f"""次の日本語の語について、指定された「読み・意味」でしか読めない自然な現代日本語の文を {n_sents} 文作成し、JSON配列（文字列のみ）で出力してください。

【対象語】「{surf}」
【指定の読み】{t_full_hira}（漢字部分「{span_surf}」＝ {t_hira}）
【指定の意味】{rinfo['sense']}
【判別の手がかり例】{rinfo['cues']}
【避けるべき別読み】{other_desc}

【厳守ルール】
1. 各文には必ず「{surf}」をちょうど1回、文脈から確実に「{t_full_hira}」と読まれる形で使うこと（他の漢字とくっつけて別語にしないこと）。
2. 「{t_hira}」「{t_full_hira}」などの読み仮名や括弧（）《》は絶対に出力文に書かないこと。
3. 人名（姓・名）、英字、算用数字（0-9）は絶対に使わないこと。
4. 1文の長さは15〜65文字程度、文末は「。」「！」「？」のいずれかにすること。
5. 他の漢語や動詞も自然で読みが一意に決まる現代日本語にすること。
{extra_rule}

出力形式（JSON配列のみ、前置きや説明不要）:
["文1。", "文2。", ...]"""

    try:
        raw = sq.ollama_generate(
            model_name,
            prompt,
            temperature=0.80,
            num_predict=1400,
            keep_alive="10m",
        )
        cands = sq.parse_json_array(raw)
    except Exception:
        counts["ollama_err"] += 1
        return 0

    accepted_here = 0
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
            counts["dup_or_near"] += 1
            continue

        pos_idx = s.find(surf)
        sp_list = se.extract_spans(s)
        matched_sp = [
            sp for sp in sp_list if sp["start"] == pos_idx and sp["key"] == key
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
        drafts_out.append(rec)
        counts["draft_accepted"] += 1
        accepted_here += 1
    return accepted_here


def run_phase_a_topup(
    groups: List[dict],
    blacklist: sqp.LSHBlacklist,
    model_name: str,
) -> List[dict]:
    """不足ターゲットに対する Phase A 生成＆一次フィルタを実行する。"""
    t0 = time.time()
    se = SpanExtractor(MASTER_DIR)
    sudachi_tok = dictionary.Dictionary(dict="core").tokenizer()
    unidic_tagger = fugashi.Tagger()
    protected: Set[str] = {item["surface"] for item in hc.HOMOGRAPH_CATALOG} | {
        item["span_surface"] for item in hc.HOMOGRAPH_CATALOG
    }

    existing: List[dict] = []
    done_targets: Counter = Counter()
    if TOPUP_DRAFTS_PATH.exists():
        with open(TOPUP_DRAFTS_PATH, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                existing.append(r)
                blacklist.add(r["text"])
                done_targets[(r["key"], r["target_reading"])] += 1
        print(f"Resuming Topup Phase A from {len(existing)} drafts.", flush=True)

    counts: Counter = Counter()
    total_targets = sum(len(g["need_readings"]) for g in groups)
    t_idx = 0

    with open(TOPUP_DRAFTS_PATH, "a", encoding="utf-8") as fout:
        for g in groups:
            key = g["key"]
            for rinfo in g["need_readings"]:
                t_idx += 1
                t_read = rinfo["reading"]
                cur_ver = rinfo["existing_verified"]
                n_batches = 2 if cur_ver == 0 else 1
                sents_per_batch = 14 if cur_ver == 0 else 16
                target_goal = 10

                if done_targets[(key, t_read)] >= target_goal:
                    continue

                for _ in range(n_batches):
                    got = generate_batch_and_filter(
                        g,
                        rinfo,
                        sents_per_batch,
                        model_name,
                        blacklist,
                        se,
                        sudachi_tok,
                        unidic_tagger,
                        protected,
                        fout,
                        existing,
                        counts,
                    )
                    done_targets[(key, t_read)] += got

                # 不足（新規ドラフト < 10 件）なら自動で追加 1 回リトライ
                if done_targets[(key, t_read)] < target_goal:
                    counts["retry_triggered"] += 1
                    got_retry = generate_batch_and_filter(
                        g,
                        rinfo,
                        16,
                        model_name,
                        blacklist,
                        se,
                        sudachi_tok,
                        unidic_tagger,
                        protected,
                        fout,
                        existing,
                        counts,
                    )
                    done_targets[(key, t_read)] += got_retry

                if t_idx % 10 == 0 or t_idx == total_targets:
                    fout.flush()
                    print(
                        f"[Topup Phase A] targets {t_idx}/{total_targets}, "
                        f"drafts={len(existing)}, elapsed={time.time()-t0:.1f}s",
                        flush=True,
                    )

    print("Topup Phase A done:", json.dumps(dict(counts), ensure_ascii=False), flush=True)
    return existing


def judge_batch_with_retry(
    model_name: str,
    surf: str,
    opt_lines: str,
    batch: List[dict],
    counts: Counter,
) -> List[tuple]:
    """バッチ判定を実行し、件数不一致時は batch_size=5 で 1 回リトライして [(item, pred_char)] を返す。"""

    def _call_one(sub_batch: List[dict]) -> List[str]:
        sent_lines = "\n".join(f"{i+1}. {it['text']}" for i, it in enumerate(sub_batch))
        prompt = f"""次の各文に含まれる「{surf}」の文脈上の正しい読みを、下の選択肢から1つずつ選んでください。
少しでも意味が通らない文や、どちらの読みとも解釈できて曖昧な文は必ず "X" にしてください。

選択肢:
{opt_lines}
- X: どちらとも取れる・曖昧・不自然・対象語が別の語の一部

出力ルール:
各文番号の順番通りに回答記号のみを並べた JSON 配列だけを出力してください（長さ {len(sub_batch)} の配列）。例: ["A", "B", ...]

文リスト:
{sent_lines}"""
        raw = sq.ollama_generate(
            model_name,
            prompt,
            temperature=0.0,
            num_predict=220,
            keep_alive="10m",
        )
        return sq.parse_json_array(raw)

    try:
        preds = _call_one(batch)
        if len(preds) == len(batch):
            return list(zip(batch, preds))
    except Exception:
        pass

    # 件数不一致・例外時は半サイズ (5) で 1 回リトライ
    counts["len_mismatch_retry"] += 1
    out: List[tuple] = []
    half = max(1, min(5, len(batch) // 2))
    for s_idx in range(0, len(batch), half):
        sub = batch[s_idx : s_idx + half]
        try:
            sub_preds = _call_one(sub)
            if len(sub_preds) == len(sub):
                out.extend(zip(sub, sub_preds))
            else:
                counts["len_mismatch_final"] += len(sub)
        except Exception:
            counts["verify_err"] += len(sub)
    return out


def run_phase_b_topup(
    drafts: List[dict],
    verify_model: str,
    batch_size: int,
    seed: int,
) -> List[dict]:
    """Phase B: 既存検証済み同キー別読みアンカーと混合シャッフルして qwen3:14b 独立逆検証＋全文辞書検証を行う。"""
    t0 = time.time()
    rng = random.Random(seed)
    val = LexiconValidator(MASTER_DIR)
    se = SpanExtractor(MASTER_DIR)
    unidic_tagger = fugashi.Tagger()

    # 全カタログ有効グループ辞書
    group_by_key: Dict[str, dict] = {}
    for item in hc.HOMOGRAPH_CATALOG:
        vr = [r for r in item["readings"] if (item["surface"], r["reading"]) not in sq._SKIP_SENSES]
        if len(vr) >= 2:
            cp = dict(item)
            cp["readings"] = vr
            group_by_key[item["key"]] = cp

    anchors_map = load_existing_anchors(max_per_key_read=4)
    drafts_by_key: Dict[str, List[dict]] = defaultdict(list)
    for d in drafts:
        d_cp = dict(d)
        d_cp["is_anchor"] = False
        drafts_by_key[d["key"]].append(d_cp)

    already_verified: Set[str] = set()
    verified_records: List[dict] = []
    if TOPUP_VERIFIED_PATH.exists():
        with open(TOPUP_VERIFIED_PATH, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                verified_records.append(r)
                already_verified.add(r["text"])
        print(f"Resuming Topup Phase B from {len(verified_records)} verified.", flush=True)

    counts: Counter = Counter()
    keys_list = sorted(drafts_by_key.keys())

    with open(TOPUP_VERIFIED_PATH, "a", encoding="utf-8") as fout:
        for ki, key in enumerate(keys_list, 1):
            g = group_by_key.get(key)
            if not g:
                continue
            new_items = [d for d in drafts_by_key[key] if d["text"] not in already_verified]
            if not new_items:
                continue

            # 同キーの別読みアンカー（最大 8 件）を混合して単一読みバイアスを防止
            new_reads = {d["target_reading"] for d in new_items}
            other_anchors: List[dict] = []
            same_anchors: List[dict] = []
            for rk, alist in anchors_map.get(key, {}).items():
                if rk not in new_reads:
                    other_anchors.extend(alist)
                else:
                    same_anchors.extend(alist)
            anchor_pool = (other_anchors + same_anchors)[:8]

            mixed = list(new_items) + list(anchor_pool)
            rng.shuffle(mixed)

            letters = ["A", "B", "C", "D", "E"]
            read_to_letter = {r["reading"]: letters[idx] for idx, r in enumerate(g["readings"])}
            opt_lines = "\n".join(
                f"- {letters[idx]}: {ph.kata_to_hira(r['full_reading'])}（{r['sense']}）"
                for idx, r in enumerate(g["readings"])
            )
            valid_letters = {letters[idx] for idx in range(len(g["readings"]))}

            for b_start in range(0, len(mixed), batch_size):
                batch = mixed[b_start : b_start + batch_size]
                pairs = judge_batch_with_retry(
                    verify_model, g["surface"], opt_lines, batch, counts
                )

                for it, pred_char in pairs:
                    if it.get("is_anchor"):
                        continue
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

                    ud_map: Dict[tuple, str] = {}
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
                        "id": f"qwen3_pt_{len(verified_records):06d}",
                        "source": "qwen3_synth",
                        "license": "Apache-2.0",
                        "models": {"draft": verify_model, "verify": verify_model},
                        "text": text,
                        "spans": all_spans,
                        "target_key": key,
                        "target_span": target_sp_obj,
                        "category": it["category"],
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    verified_records.append(rec)

            if ki % 10 == 0 or ki == len(keys_list):
                fout.flush()
                print(
                    f"[Topup Phase B] keys {ki}/{len(keys_list)}, "
                    f"verified={len(verified_records)}, elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )

    val.close()
    print("Topup Phase B done:", json.dumps(dict(counts), ensure_ascii=False), flush=True)
    return verified_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3:14b")
    parser.add_argument("--min-verified", type=int, default=6)
    parser.add_argument("--verify-batch-size", type=int, default=10)
    parser.add_argument("--skip-phase-a", action="store_true")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    t0 = time.time()
    existing_cnt = count_existing_verified()
    groups = collect_topup_targets(existing_cnt, min_verified=args.min_verified)
    n_targets = sum(len(g["need_readings"]) for g in groups)
    n_zero = sum(1 for g in groups for r in g["need_readings"] if r["existing_verified"] == 0)
    print(
        f"Topup scope: {len(groups)} groups, {n_targets} targets (zero={n_zero}, 1..5={n_targets-n_zero})",
        flush=True,
    )

    blacklist = sqp.LSHBlacklist(
        [
            PILOT_DIR / "train.jsonl",
            PILOT_DIR / "dev.jsonl",
            PILOT_DIR / "homograph-eval.jsonl",
            RAW_DIR / "qwen3_drafts.jsonl",
            RAW_DIR / "qwen3_verified.jsonl",
            RAW_DIR / "qwen3_prod_drafts.jsonl",
            RAW_DIR / "qwen3_prod_verified.jsonl",
            RAW_DIR / "aozora_prod_clean.jsonl",
        ]
    )

    try:
        if not args.skip_phase_a:
            drafts = run_phase_a_topup(groups, blacklist, model_name=args.model)
        else:
            drafts = [json.loads(l) for l in open(TOPUP_DRAFTS_PATH, encoding="utf-8") if l.strip()]

        verified = run_phase_b_topup(
            drafts,
            verify_model=args.model,
            batch_size=args.verify_batch_size,
            seed=args.seed,
        )
    finally:
        sq.unload_ollama_model(args.model)

    by_kr = Counter((r["target_key"], r["target_span"]["reading_kata"]) for r in verified)
    stats = {
        "model": args.model,
        "target_groups": len(groups),
        "target_readings": n_targets,
        "total_drafts": len(drafts),
        "total_verified": len(verified),
        "pass_rate": round(len(verified) / max(1, len(drafts)), 4),
        "elapsed_sec": round(time.time() - t0, 2),
        "verified_by_target_top": [
            {"key": k, "reading": rd, "count": c}
            for (k, rd), c in by_kr.most_common()
        ],
    }
    with open(TOPUP_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(
        f"TOPUP COMPLETE | drafts={len(drafts)} | verified={len(verified)} | "
        f"pass_rate={stats['pass_rate']:.4f} | elapsed={stats['elapsed_sec']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
