#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 共通評価モジュール（モデル・辞書方式統一）。

特徴:
- 連続漢字ブロック連結正規化により、「報告書」vs「報告+書」のような形態素分割粒度差での
  不当減点を排除しつつ、文中の1文字でも読みが違えば即不正解とする厳密な文単位完全一致率を算出。
- 表記読み（ひらがな）、発音読み（UniDic準拠カタカナ）、ターゲット同形異音語単体正解率、
  ソース別内訳（実文 vs 合成）、誤答上位語・誤答サンプルを一括集計する。
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phonology as ph  # noqa: E402


def spans_to_kanji_blocks(spans: List[dict]) -> List[Tuple[int, int, str, str, str]]:
    """スパン列を連続漢字ブロック [(start, end, surface, reading_hira, pron)] に連結正規化する。"""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda x: (int(x["start"]), int(x["end"])))
    blocks: List[List] = []
    for sp in ordered:
        st = int(sp["start"])
        ed = int(sp["end"])
        surf = sp["surface"]
        r_hira = ph.kata_to_hira(ph.normalize_kana(sp["reading"]))
        pron = ph.normalize_kana(sp["pron"])
        if blocks and st == blocks[-1][1]:
            blocks[-1][1] = ed
            blocks[-1][2] += surf
            blocks[-1][3] += r_hira
            blocks[-1][4] += pron
        else:
            blocks.append([st, ed, surf, r_hira, pron])
    return [(b[0], b[1], b[2], b[3], b[4]) for b in blocks]


def extract_subspan_reading(
    pred_spans: List[dict], t_start: int, t_end: int
) -> Optional[Tuple[str, str]]:
    """予測スパン列から [t_start, t_end) に厳密一致する（または厳密に覆う連結）読みを抽出する。"""
    exact = [
        sp for sp in pred_spans if int(sp["start"]) == t_start and int(sp["end"]) == t_end
    ]
    if exact:
        sp = exact[0]
        return (
            ph.kata_to_hira(ph.normalize_kana(sp["reading"])),
            ph.normalize_kana(sp["pron"]),
        )
    # 辞書方式で「一+日」のように細分化されている場合は厳密に [t_start, t_end) をぴったり覆う連結を探す
    parts = sorted(
        [sp for sp in pred_spans if int(sp["start"]) >= t_start and int(sp["end"]) <= t_end],
        key=lambda x: int(x["start"]),
    )
    if (
        parts
        and int(parts[0]["start"]) == t_start
        and int(parts[-1]["end"]) == t_end
        and all(int(parts[i]["end"]) == int(parts[i + 1]["start"]) for i in range(len(parts) - 1))
    ):
        r = "".join(ph.kata_to_hira(ph.normalize_kana(p["reading"])) for p in parts)
        p = "".join(ph.normalize_kana(p["pron"]) for p in parts)
        return (r, p)
    return None


def evaluate_predictions(
    gold_rows: List[dict],
    pred_spans_list: List[List[dict]],
    eval_sec: float = 0.0,
) -> dict:
    """ゴールドデータと予測スパン列のリストから全評価指標を算出する。"""
    assert len(gold_rows) == len(pred_spans_list), "件数不一致"

    n_sent = len(gold_rows)
    total_chars = 0
    sent_r_ok = 0
    sent_p_ok = 0
    sent_both_ok = 0

    total_blocks = 0
    block_r_ok = 0
    block_p_ok = 0

    target_total = 0
    target_r_ok = 0
    target_p_ok = 0

    by_source: Dict[str, dict] = defaultdict(
        lambda: {
            "n": 0,
            "sent_r_ok": 0,
            "sent_p_ok": 0,
            "sent_both_ok": 0,
            "target_n": 0,
            "target_r_ok": 0,
        }
    )
    by_subset_type: Dict[str, dict] = defaultdict(
        lambda: {"n": 0, "sent_r_ok": 0, "target_n": 0, "target_r_ok": 0}
    )

    err_word_counter: Counter = Counter()
    err_pair_counter: Counter = Counter()
    err_samples: List[dict] = []

    for row, pred_spans in zip(gold_rows, pred_spans_list):
        text = row["text"]
        total_chars += len(text)
        src = row.get("source", "unknown")
        subset_type = "synth" if src == "qwen3_synth" else "real"

        g_blocks = spans_to_kanji_blocks(row["spans"])
        p_blocks = spans_to_kanji_blocks(pred_spans)

        # 位置と表面ごとの比較（通常、漢字ブロック構造が一致しない場合＝人名誤除外等も不正解として厳密に判定）
        g_key_r = [(b[0], b[1], b[3]) for b in g_blocks]
        p_key_r = [(b[0], b[1], b[3]) for b in p_blocks]
        g_key_p = [(b[0], b[1], b[4]) for b in g_blocks]
        p_key_p = [(b[0], b[1], b[4]) for b in p_blocks]

        r_match = g_key_r == p_key_r
        p_match = g_key_p == p_key_p

        if r_match:
            sent_r_ok += 1
        if p_match:
            sent_p_ok += 1
        if r_match and p_match:
            sent_both_ok += 1

        by_source[src]["n"] += 1
        if r_match:
            by_source[src]["sent_r_ok"] += 1
        if p_match:
            by_source[src]["sent_p_ok"] += 1
        if r_match and p_match:
            by_source[src]["sent_both_ok"] += 1

        by_subset_type[subset_type]["n"] += 1
        if r_match:
            by_subset_type[subset_type]["sent_r_ok"] += 1

        # ブロック単位精度
        p_block_map = {(b[0], b[1]): (b[3], b[4]) for b in p_blocks}
        for bs, be, bsurf, br, bp in g_blocks:
            total_blocks += 1
            pr_pair = p_block_map.get((bs, be))
            if pr_pair is not None:
                if pr_pair[0] == br:
                    block_r_ok += 1
                else:
                    err_word_counter[bsurf] += 1
                    err_pair_counter[(bsurf, br, pr_pair[0])] += 1
                if pr_pair[1] == bp:
                    block_p_ok += 1
            else:
                err_word_counter[bsurf] += 1
                err_pair_counter[(bsurf, br, "<MISSING>")] += 1

        # ターゲット同形異音語単体の判定（target_spans または target_span / explicit_homograph_targets）
        t_list = row.get("target_spans")
        if not t_list and row.get("target_span"):
            t_list = [row["target_span"]]
        if t_list:
            for tsp in t_list:
                target_total += 1
                by_source[src]["target_n"] += 1
                by_subset_type[subset_type]["target_n"] += 1
                g_tr = ph.kata_to_hira(ph.normalize_kana(tsp["reading"]))
                g_tp = ph.normalize_kana(tsp["pron"])
                got = extract_subspan_reading(pred_spans, int(tsp["start"]), int(tsp["end"]))
                if got is not None:
                    if got[0] == g_tr:
                        target_r_ok += 1
                        by_source[src]["target_r_ok"] += 1
                        by_subset_type[subset_type]["target_r_ok"] += 1
                    if got[1] == g_tp:
                        target_p_ok += 1

        if not r_match and len(err_samples) < 25:
            # 最初の不一致ブロックを記録
            diffs = []
            for bs, be, bsurf, br, _ in g_blocks:
                pr_pair = p_block_map.get((bs, be))
                pr_r = pr_pair[0] if pr_pair else "<MISSING>"
                if pr_r != br:
                    diffs.append({"surface": bsurf, "gold": br, "pred": pr_r})
            err_samples.append({"id": row.get("id"), "source": src, "text": text, "diffs": diffs})

    src_summary = {}
    for k, v in sorted(by_source.items()):
        n = max(1, v["n"])
        tn = v["target_n"]
        src_summary[k] = {
            "n": v["n"],
            "sentence_reading_acc": round(v["sent_r_ok"] / n, 4),
            "sentence_pron_acc": round(v["sent_p_ok"] / n, 4),
            "sentence_both_acc": round(v["sent_both_ok"] / n, 4),
            "target_n": tn,
            "target_homograph_acc": round(v["target_r_ok"] / tn, 4) if tn > 0 else None,
        }

    subset_summary = {}
    for k, v in sorted(by_subset_type.items()):
        n = max(1, v["n"])
        tn = v["target_n"]
        subset_summary[k] = {
            "n": v["n"],
            "sentence_reading_acc": round(v["sent_r_ok"] / n, 4),
            "target_n": tn,
            "target_homograph_acc": round(v["target_r_ok"] / tn, 4) if tn > 0 else None,
        }

    return {
        "n_sentences": n_sent,
        "total_chars": total_chars,
        "sentence_reading_acc": round(sent_r_ok / max(1, n_sent), 4),
        "sentence_pron_acc": round(sent_p_ok / max(1, n_sent), 4),
        "sentence_both_acc": round(sent_both_ok / max(1, n_sent), 4),
        "sentence_reading_correct": sent_r_ok,
        "total_blocks": total_blocks,
        "span_reading_acc": round(block_r_ok / max(1, total_blocks), 4),
        "span_pron_acc": round(block_p_ok / max(1, total_blocks), 4),
        "target_homograph_n": target_total,
        "target_homograph_acc": round(target_r_ok / target_total, 4) if target_total > 0 else None,
        "target_homograph_pron_acc": round(target_p_ok / target_total, 4) if target_total > 0 else None,
        "eval_sec": round(eval_sec, 3),
        "ms_per_sentence": round((eval_sec * 1000.0) / max(1, n_sent), 3) if eval_sec > 0 else None,
        "chars_per_sec": round(total_chars / eval_sec, 1) if eval_sec > 0 else None,
        "by_source": src_summary,
        "by_subset_type": subset_summary,
        "top_error_words": [
            {"surface": w, "errors": c} for w, c in err_word_counter.most_common(20)
        ],
        "top_error_pairs": [
            {"surface": w, "gold": g, "pred": p, "errors": c}
            for (w, g, p), c in err_pair_counter.most_common(25)
        ],
        "error_samples": err_samples,
    }
