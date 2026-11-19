#!/usr/bin/env python3
"""第7号 ja-address-ner-v1 第2段階 辞書リンカー評価スクリプト。

dev.jsonl 上で以下 3 モードを並列比較し、runs/linker_eval_dev.json を出力する：
  1. oracle_gold_components  : 正解 5 要素入力（リンカー単体検証）
  2. modernbert_plus_linker  : ModernBERT 予測分解 -> 辞書リンカー
  3. bert_v3_plus_linker     : BERT-v3 予測分解 -> 辞書リンカー

注意: 封印テスト（sealed-test.jsonl）は一切入力禁止。
"""
import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linker import AddressLinker  # noqa: E402


def _match_overlap_1to1(gold_addrs: List[dict], pred_addrs: List[dict]) -> Dict[int, int]:
    """正解住所インデックス -> 予測住所インデックス（重なり文字数最大の1対1貪欲対応）。"""
    pairs = []
    for gi, g in enumerate(gold_addrs):
        gs, ge = int(g["start"]), int(g["end"])
        for pi, p in enumerate(pred_addrs):
            ps, pe = int(p["start"]), int(p["end"])
            ov = min(ge, pe) - max(gs, ps)
            if ov > 0:
                pairs.append((ov, gi, pi))
    pairs.sort(key=lambda x: x[0], reverse=True)
    used_g, used_p = set(), set()
    mapping: Dict[int, int] = {}
    for _, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            mapping[gi] = pi
    return mapping


def _safe_rate(num: int, den: int) -> float:
    return round(num / den, 4) if den > 0 else 0.0


def evaluate_mode(
    dev_rows: List[dict],
    preds_by_id: Dict[str, List[dict]],
    linker: AddressLinker,
    is_oracle: bool = False,
) -> dict:
    """1 モード（Oracle または予測分解結果）に対するリンカー指標を一括算出する。"""
    res_total = res_correct = res_wrong_code = res_false_reject = res_missed = 0
    unres_total = unres_strict_reject = unres_missed = unres_false_assert = 0

    fp_unmatched_addrs = 0
    fp_unmatched_resolved = 0
    fp_unmatched_rejected = 0
    noaddr_fp_resolved = 0

    reason_counts = collections.Counter()
    by_cat: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    false_assertion_examples: List[dict] = []
    wrong_code_examples: List[dict] = []
    false_reject_examples: List[dict] = []

    for row in dev_rows:
        rid = row["id"]
        gold_addrs = row["addresses"]
        if is_oracle:
            pred_addrs = [
                {
                    "start": ga["start"],
                    "end": ga["end"],
                    "text": ga["text"],
                    "components": ga["components"],
                }
                for ga in gold_addrs
            ]
        else:
            pred_addrs = preds_by_id[rid]

        g2p = _match_overlap_1to1(gold_addrs, pred_addrs)
        matched_p = set(g2p.values())

        # 正解住所ごとの評価
        for gi, ga in enumerate(gold_addrs):
            cat = ga["category"]
            exp_status = ga["expected_status"]
            exp_code = ga["expected_code"]

            if gi not in g2p:
                if exp_status == "RESOLVED":
                    res_total += 1
                    res_missed += 1
                    by_cat[cat]["res_total"] += 1
                    by_cat[cat]["missed"] += 1
                else:
                    unres_total += 1
                    unres_missed += 1
                    by_cat[cat]["unres_total"] += 1
                    by_cat[cat]["missed"] += 1
                continue

            pa = pred_addrs[g2p[gi]]
            lres = linker.link_components(pa["components"])
            reason_counts[lres["reason"]] += 1

            if exp_status == "RESOLVED":
                res_total += 1
                by_cat[cat]["res_total"] += 1
                if lres["status"] == "RESOLVED":
                    if lres["municipality_code"] == exp_code:
                        res_correct += 1
                        by_cat[cat]["correct"] += 1
                    else:
                        res_wrong_code += 1
                        by_cat[cat]["wrong_code"] += 1
                        if len(wrong_code_examples) < 10:
                            wrong_code_examples.append(
                                {
                                    "id": rid,
                                    "category": cat,
                                    "gold_text": ga["text"],
                                    "gold_comps": ga["components"],
                                    "gold_code": exp_code,
                                    "pred_comps": pa["components"],
                                    "pred_code": lres["municipality_code"],
                                    "reason": lres["reason"],
                                }
                            )
                else:
                    res_false_reject += 1
                    by_cat[cat]["false_reject"] += 1
                    if len(false_reject_examples) < 10:
                        false_reject_examples.append(
                            {
                                "id": rid,
                                "category": cat,
                                "gold_text": ga["text"],
                                "gold_comps": ga["components"],
                                "gold_code": exp_code,
                                "pred_comps": pa["components"],
                                "reason": lres["reason"],
                            }
                        )
            else:  # UNRESOLVABLE
                unres_total += 1
                by_cat[cat]["unres_total"] += 1
                if lres["status"] == "UNRESOLVABLE":
                    unres_strict_reject += 1
                    by_cat[cat]["strict_reject"] += 1
                else:
                    unres_false_assert += 1
                    by_cat[cat]["false_assertion"] += 1
                    if len(false_assertion_examples) < 10:
                        false_assertion_examples.append(
                            {
                                "id": rid,
                                "category": cat,
                                "subtype": ga.get("subtype"),
                                "gold_text": ga["text"],
                                "gold_comps": ga["components"],
                                "pred_comps": pa["components"],
                                "pred_code": lres["municipality_code"],
                                "canonical": f"{lres['canonical_pref']}{lres['canonical_city']}",
                                "reason": lres["reason"],
                            }
                        )

        # 未対応の余分な予測住所（FP住所）に対するリンカー挙動
        for pi, pa in enumerate(pred_addrs):
            if pi in matched_p:
                continue
            fp_unmatched_addrs += 1
            lres = linker.link_components(pa["components"])
            if lres["status"] == "RESOLVED":
                fp_unmatched_resolved += 1
                if row["category"] == "no_address":
                    noaddr_fp_resolved += 1
            else:
                fp_unmatched_rejected += 1

    total_gold_addrs = res_total + unres_total
    res_extracted = res_total - res_missed
    unres_extracted = unres_total - unres_missed
    wrong_assertions_on_gold = res_wrong_code + unres_false_assert
    total_assertions_on_gold = res_correct + res_wrong_code + unres_false_assert
    total_assertions_e2e = total_assertions_on_gold + fp_unmatched_resolved

    cat_summary: Dict[str, dict] = {}
    for cat, cnt in sorted(by_cat.items()):
        rt = cnt["res_total"]
        ut = cnt["unres_total"]
        d: dict = {}
        if rt > 0:
            d.update(
                {
                    "resolvable_total": rt,
                    "correct": cnt["correct"],
                    "wrong_code": cnt["wrong_code"],
                    "false_reject": cnt["false_reject"],
                    "missed": cnt["missed"],
                    "accuracy": _safe_rate(cnt["correct"], rt),
                    "wrong_code_rate": _safe_rate(cnt["wrong_code"], rt),
                }
            )
        if ut > 0:
            d.update(
                {
                    "unresolvable_total": ut,
                    "strict_reject": cnt["strict_reject"],
                    "missed": cnt["missed"],
                    "false_assertion": cnt["false_assertion"],
                    "strict_rejection_rate": _safe_rate(cnt["strict_reject"], ut),
                    "non_assertion_rate": _safe_rate(cnt["strict_reject"] + cnt["missed"], ut),
                    "false_assertion_rate": _safe_rate(cnt["false_assertion"], ut),
                }
            )
        cat_summary[cat] = d

    return {
        "total_gold_addresses": total_gold_addrs,
        "resolvable": {
            "total": res_total,
            "correct": res_correct,
            "wrong_code": res_wrong_code,
            "false_reject": res_false_reject,
            "missed": res_missed,
            "accuracy": _safe_rate(res_correct, res_total),
            "accuracy_on_extracted": _safe_rate(res_correct, res_extracted),
            "wrong_code_rate": _safe_rate(res_wrong_code, res_total),
            "false_reject_rate": _safe_rate(res_false_reject, res_total),
            "missed_rate": _safe_rate(res_missed, res_total),
        },
        "unresolvable": {
            "total": unres_total,
            "strict_reject": unres_strict_reject,
            "missed": unres_missed,
            "false_assertion": unres_false_assert,
            "strict_rejection_rate": _safe_rate(unres_strict_reject, unres_total),
            "strict_rejection_rate_on_extracted": _safe_rate(
                unres_strict_reject, unres_extracted
            ),
            "non_assertion_rate": _safe_rate(
                unres_strict_reject + unres_missed, unres_total
            ),
            "false_assertion_rate": _safe_rate(unres_false_assert, unres_total),
            "false_assertion_rate_on_extracted": _safe_rate(
                unres_false_assert, unres_extracted
            ),
        },
        "assertion_summary": {
            "overall_wrong_assertion_rate": _safe_rate(
                wrong_assertions_on_gold, total_gold_addrs
            ),
            "assertion_precision_on_gold": _safe_rate(
                res_correct, total_assertions_on_gold
            ),
            "assertion_precision_end_to_end": _safe_rate(
                res_correct, total_assertions_e2e
            ),
            "fp_unmatched_pred_addrs": fp_unmatched_addrs,
            "fp_unmatched_resolved": fp_unmatched_resolved,
            "fp_unmatched_rejected": fp_unmatched_rejected,
            "no_address_fp_resolved": noaddr_fp_resolved,
        },
        "by_category": cat_summary,
        "reason_counts": dict(reason_counts.most_common()),
        "error_examples": {
            "false_assertion_examples": false_assertion_examples,
            "wrong_code_examples": wrong_code_examples,
            "false_reject_examples": false_reject_examples,
        },
    }


def load_preds(path: Path) -> Dict[str, List[dict]]:
    mp: Dict[str, List[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            mp[r["id"]] = r["pred_addresses"]
    return mp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--modernbert-preds", required=True)
    ap.add_argument("--bert-preds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for p_str in (args.dev, args.modernbert_preds, args.bert_preds):
        if "sealed" in Path(p_str).name.lower():
            raise RuntimeError(f"封印テストファイルは入力禁止です: {p_str}")

    dev_rows = [json.loads(l) for l in open(args.dev, encoding="utf-8")]
    mb_preds = load_preds(Path(args.modernbert_preds))
    bv_preds = load_preds(Path(args.bert_preds))
    linker = AddressLinker()

    oracle_res = evaluate_mode(dev_rows, {}, linker, is_oracle=True)
    # 自己整合確認：Oracle入力は100%正解・100%棄却・0%誤断言
    assert oracle_res["resolvable"]["accuracy"] == 1.0, oracle_res["resolvable"]
    assert oracle_res["unresolvable"]["strict_rejection_rate"] == 1.0, oracle_res["unresolvable"]
    assert oracle_res["unresolvable"]["false_assertion_rate"] == 0.0, oracle_res["unresolvable"]

    mb_res = evaluate_mode(dev_rows, mb_preds, linker, is_oracle=False)
    bv_res = evaluate_mode(dev_rows, bv_preds, linker, is_oracle=False)

    out_obj = {
        "dev_rows": len(dev_rows),
        "oracle_gold_components": oracle_res,
        "modernbert_plus_linker": mb_res,
        "bert_v3_plus_linker": bv_res,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {}
    for k in ("oracle_gold_components", "modernbert_plus_linker", "bert_v3_plus_linker"):
        d = out_obj[k]
        summary[k] = {
            "resolvable_acc": d["resolvable"]["accuracy"],
            "wrong_code_rate": d["resolvable"]["wrong_code_rate"],
            "false_reject_rate": d["resolvable"]["false_reject_rate"],
            "missed_rate": d["resolvable"]["missed_rate"],
            "unres_strict_reject": d["unresolvable"]["strict_rejection_rate"],
            "unres_non_assert": d["unresolvable"]["non_assertion_rate"],
            "unres_false_assert": d["unresolvable"]["false_assertion_rate"],
            "e2e_assert_prec": d["assertion_summary"]["assertion_precision_end_to_end"],
            "fp_resolved": d["assertion_summary"]["fp_unmatched_resolved"],
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
