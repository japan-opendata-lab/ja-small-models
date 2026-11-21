#!/usr/bin/env python3
"""第11号 ja-nli-v1 本番用 脱ショートカット・複数クラス対生成＆独立検証パイプライン。

第1工程パイロットで判明した仮説単体バイアス（「今後」「すべての」「動機」「〜ず、」「〜ため、〜し、」等）を
抑止し、同一前提からの複数クラス対生成（Pass 1 + 別観点 Pass 2）と厳格な強化規則フィルタ
（strict_demonstratives=True + filter_intra_chunk_hypotheses）を実行する。

主なモード:
  --phase all --smoke-chunks 1000 --out-dir out_smoke
      -> train プール層化 1,000 チャンク（Pass1+2 計約1,800プロンプト）で事前スモーク検証＋
         仮説単体 5-fold CV ショートカット検査を実行して smoke_report.json を出力。
  --phase gen / verify / all --out-dir out_prod
      -> サブプロセス分離により VRAM を 100% 解放しながら全チャンク生成・Gemma-4 ブラインド検証を実行。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nli_rules as nr  # noqa: E402

# 脱ショートカット向けサブタイプ定義（定型接続詞・ぼかし語・安易な否定を抑止）
SUBTYPES_E = [
    (
        "E1_plain_fact",
        0.25,
        "根拠中の一つの核心事実を単文の平叙文（「〜だ／〜である／〜する／〜している」）で述べる。「〜ため、」「〜し、」「〜により」の連結は避ける",
    ),
    (
        "E2_multi_sentence",
        0.20,
        "根拠中の複数文の情報を統合し、無理に長い連結文にせず簡潔な事実平叙文1文で結論を述べる",
    ),
    (
        "E3_numeric_order",
        0.15,
        "根拠中の数値・時期・範囲・順序・大小比較から論理的に必ず成り立つ関係を具体的な平叙文1文で述べる",
    ),
    (
        "E4_negation_preserve",
        0.25,
        "自然な否定表現（「〜ない」「〜ではない」「〜しない」「対象外である」「除かれる」）を含みつつ、根拠と完全に一致する事実を1文で述べる",
    ),
    (
        "E5_modal_scope_preserve",
        0.15,
        "根拠自体が述べる一般性・見通し・条件範囲を歪めず言い換え、根拠のみから確実に真と言える事実を1文で述べる",
    ),
]

SUBTYPES_N = [
    (
        "N1_concrete_detail",
        0.35,
        "根拠に一切登場しない具体的な日時・地域・対象範囲・金額・件数等の別詳細を必ず1つ補い、通常の断定平叙文1文にする（推測語「今後」「可能性」や全称「すべて」は禁止）",
    ),
    (
        "N2_plausible_comparison",
        0.25,
        "根拠に記載のない別年度・別統計・関連領域の具体的水準や追加要件を通常の断定平叙文1文で述べる（「他の〜」「同様」「異なる」等の決まり文句に偏らず、既述事実の単なる言い換えにしない）",
    ),
    (
        "N3_unstated_condition_or_cause",
        0.20,
        "根拠に書かれていない実施要件・運用細目・具体的背景を1つ補った通常の断定平叙文1文にする（既に根拠に書かれた理由を繰り返さず、「動機」「意図」「期待」「可能性」は禁止）",
    ),
    (
        "N4_missing_detail_neg",
        0.20,
        "否定表現（「〜ない」「〜しなかった」「〜ではない」）を含みつつ、根拠に記述がない別側面の未確認事項を平叙文1文で述べ、真とも偽とも言えない中立文にする",
    ),
]

SUBTYPES_C = [
    (
        "C1_numeric_quant",
        0.25,
        "否定語に頼らず、根拠中の数値・割合・時期・件数・順序を両立不能な別の具体的数値へ置き換えた肯定形の矛盾文を1文で作る",
    ),
    (
        "C2_entity_role",
        0.25,
        "否定語に頼らず、主語・実施主体・対象・因果や授受の方向を入れ替えた肯定形の矛盾文を1文で作る",
    ),
    (
        "C3_modality_exception",
        0.20,
        "義務と任意、禁止と許可、原則と例外を入れ替えた矛盾文を1文で作る（「不要」「一切」「無視」などの決まり文句は禁止）",
    ),
    (
        "C4_antonym_state",
        0.20,
        "「〜ない」の追加を避け、増加↔減少・上昇↔低下・維持↔廃止・導入↔撤回などの明確な反意状態へ変えた肯定形の矛盾文を1文で作る",
    ),
    (
        "C5_negation_scope",
        0.10,
        "根拠の核心主張を自然な日本語の否定文で明確に否定する矛盾文を1文で作る（「一切」「完全に無視」などの強い誇張語は使わない）",
    ),
]


def _pick_weighted(
    rng: random.Random,
    specs: List[Tuple[str, float, str]],
    exclude_codes: Optional[set] = None,
) -> Tuple[str, str]:
    ex = exclude_codes or set()
    filtered = [(c, w, d) for c, w, d in specs if c not in ex]
    if not filtered:
        filtered = list(specs)
    tot = sum(w for _, w, _ in filtered)
    r = rng.random() * tot
    acc = 0.0
    for code, w, desc in filtered:
        acc += w
        if r <= acc:
            return code, desc
    return filtered[-1][0], filtered[-1][2]


def assign_round_subtypes(
    seed_idx: int,
    round_id: int,
    seed: int = 42,
    prev_list: Optional[List[Tuple[str, str, str]]] = None,
) -> Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]]:
    rng = random.Random(seed * 1000003 + seed_idx * 37 + round_id)
    ex_e = {p[0] for p in prev_list} if prev_list else set()
    ex_n = {p[1] for p in prev_list} if prev_list else set()
    ex_c = {p[2] for p in prev_list} if prev_list else set()
    return (
        _pick_weighted(rng, SUBTYPES_E, exclude_codes=ex_e),
        _pick_weighted(rng, SUBTYPES_N, exclude_codes=ex_n),
        _pick_weighted(rng, SUBTYPES_C, exclude_codes=ex_c),
    )


def build_gen_messages(
    premise: str,
    sub_e: Tuple[str, str],
    sub_n: Tuple[str, str],
    sub_c: Tuple[str, str],
    round_id: int,
) -> List[dict]:
    if round_id == 1:
        round_note = ""
    elif round_id == 2:
        round_note = "\n- 【複数観点生成（第2パス）】第1観点と重ならない別の文・数値・条件に着目し、同一根拠から別観点の主張文3本を作ること。"
    else:
        round_note = "\n- 【複数観点生成（第3パス）】これまでの観点と異なる第3の詳細・例外・付随要件に着目し、既出文と言い換えにならない新たな主張文3本を作ること。"
    sys_msg = (
        "あなたは日本語自然言語推論（NLI）データセットの精密な作成者です。"
        "根拠テキストだけに基づいて、3クラス（entailment / neutral / contradiction）の仮説文を各1文ずつ作成してください。"
        "仮説文の文体や語彙から答えが推測されないよう、3文とも同じ客観的・標準的な日本語の断定平叙文として書いてください。"
        "出力は有効なJSONオブジェクト1個のみとし、説明文やコードブロック記号は出さないでください。"
    )
    usr_msg = f"""以下の根拠テキストを読み、3種類の関係に対応する仮説文を各1文作成してください。

【脱ショートカット・品質厳守ルール】
1. 3文とも30〜75文字程度の単体で意味が通じる自然な日本語1文（「。」で終える、常体）にし、長さや文体を揃えること。
2. メタ言及（「前提」「仮説」「含意」「矛盾」「中立」「本文」「文章では」「この文書」「上記」「以下」「述べられている」「記載されている」等）は一切禁止。
3. 文頭および文中に、指す対象が根拠外にある曖昧な指示語（「これにより」「このレッスン」「このプログラム」「この法律」「この機能」「これらの演算」「本事業」等）を残さず、必ず固有の名称・対象を明記すること。
4. クラス特有の決まり文句を禁止：
   - entailment だからといって「〜ため、〜し、〜により」と連結しすぎない。
   - neutral に「今後」「将来」「可能性がある」「かもしれない」「すべて」「あらゆる」「動機」「意図」「期待される」「とされている」を使わない（普通の事実陳述文の形式で、根拠に書かれていない情報だけを入れる）。
   - contradiction に「一切」「無視」「不要で」「全く」等の誇張語を安易に使わない。
5. 根拠テキストの長い箇所をそのまま丸コピーしないこと。{round_note}

【今回の推論タイプ指定】
- entailment（含意）[{sub_e[0]}]: {sub_e[1]}
- neutral（中立）[{sub_n[0]}]: {sub_n[1]}
- contradiction（矛盾）[{sub_c[0]}]: {sub_c[1]}

【根拠テキスト】
{premise}

【出力形式（JSONのみ）】
{{"entailment": "...", "neutral": "...", "contradiction": "..."}}"""
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}]


def build_verify_messages(premise: str, hypothesis: str) -> List[dict]:
    sys_msg = (
        "あなたは日本語の論理関係および文章自然性の厳格な独立審査官です。"
        "根拠テキストのみを事実とみなし、主張文との論理関係を判定してください。"
        "知識の補完や好意的な推測はせず、根拠から厳密に判断し、JSONオブジェクト1個のみを出力してください。"
    )
    usr_msg = f"""以下の「根拠テキスト」に対し、「主張文」の論理関係と日本語としての自然さを評価してください。

【判定ラベル定義】
- entailment: 根拠テキストの内容だけから主張文が確実に成り立つ（背景知識を足さずに論理的に含意される）。
- contradiction: 主張文は根拠テキストと論理的に両立せず、確実に誤りとなる。
- neutral: 根拠テキストに書かれていない情報を含む、または条件が欠けており、真とも偽とも断定できない。

【自然さ (naturalness) 1〜5】
5: 完全に自然な独立した日本語文（単体で指示対象が明確）
4: 自然で問題ない
3: やや不自然・翻訳調・指示語「この〜」等で文脈不足
2: 明らかに不自然またはメタ言及がある
1: 意味不明・破綻・文未完

【曖昧さ (ambiguity)】
- clear: 論理関係の解釈に迷いがない
- ambiguous: 解釈によってラベルが揺れうる、または文脈依存で判定困難

【根拠テキスト】
{premise}

【主張文】
{hypothesis}

【出力形式（JSONのみ）】
{{"label": "entailment|neutral|contradiction", "naturalness": 1, "ambiguity": "clear|ambiguous", "reason": "30〜60字の判定根拠"}}"""
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}]


def parse_json_obj(raw: str) -> Optional[dict]:
    if not raw:
        return None
    txt = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    st = txt.find("{")
    ed = txt.rfind("}")
    if st < 0 or ed <= st:
        return None
    sub = txt[st : ed + 1]
    try:
        obj = json.loads(sub)
        return obj if isinstance(obj, dict) else None
    except Exception:
        sub2 = re.sub(r",\s*}", "}", sub)
        try:
            obj = json.loads(sub2)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def apply_chat_template_safe(tokenizer, messages: List[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _init_vllm_llm(model_id: str, max_model_len: int = 3072):
    from vllm import LLM

    base_kw = dict(
        model=model_id,
        dtype="auto",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        trust_remote_code=False,
        disable_log_stats=True,
    )
    for extra in (
        {"enable_prefix_caching": True, "limit_mm_per_prompt": {"image": 0}},
        {"enable_prefix_caching": True},
        {},
    ):
        try:
            return LLM(**base_kw, **extra)
        except TypeError:
            continue
    return LLM(**base_kw)


def build_generation_tasks(chunks: List[dict], args) -> List[dict]:
    """Pass 1 (全チャンク) と Pass 2 / Pass 3 (別観点複数パス) の生成タスクを組む。"""
    tasks: List[dict] = []
    hist_subs: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for c in chunks:
        se, sn, sc = assign_round_subtypes(c["seed_idx"], round_id=1)
        tasks.append({"chunk": c, "round": 1, "subs": (se, sn, sc)})
        hist_subs[c["chunk_id"]].append((se[0], sn[0], sc[0]))

    target_by_pool = {
        "train": args.target_train_prompts,
        "dev": args.target_dev_prompts,
        "sealed_test": args.target_sealed_prompts,
    }
    by_pool: Dict[str, List[dict]] = defaultdict(list)
    for c in chunks:
        by_pool[c["split_pool"]].append(c)

    rng = random.Random(42)
    for pool, plist in sorted(by_pool.items()):
        p1_cnt = len(plist)
        target = p1_cnt if args.smoke_chunks > 0 else target_by_pool.get(pool, p1_cnt)
        if args.smoke_chunks > 0:
            need_p2 = int(p1_cnt * args.smoke_pass2_ratio)
            need_p3 = 0
        else:
            rem = max(0, target - p1_cnt)
            elig_p2 = [c for c in plist if len(c["premise"]) >= args.pass2_min_chars]
            rng.shuffle(elig_p2)
            chosen_p2 = elig_p2[:rem]
            need_p2 = len(chosen_p2)
            rem_after_p2 = max(0, rem - need_p2)
            elig_p3 = [c for c in plist if len(c["premise"]) >= args.pass3_min_chars]
            rng.shuffle(elig_p3)
            chosen_p3 = elig_p3[:rem_after_p2]
            need_p3 = len(chosen_p3)

        if args.smoke_chunks > 0:
            elig_p2 = [c for c in plist if len(c["premise"]) >= args.pass2_min_chars]
            rng.shuffle(elig_p2)
            chosen_p2 = elig_p2[:need_p2]
            chosen_p3 = []

        for c in chosen_p2:
            se2, sn2, sc2 = assign_round_subtypes(
                c["seed_idx"], round_id=2, prev_list=hist_subs[c["chunk_id"]]
            )
            tasks.append({"chunk": c, "round": 2, "subs": (se2, sn2, sc2)})
            hist_subs[c["chunk_id"]].append((se2[0], sn2[0], sc2[0]))

        for c in chosen_p3:
            se3, sn3, sc3 = assign_round_subtypes(
                c["seed_idx"], round_id=3, prev_list=hist_subs[c["chunk_id"]]
            )
            tasks.append({"chunk": c, "round": 3, "subs": (se3, sn3, sc3)})
            hist_subs[c["chunk_id"]].append((se3[0], sn3[0], sc3[0]))

        print(
            f"  [PLAN] pool={pool}: p1={p1_cnt}, p2={len(chosen_p2)}, p3={len(chosen_p3)}, "
            f"total_prompts={p1_cnt + len(chosen_p2) + len(chosen_p3)}"
        )
    return tasks


def load_target_chunks(args) -> List[dict]:
    with Path(args.chunks).open("r", encoding="utf-8") as f:
        chunks = [json.loads(ln) for ln in f if ln.strip()]
    if args.pool_filter:
        pools = set(args.pool_filter.split(","))
        chunks = [c for c in chunks if c["split_pool"] in pools]
    if args.smoke_chunks > 0:
        # smoke 時は train プールからドメイン×長さ帯で層化抽出
        tr_chunks = [c for c in chunks if c["split_pool"] == "train"]
        rng = random.Random(42)
        by_key = defaultdict(list)
        for c in tr_chunks:
            by_key[(c["domain_group"], c["len_bucket"])].append(c)
        picked = []
        keys = sorted(by_key.keys())
        for k in keys:
            rng.shuffle(by_key[k])
        idx = 0
        while len(picked) < args.smoke_chunks and any(by_key.values()):
            k = keys[idx % len(keys)]
            if by_key[k]:
                picked.append(by_key[k].pop())
            idx += 1
        return picked
    return chunks


def phase_gen(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_raw = out_dir / "synth_raw.jsonl"

    chunks = load_target_chunks(args)
    tasks_all = build_generation_tasks(chunks, args)

    # レジューム機能: (chunk_id, round) 単位で済を除外、既存ルール通過ペアを保持
    done_keys = set()
    kept_by_cid: Dict[str, List[dict]] = defaultdict(list)
    if out_raw.exists() and not args.overwrite:
        with out_raw.open("r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    row = json.loads(ln)
                    done_keys.add((row["chunk_id"], row.get("round", 1)))
                    kept_by_cid[row["chunk_id"]].append(row)

    todo = [t for t in tasks_all if (t["chunk"]["chunk_id"], t["round"]) not in done_keys]
    print(f"[GEN] total_prompts={len(tasks_all)}, already_done_prompts~={len(done_keys)//3}, todo_prompts={len(todo)}")
    if not todo:
        return

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    model_used = args.gen_model
    tok = None
    llm = None
    for cand in [args.gen_model, args.gen_fallback]:
        if not cand:
            continue
        try:
            tok = AutoTokenizer.from_pretrained(cand)
            llm = _init_vllm_llm(cand, max_model_len=3072)
            model_used = cand
            break
        except Exception as e:
            print(f"[WARN] vLLM init failed for {cand}: {e}")
    if llm is None or tok is None:
        raise RuntimeError("Generator vLLM init failed")

    prompts_all = [
        apply_chat_template_safe(
            tok,
            build_gen_messages(
                t["chunk"]["premise"],
                t["subs"][0],
                t["subs"][1],
                t["subs"][2],
                t["round"],
            ),
        )
        for t in todo
    ]
    sp = SamplingParams(temperature=0.70, top_p=0.92, max_tokens=420)

    batch_sz = 1000
    mode = "w" if (args.overwrite or not out_raw.exists()) else "a"
    stats = Counter()
    with out_raw.open(mode, encoding="utf-8") as fout:
        for st in range(0, len(todo), batch_sz):
            ed = min(st + batch_sz, len(todo))
            t_b = time.time()
            outs = llm.generate(prompts_all[st:ed], sp)
            for t, o in zip(todo[st:ed], outs):
                c = t["chunk"]
                rd = t["round"]
                sub_map = {
                    "entailment": t["subs"][0][0],
                    "neutral": t["subs"][1][0],
                    "contradiction": t["subs"][2][0],
                }
                raw_txt = o.outputs[0].text if o.outputs else ""
                obj = parse_json_obj(raw_txt)
                stats["prompts_processed"] += 1
                stats[f"prompts_round_{rd}"] += 1
                if not obj:
                    stats["json_parse_fail"] += 1
                    continue
                stats["json_parse_ok"] += 1
                cands = []
                for lbl in nr.VALID_LABELS:
                    hyp = obj.get(lbl)
                    if not isinstance(hyp, str):
                        stats[f"missing_{lbl}"] += 1
                        continue
                    hyp = nr.clean_surface(hyp)
                    ok, reasons = nr.check_pair_rules(
                        c["premise"], hyp, lbl, strict_demonstratives=True
                    )
                    if not ok:
                        for r in reasons:
                            stats[f"rule_reject_{r}"] += 1
                        stats["rule_reject_pairs"] += 1
                        continue
                    cands.append({
                        "pair_id": f"{c['chunk_id']}::r{rd}::{lbl[0]}",
                        "chunk_id": c["chunk_id"],
                        "round": rd,
                        "doc_id": c["doc_id"],
                        "cluster_id": c["cluster_id"],
                        "split_pool": c["split_pool"],
                        "source": c["source"],
                        "domain_group": c["domain_group"],
                        "license": c["license"],
                        "len_bucket": c["len_bucket"],
                        "premise": c["premise"],
                        "hypothesis": hyp,
                        "intended_label": lbl,
                        "subtype": sub_map[lbl],
                        "generator_model": model_used,
                    })

                # 同一チャンクの既存(Pass1含む)候補と合算して重複・同クラス近似・異クラス過度近似を検査
                prev_for_cid = kept_by_cid[c["chunk_id"]]
                combined = prev_for_cid + cands
                kept_all, drop_n = nr.filter_intra_chunk_hypotheses(
                    combined, same_class_thresh=0.65, cross_class_thresh=0.88
                )
                # prev_for_cid 側にあった要素を除いた新規のみ書き出す
                prev_ids = {x["pair_id"] for x in prev_for_cid}
                newly_kept = [x for x in kept_all if x["pair_id"] not in prev_ids]
                kept_by_cid[c["chunk_id"]].extend(newly_kept)
                stats["intra_chunk_dropped"] += drop_n
                for row in newly_kept:
                    stats["rule_passed_pairs"] += 1
                    stats[f"rule_passed_{row['intended_label']}"] += 1
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            print(
                f"[GEN] {ed}/{len(todo)} prompts ({time.time()-t_b:.1f}s) | "
                f"passed_pairs={stats['rule_passed_pairs']}"
            )

    (out_dir / "gen_stats.json").write_text(
        json.dumps({"generator_model": model_used, "counts": dict(stats)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_verifier_vllm_or_tf(model_candidates: List[str], backend: str = "auto"):
    from transformers import AutoTokenizer

    cands = [m for m in model_candidates if m]
    if backend in ("auto", "vllm"):
        for mid in cands:
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                llm = _init_vllm_llm(mid, max_model_len=3072)
                return ("vllm", mid, tok, llm)
            except Exception as e:
                print(f"[WARN] vLLM init failed for verifier {mid}: {e}")
    if backend in ("auto", "transformers"):
        import torch
        from transformers import AutoModelForCausalLM

        for mid in cands:
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                tok.padding_side = "left"
                mdl = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="auto")
                mdl.eval()
                return ("transformers", mid, tok, mdl)
            except Exception as e:
                print(f"[WARN] transformers init failed for verifier {mid}: {e}")
    raise RuntimeError(f"All verifier candidates failed: {cands}")


def _infer_verifier_batch(engine_tuple, msgs_batch: List[List[dict]]) -> List[str]:
    mode, _mid, tok, eng = engine_tuple
    prompts = [apply_chat_template_safe(tok, m) for m in msgs_batch]
    if mode == "vllm":
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, max_tokens=180)
        outs = eng.generate(prompts, sp)
        return [o.outputs[0].text if o.outputs else "" for o in outs]
    else:
        import torch

        results = []
        bs = 16
        for st in range(0, len(prompts), bs):
            b = prompts[st : st + bs]
            enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=3584).to(eng.device)
            in_len = enc["input_ids"].shape[1]
            with torch.inference_mode():
                gen = eng.generate(**enc, max_new_tokens=180, do_sample=False, pad_token_id=tok.pad_token_id)
            for row in gen:
                results.append(tok.decode(row[in_len:], skip_special_tokens=True))
        return results


def phase_verify(args) -> None:
    out_dir = Path(args.out_dir)
    raw_path = out_dir / "synth_raw.jsonl"
    ver_path = out_dir / "synth_verified.jsonl"
    log_path = out_dir / "synth_verify_log.jsonl"

    with raw_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    done_pids = set()
    if log_path.exists() and not args.overwrite:
        with log_path.open("r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    done_pids.add(json.loads(ln)["pair_id"])
    todo = [r for r in rows if r["pair_id"] not in done_pids]
    # 同一 premise のペアを隣接させて vLLM prefix caching 効率を最大化
    todo.sort(key=lambda r: (r["premise"], r["pair_id"]))
    print(f"[VERIFY] total_raw={len(rows)}, already_done={len(done_pids)}, todo={len(todo)}")

    cands = [args.verifier_model] + ([args.verifier_fallback] if args.verifier_fallback else [])
    if todo:
        eng = _load_verifier_vllm_or_tf(cands, backend=args.verifier_backend)
        model_used = eng[1]
        print(f"[VERIFY] backend={eng[0]}, model={model_used}")
        batch_sz = 2000
        mode_log = "w" if (args.overwrite or not log_path.exists()) else "a"
        mode_ver = "w" if (args.overwrite or not ver_path.exists()) else "a"
        with log_path.open(mode_log, encoding="utf-8") as flog, ver_path.open(mode_ver, encoding="utf-8") as fver:
            for st in range(0, len(todo), batch_sz):
                ed = min(st + batch_sz, len(todo))
                t_b = time.time()
                msgs = [build_verify_messages(r["premise"], r["hypothesis"]) for r in todo[st:ed]]
                outs = _infer_verifier_batch(eng, msgs)
                accepted_b = 0
                for r, raw_txt in zip(todo[st:ed], outs):
                    obj = parse_json_obj(raw_txt)
                    v_lbl = obj.get("label") if obj else None
                    v_nat = obj.get("naturalness") if obj else None
                    v_amb = obj.get("ambiguity") if obj else None
                    v_rsn = str(obj.get("reason", ""))[:120] if obj else ""
                    try:
                        v_nat_int = int(v_nat) if v_nat is not None else 0
                    except Exception:
                        v_nat_int = 0
                    label_match = (v_lbl == r["intended_label"])
                    accepted = (
                        v_lbl in nr.VALID_LABELS
                        and label_match
                        and v_nat_int >= 4
                        and v_amb == "clear"
                    )
                    rec = {
                        **r,
                        "verifier_model": model_used,
                        "verifier_label": v_lbl,
                        "verifier_naturalness": v_nat_int,
                        "verifier_ambiguity": v_amb,
                        "verifier_reason": v_rsn,
                        "verifier_agree": bool(label_match),
                        "accepted": bool(accepted),
                    }
                    flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if accepted:
                        accepted_b += 1
                        fver.write(json.dumps(rec, ensure_ascii=False) + "\n")
                flog.flush()
                fver.flush()
                print(f"[VERIFY] {ed}/{len(todo)} done ({time.time()-t_b:.1f}s) | accepted_in_batch={accepted_b}/{ed-st}")

    # 全体統計集計
    vstats = Counter()
    by_cls = defaultdict(Counter)
    by_pool = Counter()
    with log_path.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            rec = json.loads(ln)
            lbl = rec["intended_label"]
            vstats["total_verified_input"] += 1
            if rec["verifier_label"] in nr.VALID_LABELS:
                vstats["verifier_json_ok"] += 1
            else:
                vstats["verifier_json_fail"] += 1
            if rec["verifier_agree"]:
                vstats["label_agree"] += 1
                by_cls[lbl]["agree"] += 1
            by_cls[lbl]["total"] += 1
            vstats[f"nat_{rec['verifier_naturalness']}"] += 1
            vstats[f"amb_{rec['verifier_ambiguity']}"] += 1
            if rec["accepted"]:
                vstats["accepted_total"] += 1
                by_cls[lbl]["accepted"] += 1
                by_pool[f"{rec['split_pool']}::{lbl}"] += 1

    gen_stat_path = out_dir / "gen_stats.json"
    gen_stats = json.loads(gen_stat_path.read_text(encoding="utf-8")) if gen_stat_path.exists() else {}
    summary = {
        "gen": gen_stats,
        "verify": {
            "counts": dict(vstats),
            "by_class": {k: dict(v) for k, v in by_cls.items()},
            "accepted_by_pool_class": dict(sorted(by_pool.items())),
        },
    }
    (out_dir / "synth_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[VERIFY SUMMARY] total={vstats['total_verified_input']}, agree={vstats['label_agree']}, accepted={vstats['accepted_total']}")
    print("  by_class:", {k: dict(v) for k, v in by_cls.items()})
    print("  accepted_by_pool_class:", dict(sorted(by_pool.items())))


def run_smoke_shortcut_probe(out_dir: Path) -> None:
    """スモーク検証済みデータに対し仮説単体 5-fold CV プローブを実行してショートカット度を検査する。"""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold

    ver_path = out_dir / "synth_verified.jsonl"
    with ver_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    lbl2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    hyps = [r["hypothesis"] for r in rows]
    y = np.array([lbl2id[r["intended_label"]] for r in rows], dtype=np.int64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(rows), dtype=np.int64)
    for tr_i, va_i in skf.split(hyps, y):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=30000, sublinear_tf=True)
        x_tr = vec.fit_transform([hyps[i] for i in tr_i])
        x_va = vec.transform([hyps[i] for i in va_i])
        clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
        clf.fit(x_tr, y[tr_i])
        oof[va_i] = clf.predict(x_va)

    acc = float(accuracy_score(y, oof))
    mf1 = float(f1_score(y, oof, average="macro"))

    full_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=30000, sublinear_tf=True)
    x_full = full_vec.fit_transform(hyps)
    full_clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    full_clf.fit(x_full, y)
    fnames = np.array(full_vec.get_feature_names_out())
    top_feats = {}
    for cid, lbl in enumerate(["entailment", "neutral", "contradiction"]):
        w = full_clf.coef_[cid]
        ids = np.argsort(w)[::-1][:10]
        top_feats[lbl] = [{"ngram": str(fnames[j]), "weight": round(float(w[j]), 3)} for j in ids]

    stats = json.loads((out_dir / "synth_stats.json").read_text(encoding="utf-8"))
    tot_in = stats["verify"]["counts"].get("total_verified_input", 0)
    tot_acc = stats["verify"]["counts"].get("accepted_total", 0)
    n_info = stats["verify"]["by_class"].get("neutral", {})
    n_rate = n_info.get("accepted", 0) / max(1, n_info.get("total", 1))
    ov_rate = tot_acc / max(1, tot_in)

    report = {
        "verified_rows": len(rows),
        "overall_accept_rate": round(ov_rate, 4),
        "neutral_accept_rate": round(n_rate, 4),
        "hyp_only_5cv_accuracy": round(acc, 4),
        "hyp_only_5cv_macro_f1": round(mf1, 4),
        "top_positive_features": top_feats,
    }
    (out_dir / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[SMOKE DIAGNOSTIC]", json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gen", "verify", "all"], required=True)
    ap.add_argument("--chunks", default="chunks_prod.jsonl")
    ap.add_argument("--out-dir", default="out_prod")
    ap.add_argument("--smoke-chunks", type=int, default=0)
    ap.add_argument("--smoke-pass2-ratio", type=float, default=0.80)
    ap.add_argument("--target-train-prompts", type=int, default=82000)
    ap.add_argument("--target-dev-prompts", type=int, default=5500)
    ap.add_argument("--target-sealed-prompts", type=int, default=5500)
    ap.add_argument("--pass2-min-chars", type=int, default=140)
    ap.add_argument("--pass3-min-chars", type=int, default=240)
    ap.add_argument("--pool-filter", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--gen-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--gen-fallback", default="Qwen/Qwen3-14B")
    ap.add_argument("--verifier-model", default="google/gemma-4-26b-a4b-it")
    ap.add_argument("--verifier-fallback", default="google/gemma-4-12b-it")
    ap.add_argument("--verifier-backend", default="auto", choices=["auto", "vllm", "transformers"])
    args = ap.parse_args()

    if args.phase == "gen":
        phase_gen(args)
    elif args.phase == "verify":
        phase_verify(args)
    elif args.phase == "all":
        script = str(Path(__file__).resolve())
        flags = [
            "--chunks", args.chunks,
            "--out-dir", args.out_dir,
            "--smoke-chunks", str(args.smoke_chunks),
            "--smoke-pass2-ratio", str(args.smoke_pass2_ratio),
            "--target-train-prompts", str(args.target_train_prompts),
            "--target-dev-prompts", str(args.target_dev_prompts),
            "--target-sealed-prompts", str(args.target_sealed_prompts),
            "--pass2-min-chars", str(args.pass2_min_chars),
            "--pass3-min-chars", str(args.pass3_min_chars),
            "--gen-model", args.gen_model,
            "--gen-fallback", args.gen_fallback,
            "--verifier-model", args.verifier_model,
            "--verifier-fallback", args.verifier_fallback,
            "--verifier-backend", args.verifier_backend,
        ]
        if args.pool_filter:
            flags.extend(["--pool-filter", args.pool_filter])
        if args.overwrite:
            flags.append("--overwrite")

        for ph in ["gen", "verify"]:
            print(f"=== Launching subprocess phase: {ph} ===")
            subprocess.run([sys.executable, script, "--phase", ph] + flags, check=True)
        if args.smoke_chunks > 0:
            run_smoke_shortcut_probe(Path(args.out_dir))


if __name__ == "__main__":
    main()
