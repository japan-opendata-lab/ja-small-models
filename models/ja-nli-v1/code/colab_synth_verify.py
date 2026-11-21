#!/usr/bin/env python
"""第11号 ja-nli-v1 Colab 80GB GPU 用 合成＆独立検証パイプライン。

サブプロセス分離可能なフェーズ実行（VRAM 完全解放設計）:
  --phase gen     : 生成役 (Qwen3-30B-A3B-Instruct-2507, Apache 2.0) で E/N/C 3仮説を生成 + 規則フィルタ適用
  --phase verify  : 独立検証役 (Gemma-4-26B-A4B-IT 等, Apache 2.0) で完全ブラインド照合
  --phase verify3 : human_eval プール専用の第3検証役照合 (Mistral-Small-3.2 / Qwen3-14B 等, Apache 2.0)
  --phase all     : サブプロセス経由で gen -> verify (-> verify3) を連続実行し統計 JSON を出力
"""
import argparse
import gc
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nli_rules as nr  # noqa: E402

# サブタイプ定義（ショートカット防止）
SUBTYPES_E = [
    ("E1_paraphrase", 0.35, "語彙や構文を言い換えつつ、前提文から確実に成り立つ事実を1文で述べる（単純な単語置換や抜き出しにしない）"),
    ("E2_multi_sentence", 0.25, "前提文の複数箇所の情報を統合・要約して確実に言える結論を1文で述べる"),
    ("E3_numeric_order", 0.20, "前提文中の数値・時期・範囲・包含関係から論理的に必ず成り立つ比較や帰結を1文で述べる"),
    ("E4_negation_preserve", 0.20, "否定語（「〜ない」「〜ではない」「除き」等）を含みつつも、前提文の意味と完全に一致する含意文を1文で作る"),
]
SUBTYPES_N = [
    ("N1_unstated_cause", 0.25, "前提文と同じ主題を扱いながら、前提には書かれていない原因・動機・目的・背景を付け足し、真偽が判断できない中立文を1文で作る"),
    ("N2_overgeneralization", 0.25, "前提文の限定条件（「一部」「特定の」「場合により」等）を外して一般化し、前提だけでは真偽を断定できない中立文を1文で作る"),
    ("N3_future_or_effect", 0.25, "前提文の内容から連想されるものの、前提には記載のない将来予測・評価・副次効果を述べ、真偽不明の中立文を1文で作る"),
    ("N4_missing_detail_neg", 0.25, "否定語や具体的数値を含みつつも、前提文に記述がない別側面・比較対象を述べ、矛盾も含意もしない中立文を1文で作る"),
]
SUBTYPES_C = [
    ("C1_numeric_quant", 0.25, "前提文中の数値・割合・期間・順序・以上/未満などを論理的に両立不能な値へ変更した矛盾文を1文で作る"),
    ("C2_entity_role", 0.22, "主語・対象・因果関係・実施主体・方向性を入れ替え、前提文と両立しない矛盾文を1文で作る"),
    ("C3_modality_exception", 0.20, "義務/任意、必須/不要、禁止/許可、例外の有無を逆転させた矛盾文を1文で作る"),
    ("C4_antonym_state", 0.18, "「〜ない」の単純追加を避け、反意語や両立不可能な状態・属性（「増加↔減少」「拡大↔縮小」等）へ置き換えた矛盾文を1文で作る"),
    ("C5_negation_scope", 0.15, "自然な否定表現を用いて前提文の主張を明確に否定する矛盾文を1文で作る"),
]


def _pick_weighted(rng: random.Random, specs: List[Tuple[str, float, str]]) -> Tuple[str, str]:
    r = rng.random()
    acc = 0.0
    for code, w, desc in specs:
        acc += w
        if r <= acc:
            return code, desc
    return specs[-1][0], specs[-1][2]


def assign_subtypes(seed_idx: int, seed: int = 42) -> Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]]:
    rng = random.Random(seed * 1000003 + seed_idx)
    return (
        _pick_weighted(rng, SUBTYPES_E),
        _pick_weighted(rng, SUBTYPES_N),
        _pick_weighted(rng, SUBTYPES_C),
    )


def build_gen_messages(premise: str, sub_e: Tuple[str, str], sub_n: Tuple[str, str], sub_c: Tuple[str, str]) -> List[dict]:
    sys_msg = (
        "あなたは日本語自然言語推論（NLI）データセットの精密な作成者です。"
        "与えられた根拠テキストのみに基づき、RAGの回答チェックに使える自然で独立した日本語の主張文を3つ（entailment / neutral / contradiction）生成してください。"
        "必ず有効なJSONオブジェクト1個のみを出力し、説明文やマークダウンは一切出力しないでください。"
    )
    usr_msg = f"""以下の根拠テキストを読み、3種類の関係に対応する仮説文を各1文ずつ作成してください。

【共通の厳守ルール】
1. 各文は20〜110文字程度の、単体で意味が通じる自然な日本語1文（「。」で終える）にすること。
2. メタ言及（「前提」「仮説」「本文」「文章では」「この文書」「上記」「以下」「述べられている」「記載されている」等）は絶対に使わないこと。
3. 根拠テキストの表現をそのまま長く丸コピーしないこと。
4. 文頭に「これ」「それ」「この結果」などの文脈依存の指示語を置かないこと。

【今回の推論タイプ指定】
- entailment（含意：根拠テキストから確実に真と言える文）[{sub_e[0]}]: {sub_e[1]}
- neutral（中立：根拠テキストだけでは真偽が定まらない文）[{sub_n[0]}]: {sub_n[1]}
- contradiction（矛盾：根拠テキストと論理的に両立せず確実に偽となる文）[{sub_c[0]}]: {sub_c[1]}

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
5: 完全に自然な独立した日本語文
4: 自然で問題ない
3: やや不自然・翻訳調・文脈不足
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
    # thinkingタグ除去
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
        # 末尾カンマ等の軽微修復
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
    # prefix caching とマルチモーダル画像無効化を試行し、古い vLLM ならフォールバック
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


def run_vllm_prompts(
    model_id: str,
    prompts: List[str],
    *,
    temperature: float,
    max_tokens: int,
    max_model_len: int = 4096,
    gpu_util: float = 0.88,
) -> List[str]:
    from vllm import SamplingParams

    llm = _init_vllm_llm(model_id, max_model_len=max_model_len)
    sp = SamplingParams(
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(prompts, sp)
    texts = [o.outputs[0].text if o.outputs else "" for o in outputs]
    del llm
    gc.collect()
    return texts


def run_transformers_prompts(
    model_id: str,
    prompts: List[str],
    *,
    temperature: float,
    max_tokens: int,
    batch_size: int = 16,
) -> List[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    results: List[str] = []
    for st in range(0, len(prompts), batch_size):
        batch = prompts[st : st + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=3584).to(model.device)
        in_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tok.pad_token_id,
            )
        for row in gen:
            out_ids = row[in_len:]
            results.append(tok.decode(out_ids, skip_special_tokens=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def generate_batch_with_fallback(
    model_id: str,
    fallback_model_id: Optional[str],
    messages_list: List[List[dict]],
    *,
    backend: str = "auto",
    temperature: float = 0.0,
    max_tokens: int = 256,
    batch_size: int = 16,
) -> Tuple[str, List[str]]:
    from transformers import AutoTokenizer

    candidates = [model_id] + ([fallback_model_id] if fallback_model_id else [])
    last_err = None
    for mid in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            prompts = [apply_chat_template_safe(tok, m) for m in messages_list]
            if backend in ("auto", "vllm"):
                try:
                    t0 = time.time()
                    outs = run_vllm_prompts(mid, prompts, temperature=temperature, max_tokens=max_tokens)
                    print(f"[INFO] vLLM succeeded with {mid} ({len(prompts)} prompts in {time.time()-t0:.1f}s)")
                    return mid, outs
                except Exception as e:
                    print(f"[WARN] vLLM failed for {mid}: {e}")
                    last_err = e
                    if backend == "vllm":
                        continue
            # transformers fallback (特に小型・Gemma4向け)
            t0 = time.time()
            outs = run_transformers_prompts(
                mid, prompts, temperature=temperature, max_tokens=max_tokens, batch_size=batch_size
            )
            print(f"[INFO] transformers succeeded with {mid} ({len(prompts)} prompts in {time.time()-t0:.1f}s)")
            return mid, outs
        except Exception as e:
            print(f"[WARN] candidate {mid} failed completely: {e}")
            last_err = e
            gc.collect()
    raise RuntimeError(f"All model candidates failed: {last_err}")


def phase_gen(args) -> None:
    chunks_path = Path(args.chunks)
    out_raw = Path(args.out_dir) / "synth_raw.jsonl"
    out_raw.parent.mkdir(parents=True, exist_ok=True)

    chunks = [json.loads(ln) for ln in chunks_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.pool_filter:
        pools = set(args.pool_filter.split(","))
        chunks = [c for c in chunks if c["split_pool"] in pools]
    if args.max_chunks > 0:
        # 層化サンプリング
        rng = random.Random(42)
        by_key = defaultdict(list)
        for c in chunks:
            by_key[(c["domain_group"], c["len_bucket"])].append(c)
        picked = []
        keys = sorted(by_key.keys())
        for k in keys:
            rng.shuffle(by_key[k])
        idx = 0
        while len(picked) < args.max_chunks and any(by_key.values()):
            k = keys[idx % len(keys)]
            if by_key[k]:
                picked.append(by_key[k].pop())
            idx += 1
        chunks = picked

    # リジューム確認
    done_cids = set()
    if out_raw.exists() and not args.overwrite:
        with out_raw.open("r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    done_cids.add(json.loads(ln)["chunk_id"])
    todo = [c for c in chunks if c["chunk_id"] not in done_cids]
    print(f"[GEN] total_target={len(chunks)}, already_done={len(done_cids)}, todo={len(todo)}")
    if not todo:
        return

    from transformers import AutoTokenizer
    model_used = args.gen_model
    # まず最初のチャンクでモデル+vLLMが起動するかテストし、まとめてバッチ処理（5000チャンクずつファイル保存なら途中失敗なし）
    # vLLM は 1回ロードで全 prompts を処理するのが圧倒的に速いので、モデルを1回起動して500件ずつ書き込み！
    tok = None
    for cand in [args.gen_model, args.gen_fallback]:
        if not cand:
            continue
        try:
            tok = AutoTokenizer.from_pretrained(cand)
            model_used = cand
            break
        except Exception as e:
            print(f"[WARN] tokenizer load failed for {cand}: {e}")

    sub_map = {}
    msgs_all = []
    for c in todo:
        se, sn, sc = assign_subtypes(c["seed_idx"])
        sub_map[c["chunk_id"]] = {"entailment": se[0], "neutral": sn[0], "contradiction": sc[0]}
        msgs_all.append(build_gen_messages(c["premise"], se, sn, sc))

    llm = None
    for cand in [model_used, args.gen_fallback]:
        if not cand:
            continue
        try:
            tok = AutoTokenizer.from_pretrained(cand)
            llm = _init_vllm_llm(cand, max_model_len=3072)
            model_used = cand
            break
        except Exception as e:
            print(f"[WARN] vLLM init failed for {cand}: {e}")

    if llm is None:
        raise RuntimeError("Generator vLLM init failed")

    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.65, top_p=0.92, max_tokens=420)
    prompts_all = [apply_chat_template_safe(tok, m) for m in msgs_all]

    batch_sz = 1000
    mode = "w" if (args.overwrite or not out_raw.exists()) else "a"
    stats = Counter()
    with out_raw.open(mode, encoding="utf-8") as fout:
        for st in range(0, len(todo), batch_sz):
            ed = min(st + batch_sz, len(todo))
            t_b = time.time()
            outs = llm.generate(prompts_all[st:ed], sp)
            for c, o in zip(todo[st:ed], outs):
                raw_txt = o.outputs[0].text if o.outputs else ""
                obj = parse_json_obj(raw_txt)
                stats["chunks_processed"] += 1
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
                    ok, reasons = nr.check_pair_rules(c["premise"], hyp, lbl)
                    if not ok:
                        for r in reasons:
                            stats[f"rule_reject_{r}"] += 1
                        stats["rule_reject_pairs"] += 1
                        continue
                    cands.append({
                        "pair_id": f"{c['chunk_id']}::{lbl[0]}",
                        "chunk_id": c["chunk_id"],
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
                        "subtype": sub_map[c["chunk_id"]][lbl],
                        "generator_model": model_used,
                    })
                kept, intra_drop = nr.filter_intra_premise_similar(cands, jaccard_thresh=0.75)
                stats["intra_sim_dropped"] += intra_drop
                for row in kept:
                    stats["rule_passed_pairs"] += 1
                    stats[f"rule_passed_{row['intended_label']}"] += 1
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[GEN] {ed}/{len(todo)} chunks done ({time.time()-t_b:.1f}s) | passed_pairs={stats['rule_passed_pairs']}")

    gen_stat_path = Path(args.out_dir) / "gen_stats.json"
    gen_stat_path.write_text(json.dumps({"generator_model": model_used, "counts": dict(stats)}, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_vllm_or_tf(model_candidates: List[str], backend: str = "auto"):
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
    mode, mid, tok, eng = engine_tuple
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
    raw_path = Path(args.out_dir) / "synth_raw.jsonl"
    ver_path = Path(args.out_dir) / "synth_verified.jsonl"
    all_ver_log_path = Path(args.out_dir) / "synth_verify_log.jsonl"

    rows = [json.loads(ln) for ln in raw_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    done_pids = set()
    if all_ver_log_path.exists() and not args.overwrite:
        with all_ver_log_path.open("r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    done_pids.add(json.loads(ln)["pair_id"])
    todo = [r for r in rows if r["pair_id"] not in done_pids]
    print(f"[VERIFY] total_raw_pairs={len(rows)}, already_done={len(done_pids)}, todo={len(todo)}")

    cands = [args.verifier_model] + ([args.verifier_fallback] if args.verifier_fallback else [])
    if args.verifier_fallback2:
        cands.append(args.verifier_fallback2)
    if args.verifier_fallback3:
        cands.append(args.verifier_fallback3)

    if todo:
        eng = _load_vllm_or_tf(cands, backend=args.verifier_backend)
        model_used = eng[1]
        print(f"[VERIFY] using {eng[0]} backend with {model_used}")
        batch_sz = 2000
        mode_log = "w" if (args.overwrite or not all_ver_log_path.exists()) else "a"
        mode_ver = "w" if (args.overwrite or not ver_path.exists()) else "a"
        with all_ver_log_path.open(mode_log, encoding="utf-8") as flog, ver_path.open(mode_ver, encoding="utf-8") as fver:
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

    # 集計
    all_logs = [json.loads(ln) for ln in all_ver_log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    vstats = Counter()
    by_cls = defaultdict(Counter)
    by_pool = Counter()
    for rec in all_logs:
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

    gen_stat_path = Path(args.out_dir) / "gen_stats.json"
    gen_stats = json.loads(gen_stat_path.read_text(encoding="utf-8")) if gen_stat_path.exists() else {}
    summary = {
        "gen": gen_stats,
        "verify": {
            "counts": dict(vstats),
            "by_class": {k: dict(v) for k, v in by_cls.items()},
            "accepted_by_pool_class": dict(sorted(by_pool.items())),
        },
    }
    (Path(args.out_dir) / "synth_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[VERIFY SUMMARY] total={vstats['total_verified_input']}, agree={vstats['label_agree']}, accepted={vstats['accepted_total']}")
    print("  accepted_by_pool_class:", dict(sorted(by_pool.items())))


def phase_verify3(args) -> None:
    """human_eval プール用の第3検証役ブラインド照合。"""
    ver_path = Path(args.out_dir) / "synth_verified.jsonl"
    out_he = Path(args.out_dir) / "human_eval_verified.jsonl"
    rows = [
        json.loads(ln)
        for ln in ver_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and json.loads(ln)["split_pool"] == "human_eval"
    ]
    print(f"[VERIFY3] human_eval candidates={len(rows)}")
    if not rows:
        out_he.write_text("", encoding="utf-8")
        return

    cands = [args.verifier3_model] + ([args.verifier3_fallback] if args.verifier3_fallback else [])
    eng = _load_vllm_or_tf(cands, backend="auto")
    model3 = eng[1]
    msgs = [build_verify_messages(r["premise"], r["hypothesis"]) for r in rows]
    outs = _infer_verifier_batch(eng, msgs)

    agree3 = 0
    with out_he.open("w", encoding="utf-8") as fout:
        for r, raw_txt in zip(rows, outs):
            obj = parse_json_obj(raw_txt)
            v3_lbl = obj.get("label") if obj else None
            v3_nat = obj.get("naturalness") if obj else 0
            v3_amb = obj.get("ambiguity") if obj else None
            v3_rsn = str(obj.get("reason", ""))[:120] if obj else ""
            ag = (v3_lbl == r["intended_label"])
            if ag:
                agree3 += 1
            rec = {
                **r,
                "verifier3_model": model3,
                "verifier3_label": v3_lbl,
                "verifier3_naturalness": v3_nat,
                "verifier3_ambiguity": v3_amb,
                "verifier3_reason": v3_rsn,
                "verifier3_agree": bool(ag),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[VERIFY3] model={model3}, agree3={agree3}/{len(rows)} ({agree3/max(1,len(rows)):.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gen", "verify", "verify3", "all"], required=True)
    ap.add_argument("--chunks", default="chunks_all.jsonl")
    ap.add_argument("--out-dir", default="out_synth")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--pool-filter", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--gen-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--gen-fallback", default="Qwen/Qwen3-14B")
    ap.add_argument("--verifier-model", default="google/gemma-4-26b-a4b-it")
    ap.add_argument("--verifier-fallback", default="google/gemma-4-12b-it")
    ap.add_argument("--verifier-fallback2", default="microsoft/phi-4")
    ap.add_argument("--verifier-fallback3", default="llm-jp/llm-jp-3.1-13b-instruct4")
    ap.add_argument("--verifier-backend", default="auto", choices=["auto", "vllm", "transformers"])
    ap.add_argument("--verifier3-model", default="microsoft/phi-4")
    ap.add_argument("--verifier3-fallback", default="llm-jp/llm-jp-3.1-13b-instruct4")
    args = ap.parse_args()

    if args.phase == "gen":
        phase_gen(args)
    elif args.phase == "verify":
        phase_verify(args)
    elif args.phase == "verify3":
        phase_verify3(args)
    elif args.phase == "all":
        # サブプロセス分離で VRAM を 100% クリーン解放する
        script = str(Path(__file__).resolve())
        base_flags = [
            "--chunks", args.chunks,
            "--out-dir", args.out_dir,
            "--max-chunks", str(args.max_chunks),
            "--gen-model", args.gen_model,
            "--gen-fallback", args.gen_fallback,
            "--verifier-model", args.verifier_model,
            "--verifier-fallback", args.verifier_fallback,
            "--verifier-fallback2", args.verifier_fallback2,
            "--verifier-fallback3", args.verifier_fallback3,
            "--verifier-backend", args.verifier_backend,
            "--verifier3-model", args.verifier3_model,
            "--verifier3-fallback", args.verifier3_fallback,
        ]
        if args.pool_filter:
            base_flags.extend(["--pool-filter", args.pool_filter])
        if args.overwrite:
            base_flags.append("--overwrite")

        for ph in ["gen", "verify", "verify3"]:
            if ph == "verify3" and args.max_chunks > 0 and args.pool_filter == "train":
                continue
            print(f"=== Launching subprocess phase: {ph} ===")
            cmd = [sys.executable, script, "--phase", ph] + base_flags
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
