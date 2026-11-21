#!/usr/bin/env python3
"""第12号 ja-sentiment-v1 Colab GPU 用 合成＆完全ブラインド独立検証パイプライン。

サブプロセス分離フェーズ:
  --phase gen        : pilot_recipes.jsonl から Qwen3-30B-A3B-Instruct-2507 で 3クラス同時生成 + 規則フィルタ
  --phase verify     : synth_raw.jsonl を異ファミリー検証役 (Gemma-4-26B-A4B-IT 等) で完全ブラインド照合
  --phase gen_he     : human_eval_recipes.jsonl から異ファミリー生成役 (Gemma-4-31B-IT 等) で 1文独立生成
  --phase verify_he1 : he_raw.jsonl を異ファミリー検証役 V1 (Phi-4 等) で完全ブラインド照合
  --phase verify_he2 : he_verify1.jsonl を異ファミリー検証役 V2 (Qwen3-30B-A3B 等) で完全ブラインド照合
"""
import argparse
import gc
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sentiment_rules as sr  # noqa: E402

DOMAIN_JA = {
    "product_review": "ECサイトの商品レビュー",
    "hotel_dining": "宿泊施設・飲食店の口コミ",
    "sns_short": "SNSの日常つぶやき投稿",
    "support_feedback": "カスタマーサポート・問い合わせ対応への感想",
    "app_review": "スマートフォンアプリ・Webサービスのストア評価",
    "survey_workplace": "イベント・研修・公共施設・社内アンケートの自由記述",
}


def build_pilot_gen_messages(recipe: dict) -> List[dict]:
    sys_msg = (
        "あなたは日本語の実務データ作成の専門家です。"
        "ECレビュー、飲食店口コミ、SNS投稿、サポート感想、アプリ評価、アンケート自由記述など、"
        "リアルで自然な日本語の短文・レビュー文を作成してください。"
        "必ず有効なJSONオブジェクト1個のみを出力し、説明文・マークダウン・ラベル注釈は一切付けないでください。"
    )
    subs = recipe["subtypes"]
    dom_ja = DOMAIN_JA.get(recipe["domain"], recipe["domain"])
    usr_msg = f"""次の場面設定で、互いに異なる3種類の感情極性（positive / neutral / negative）の日本語テキストを1件ずつ作成してください。

【共通の場面設定】
- 場面・媒体: {dom_ja}
- 対象・話題: {recipe['subtopic']}
- 注目ポイント: {recipe['aspect']}
- 書き手の人物像: {recipe['persona']}
- 文体: {recipe['style_desc']}
- 文章量の目安: {recipe['length_desc']}

【厳守ルール】
1. 実際の一般ユーザーが書いたような自然な日本語にし、機械的なテンプレート感や説明臭さを消すこと。
2. 「ポジティブ」「ネガティブ」「中立」「感情極性」などのメタ用語や、「レビュー文：」などの見出しは絶対に含めないこと。
3. 3つの文はそれぞれ異なる書き出し・語彙・角度で書き、同じ言い回しを使い回さないこと。
4. 実在の個人名・企業名・電話番号・メールアドレス・URL・過剰な絵文字や顔文字は書かないこと。

【各極性の作成条件（今回指定の難例・表現パターン）】
- positive [{subs['positive']['code']}]: {subs['positive']['desc']}
- neutral [{subs['neutral']['code']}]: {subs['neutral']['desc']}
- negative [{subs['negative']['code']}]: {subs['negative']['desc']}

【出力形式（JSONのみ）】
{{"positive": "...", "neutral": "...", "negative": "..."}}"""
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}]


def build_he_gen_messages(recipe: dict) -> List[dict]:
    """human_eval 専用：1件独立生成・学習データとは完全に異なるプロンプト構造。"""
    sys_msg = (
        "あなたは日本の一般利用者の声を再現するライティング担当です。"
        "指定された条件に沿ったリアルな日本語の1投稿のみを作成し、JSON形式で返してください。"
    )
    dom_ja = DOMAIN_JA.get(recipe["domain"], recipe["domain"])
    lbl = recipe["target_label"]
    lbl_ja = {"positive": "肯定的（ポジティブ）", "neutral": "中立（どちらでもない・事実中心）", "negative": "否定的（ネガティブ）"}[lbl]
    extra_neutral = (
        "\n- 中立（neutral）作成時の追加厳守事項: 書き手自身の「美味しい」「便利」「良かった」「助かる」「残念」などの主観的評価・好悪を書かず、客観的な事実・伝聞・条件確認・賛否均衡のみに留めてください。"
        if lbl == "neutral"
        else ""
    )
    usr_msg = f"""以下の利用メモをもとに、{dom_ja}に投稿されたような自然な日本語コメントを1件だけ書いてください。

- 話題: {recipe['subtopic']}（特に「{recipe['aspect']}」について）
- 投稿者: {recipe['persona']}
- トーン・文体: {recipe['style_desc']}
- 長さの目安: {recipe['length_desc']}
- 伝えるべき立場: {lbl_ja}
- 表現の工夫: {recipe['subtype_desc']}{extra_neutral}

注意：ラベル名や説明書き、前置き、引用符は出力せず、本文だけをJSONの "text" キーに入れてください。実在の固有サービス名や個人情報は書かないでください。
{{"text": "..."}}"""
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}]


def build_verify_messages(text: str, include_reason: bool = True) -> List[dict]:
    """完全ブラインド独立検証用プロンプト（狙いラベル・ドメイン・サブタイプは一切見せない）。"""
    sys_msg = (
        "あなたは日本語テキストの感情極性および日本語自然性を評価する厳格な独立審査官です。"
        "与えられた日本語テキストのみを読み、書き手の全体的な感情極性と文章の自然さを判定してください。"
        "必ずJSONオブジェクト1個のみを出力してください。"
    )
    out_fmt = (
        '{{"label": "positive|neutral|negative", "naturalness": 1, "ambiguity": "clear|ambiguous", "reason": "30〜60字の判定根拠"}}'
        if include_reason
        else '{{"label": "positive|neutral|negative", "naturalness": 1, "ambiguity": "clear|ambiguous"}}'
    )
    usr_msg = f"""次の日本語テキストを読み、全体としての感情極性（positive / neutral / negative）、日本語としての自然さ（1〜5）、判定の明確さ（clear / ambiguous）を評価してください。

【感情極性ラベル定義】
- positive: 書き手の全体的な評価・感想・満足感が肯定的（小さな不満に触れていても全体として満足・肯定なら positive。「悪くない」「文句なし」等の否定形肯定も positive）。
- negative: 書き手の全体的な評価・感想・印象が否定的・不満（良い点に触れていても全体として不満・非推奨なら negative。皮肉めいた批判や遠回しな不満も negative）。
- neutral: 客観的な事実報告・仕様説明・質問・検討メモで感情が表れていない、または良い点と悪い点が完全に拮抗して全体としてどちらとも決められない均衡状態。

【自然さ (naturalness) 1〜5】
5: 日本語として完全に自然で、実際の利用者の声として違和感がない
4: 自然で問題ない
3: やや翻訳調・不自然・説明臭い
2: 明らかに不自然・メタ用語混入・文意不明瞭
1: 日本語として破綻・文字化け・指示文の漏洩

【明確さ (ambiguity)】
- clear: 全体の感情極性を1つに迷いなく判定できる
- ambiguous: 皮肉か本気か判別不能、または文意が曖昧で極性が揺れる

【対象テキスト】
{text}

【出力形式（JSONのみ）】
{out_fmt}"""
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


def _needs_online_fp8(model_id: str) -> bool:
    low = model_id.lower()
    if any(k in low for k in ("fp8", "awq", "gptq", "bnb", "int4", "int8")):
        return False
    # L4 (22.5 GiB) で 8B〜15B 非量子化モデルを高速・省メモリ動作させるため FP8 オンライン量子化を試行
    return any(k in low for k in ("phi-4", "mistral-nemo", "qwen3-14b", "qwen3-8b", "gemma-4-12b"))


def _detect_gpu_profile() -> Tuple[int, int, float, str]:
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            name = torch.cuda.get_device_name(0)
            return major, minor, vram_gb, name
    except Exception:
        pass
    return 0, 0, 0.0, "unknown"


def _init_vllm_llm(model_id: str, max_model_len: int = 1536, gpu_util: float = 0.90):
    from vllm import LLM

    major, minor, vram_gb, gpu_name = _detect_gpu_profile()
    max_num_seqs = 256
    eff_util = gpu_util
    prefer_bf16_large = False
    prefer_fp8_fast = False

    if (major >= 9 or (major == 8 and minor >= 9)) and vram_gb >= 40:
        # H100 / G4 (Blackwell/Ada 48GB~96GB): FP8 高並列 + 失敗時はBF16へ直接フォールバック
        max_num_seqs = 512
        eff_util = max(gpu_util, 0.92)
        prefer_fp8_fast = True
    elif major == 8 and vram_gb >= 38:
        # A100 (Ampere 40GB/80GB): FP8 Marlin 変換より BF16 原寸の方が高速・安定
        prefer_bf16_large = True
        max_num_seqs = 512 if vram_gb >= 70 else 384
        eff_util = max(gpu_util, 0.92)

    print(
        f"[VLLM INIT] gpu={gpu_name} (cc={major}.{minor}, vram={vram_gb:.1f}GB) | "
        f"max_num_seqs={max_num_seqs}, util={eff_util}, prefer_bf16={prefer_bf16_large}, prefer_fp8_fast={prefer_fp8_fast}",
        flush=True,
    )

    base_kw = dict(
        model=model_id,
        dtype="auto",
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=eff_util,
        trust_remote_code=False,
        disable_log_stats=True,
    )
    quant_trials: List[dict] = []
    if prefer_fp8_fast:
        quant_trials.extend([
            {"quantization": "fp8_per_tensor"},
            {"quantization": "fp8"},
            {},
        ])
    elif prefer_bf16_large:
        # 8B〜14B モデルは 40GB/80GB VRAM 上なら BF16 無量子化が最速
        quant_trials.extend([
            {},
            {"quantization": "fp8_per_tensor"},
            {"quantization": "fp8"},
        ])
    elif _needs_online_fp8(model_id):
        quant_trials.extend([
            {"quantization": "fp8_per_tensor"},
            {"quantization": "fp8"},
            {"quantization": "bitsandbytes", "load_format": "bitsandbytes"},
            {},
        ])
    else:
        quant_trials.append({})

    mm_trials = [
        {"enable_prefix_caching": True, "limit_mm_per_prompt": {"image": 0, "audio": 0}},
        {"enable_prefix_caching": True, "limit_mm_per_prompt": {"image": 0}},
        {"enable_prefix_caching": True},
        {},
    ]
    last_err = None
    for qkw in quant_trials:
        for extra in mm_trials:
            kw = {**base_kw, **qkw, **extra}
            try:
                return LLM(**kw)
            except TypeError:
                continue
            except Exception as e:
                last_err = e
                gc.collect()
                break
    if last_err is not None:
        raise last_err
    return LLM(**base_kw)


def _load_engine(model_candidates: List[str], backend: str = "auto", max_model_len: int = 2048):
    from transformers import AutoTokenizer

    cands = [m for m in model_candidates if m]
    if backend in ("auto", "vllm"):
        for mid in cands:
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                llm = _init_vllm_llm(mid, max_model_len=max_model_len)
                return ("vllm", mid, tok, llm)
            except Exception as e:
                print(f"[WARN] vLLM init failed for {mid}: {e}", flush=True)
                gc.collect()
    if backend in ("auto", "transformers"):
        import torch
        from transformers import AutoModelForCausalLM
        try:
            from transformers import AutoModelForImageTextToText
        except Exception:
            AutoModelForImageTextToText = None

        for mid in cands:
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                tok.padding_side = "left"
                loaders = [AutoModelForCausalLM]
                if "gemma-4" in mid.lower() and AutoModelForImageTextToText is not None:
                    loaders = [AutoModelForImageTextToText, AutoModelForCausalLM]
                mdl = None
                last_e = None
                for loader_cls in loaders:
                    try:
                        from transformers import BitsAndBytesConfig
                        if _needs_online_fp8(mid):
                            qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                            mdl = loader_cls.from_pretrained(mid, quantization_config=qcfg, device_map="auto")
                        else:
                            mdl = loader_cls.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="auto")
                        break
                    except Exception as e_ld:
                        last_e = e_ld
                        continue
                if mdl is None:
                    raise last_e
                mdl.eval()
                return ("transformers", mid, tok, mdl)
            except Exception as e:
                print(f"[WARN] transformers init failed for {mid}: {e}", flush=True)
    raise RuntimeError(f"All model candidates failed: {cands}")


def _infer_batch(eng_tuple, msgs_batch: List[List[dict]], temperature: float, max_tokens: int) -> List[str]:
    mode, mid, tok, eng = eng_tuple
    prompts = [apply_chat_template_safe(tok, m) for m in msgs_batch]
    if mode == "vllm":
        from vllm import SamplingParams

        sp = SamplingParams(
            temperature=temperature,
            top_p=0.92 if temperature > 0 else 1.0,
            max_tokens=max_tokens,
        )
        outs = eng.generate(prompts, sp)
        return [o.outputs[0].text if o.outputs else "" for o in outs]
    else:
        import torch

        results = []
        bs = 16
        for st in range(0, len(prompts), bs):
            b = prompts[st : st + bs]
            enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(eng.device)
            in_len = enc["input_ids"].shape[1]
            with torch.inference_mode():
                gen = eng.generate(
                    **enc,
                    max_new_tokens=max_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else None,
                    top_p=0.92 if temperature > 0 else None,
                    pad_token_id=tok.pad_token_id,
                )
            for row in gen:
                results.append(tok.decode(row[in_len:], skip_special_tokens=True))
        return results


def _load_recipes(path: Path, pool_filter: str, max_recipes: int) -> List[dict]:
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if pool_filter:
        pools = set(x.strip() for x in pool_filter.split(",") if x.strip())
        rows = [r for r in rows if r["split_pool"] in pools]
    if max_recipes > 0 and len(rows) > max_recipes:
        # 層化サンプリング
        import random

        rng = random.Random(42)
        by_key = defaultdict(list)
        for r in rows:
            by_key[(r["split_pool"], r["domain"])].append(r)
        keys = sorted(by_key.keys())
        for k in keys:
            rng.shuffle(by_key[k])
        picked = []
        idx = 0
        while len(picked) < max_recipes and any(by_key.values()):
            k = keys[idx % len(keys)]
            if by_key[k]:
                picked.append(by_key[k].pop())
            idx += 1
        rows = picked
    return rows


def phase_gen(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_raw = out_dir / "synth_raw.jsonl"

    recipes = _load_recipes(Path(args.recipes), args.pool_filter, args.max_recipes)
    done_rids = set()
    if out_raw.exists() and not args.overwrite:
        with out_raw.open("r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    done_rids.add(json.loads(ln)["recipe_id"])
    todo = [r for r in recipes if r["recipe_id"] not in done_rids]
    print(f"[GEN] target={len(recipes)}, already_done_rids={len(done_rids)}, todo={len(todo)}", flush=True)
    if not todo:
        return

    cands = [args.gen_model, args.gen_fallback, args.gen_fallback2]
    eng = _load_engine(cands, backend=args.gen_backend, max_model_len=1536)
    model_used = eng[1]
    print(f"[GEN] using {eng[0]} backend with {model_used}", flush=True)

    batch_sz = int(getattr(args, "gen_batch_sz", 2000))
    mode = "w" if (args.overwrite or not out_raw.exists()) else "a"
    stats = Counter()
    with out_raw.open(mode, encoding="utf-8") as fout:
        for st in range(0, len(todo), batch_sz):
            ed = min(st + batch_sz, len(todo))
            t_b = time.time()
            msgs = [build_pilot_gen_messages(r) for r in todo[st:ed]]
            outs = _infer_batch(eng, msgs, temperature=0.75, max_tokens=460)
            for r, raw_txt in zip(todo[st:ed], outs):
                stats["recipes_processed"] += 1
                obj = parse_json_obj(raw_txt)
                if not obj:
                    stats["json_parse_fail"] += 1
                    continue
                stats["json_parse_ok"] += 1
                batch_items = []
                for lbl in sr.VALID_LABELS:
                    txt = obj.get(lbl)
                    if not isinstance(txt, str):
                        stats[f"missing_{lbl}"] += 1
                        continue
                    txt = sr.clean_surface(txt)
                    ok, reasons = sr.check_text_rules(txt, lbl, r["domain"], r["length_band"])
                    if not ok:
                        for rs in reasons:
                            stats[f"rule_reject_{rs}"] += 1
                        stats["rule_reject_items"] += 1
                        continue
                    sub_info = r["subtypes"][lbl]
                    batch_items.append({
                        "sample_id": f"{r['recipe_id']}::{lbl[:3]}",
                        "recipe_id": r["recipe_id"],
                        "split_pool": r["split_pool"],
                        "domain": r["domain"],
                        "subtopic": r["subtopic"],
                        "aspect": r["aspect"],
                        "persona": r["persona"],
                        "style": r["style"],
                        "length_band": r["length_band"],
                        "intended_label": lbl,
                        "subtype": sub_info["code"],
                        "difficulty": sub_info["difficulty"],
                        "text": txt,
                        "generator_model": model_used,
                    })
                kept, intra_drop = sr.filter_intra_batch_similar(batch_items, text_key="text", jaccard_thresh=0.65)
                stats["intra_sim_dropped"] += intra_drop
                for row in kept:
                    stats["rule_passed_items"] += 1
                    stats[f"rule_passed_{row['intended_label']}"] += 1
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[GEN] {ed}/{len(todo)} recipes done ({time.time()-t_b:.1f}s) | passed={stats['rule_passed_items']}", flush=True)

    (out_dir / "gen_stats.json").write_text(
        json.dumps({"generator_model": model_used, "counts": dict(stats)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def phase_verify(args) -> None:
    out_dir = Path(args.out_dir)
    raw_path = out_dir / "synth_raw.jsonl"
    ver_path = out_dir / "synth_verified.jsonl"
    log_path = out_dir / "synth_verify_log.jsonl"

    rows = [json.loads(ln) for ln in raw_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    done_ids = set()
    if log_path.exists() and not args.overwrite:
        with log_path.open("r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    done_ids.add(json.loads(ln)["sample_id"])
    todo = [r for r in rows if r["sample_id"] not in done_ids]
    print(f"[VERIFY] total_raw={len(rows)}, already_done={len(done_ids)}, todo={len(todo)}", flush=True)

    if todo:
        cands = [args.verifier_model, args.verifier_fallback, args.verifier_fallback2]
        eng = _load_engine(cands, backend=args.verifier_backend, max_model_len=1024)
        model_used = eng[1]
        print(f"[VERIFY] using {eng[0]} backend with {model_used}", flush=True)

        batch_sz = int(getattr(args, "verify_batch_sz", 3000))
        mode_log = "w" if (args.overwrite or not log_path.exists()) else "a"
        mode_ver = "w" if (args.overwrite or not ver_path.exists()) else "a"
        with log_path.open(mode_log, encoding="utf-8") as flog, ver_path.open(mode_ver, encoding="utf-8") as fver:
            for st in range(0, len(todo), batch_sz):
                ed = min(st + batch_sz, len(todo))
                t_b = time.time()
                msgs = [build_verify_messages(r["text"], include_reason=False) for r in todo[st:ed]]
                outs = _infer_batch(eng, msgs, temperature=0.0, max_tokens=48)
                acc_b = 0
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
                    agree = (v_lbl == r["intended_label"])
                    accepted = (
                        v_lbl in sr.VALID_LABELS
                        and agree
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
                        "verifier_agree": bool(agree),
                        "accepted": bool(accepted),
                    }
                    flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if accepted:
                        acc_b += 1
                        fver.write(json.dumps(rec, ensure_ascii=False) + "\n")
                flog.flush()
                fver.flush()
                print(f"[VERIFY] {ed}/{len(todo)} done ({time.time()-t_b:.1f}s) | accepted={acc_b}/{ed-st}", flush=True)

    all_logs = [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    vstats = Counter()
    by_cls = defaultdict(Counter)
    by_sub = defaultdict(Counter)
    by_pool = Counter()
    for rec in all_logs:
        lbl = rec["intended_label"]
        sub = rec.get("subtype", "unknown")
        vstats["total_verified_input"] += 1
        if rec["verifier_label"] in sr.VALID_LABELS:
            vstats["verifier_json_ok"] += 1
        else:
            vstats["verifier_json_fail"] += 1
        if rec["verifier_agree"]:
            vstats["label_agree"] += 1
            by_cls[lbl]["agree"] += 1
            by_sub[sub]["agree"] += 1
        by_cls[lbl]["total"] += 1
        by_sub[sub]["total"] += 1
        vstats[f"nat_{rec['verifier_naturalness']}"] += 1
        vstats[f"amb_{rec['verifier_ambiguity']}"] += 1
        if rec["accepted"]:
            vstats["accepted_total"] += 1
            by_cls[lbl]["accepted"] += 1
            by_sub[sub]["accepted"] += 1
            by_pool[f"{rec['split_pool']}::{lbl}"] += 1

    gen_stat_path = out_dir / "gen_stats.json"
    gen_stats = json.loads(gen_stat_path.read_text(encoding="utf-8")) if gen_stat_path.exists() else {}
    summary = {
        "gen": gen_stats,
        "verify": {
            "counts": dict(vstats),
            "by_class": {k: dict(v) for k, v in sorted(by_cls.items())},
            "by_subtype": {k: dict(v) for k, v in sorted(by_sub.items())},
            "accepted_by_pool_class": dict(sorted(by_pool.items())),
        },
    }
    (out_dir / "synth_stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[VERIFY SUMMARY] total={vstats['total_verified_input']}, agree={vstats['label_agree']}, accepted={vstats['accepted_total']}", flush=True)
    print("  accepted_by_pool_class:", dict(sorted(by_pool.items())), flush=True)


def phase_gen_he(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_raw = out_dir / "he_raw.jsonl"

    recipes = _load_recipes(Path(args.he_recipes), "", args.max_recipes)
    done_rids = set()
    if out_raw.exists() and not args.overwrite:
        with out_raw.open("r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    done_rids.add(json.loads(ln)["recipe_id"])
    todo = [r for r in recipes if r["recipe_id"] not in done_rids]
    print(f"[GEN_HE] target={len(recipes)}, done={len(done_rids)}, todo={len(todo)}", flush=True)
    if not todo:
        return

    cands = [args.he_gen_model, args.he_gen_fallback, args.he_gen_fallback2]
    eng = _load_engine(cands, backend=args.he_gen_backend, max_model_len=2048)
    model_used = eng[1]
    print(f"[GEN_HE] using {eng[0]} backend with {model_used}", flush=True)

    msgs = [build_he_gen_messages(r) for r in todo]
    outs = _infer_batch(eng, msgs, temperature=0.75, max_tokens=260)

    stats = Counter()
    mode = "w" if (args.overwrite or not out_raw.exists()) else "a"
    with out_raw.open(mode, encoding="utf-8") as fout:
        for r, raw_txt in zip(todo, outs):
            stats["recipes_processed"] += 1
            obj = parse_json_obj(raw_txt)
            if not obj or not isinstance(obj.get("text"), str):
                stats["json_parse_fail"] += 1
                continue
            stats["json_parse_ok"] += 1
            txt = sr.clean_surface(obj["text"])
            lbl = r["target_label"]
            ok, reasons = sr.check_text_rules(txt, lbl, r["domain"], r["length_band"])
            if not ok:
                for rs in reasons:
                    stats[f"rule_reject_{rs}"] += 1
                stats["rule_reject_items"] += 1
                continue
            rec = {
                "sample_id": r["recipe_id"],
                "recipe_id": r["recipe_id"],
                "split_pool": "human_eval",
                "domain": r["domain"],
                "subtopic": r["subtopic"],
                "aspect": r["aspect"],
                "persona": r["persona"],
                "style": r["style"],
                "length_band": r["length_band"],
                "intended_label": lbl,
                "subtype": r["subtype"],
                "difficulty": r["difficulty"],
                "text": txt,
                "generator_model": model_used,
            }
            stats["rule_passed_items"] += 1
            stats[f"rule_passed_{lbl}"] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    (out_dir / "he_gen_stats.json").write_text(
        json.dumps({"generator_model": model_used, "counts": dict(stats)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[GEN_HE DONE] {dict(stats)}", flush=True)


def phase_verify_he(args, stage: int) -> None:
    out_dir = Path(args.out_dir)
    if stage == 1:
        in_path = out_dir / "he_raw.jsonl"
        out_path = out_dir / "he_verify1.jsonl"
        cands = [args.he_v1_model, args.he_v1_fallback]
        prefix = "v1"
        backend = args.he_v1_backend
    else:
        in_path = out_dir / "he_verify1.jsonl"
        out_path = out_dir / "he_verified.jsonl"
        cands = [args.he_v2_model, args.he_v2_fallback]
        prefix = "v2"
        backend = args.he_v2_backend

    rows = [json.loads(ln) for ln in in_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # 生成役と同一ファミリー回避のため、generator_model が候補と重なる場合はスキップ
    gen_models = {r.get("generator_model", "") for r in rows}
    filtered_cands = [c for c in cands if c and c not in gen_models]
    if not filtered_cands:
        filtered_cands = [c for c in cands if c]

    eng = _load_engine(filtered_cands, backend=backend, max_model_len=2048)
    model_used = eng[1]
    print(f"[VERIFY_HE{stage}] rows={len(rows)}, using {eng[0]} backend with {model_used}", flush=True)

    msgs = [build_verify_messages(r["text"]) for r in rows]
    outs = _infer_batch(eng, msgs, temperature=0.0, max_tokens=160)

    stats = Counter()
    by_cls = defaultdict(Counter)
    with out_path.open("w", encoding="utf-8") as fout:
        for r, raw_txt in zip(rows, outs):
            obj = parse_json_obj(raw_txt)
            v_lbl = obj.get("label") if obj else None
            v_nat = obj.get("naturalness") if obj else 0
            v_amb = obj.get("ambiguity") if obj else None
            v_rsn = str(obj.get("reason", ""))[:120] if obj else ""
            try:
                v_nat_int = int(v_nat) if v_nat is not None else 0
            except Exception:
                v_nat_int = 0
            agree = (v_lbl == r["intended_label"])
            passed = (v_lbl in sr.VALID_LABELS and agree and v_nat_int >= 4 and v_amb == "clear")
            rec = {
                **r,
                f"{prefix}_model": model_used,
                f"{prefix}_label": v_lbl,
                f"{prefix}_naturalness": v_nat_int,
                f"{prefix}_ambiguity": v_amb,
                f"{prefix}_reason": v_rsn,
                f"{prefix}_agree": bool(agree),
                f"{prefix}_pass": bool(passed),
            }
            if stage == 2:
                rec["both_verifiers_pass"] = bool(rec.get("v1_pass") and passed)
                if rec["both_verifiers_pass"]:
                    stats["both_pass"] += 1
                    by_cls[r["intended_label"]]["both_pass"] += 1
            if agree:
                stats["agree"] += 1
                by_cls[r["intended_label"]]["agree"] += 1
            if passed:
                stats["pass"] += 1
                by_cls[r["intended_label"]]["pass"] += 1
            by_cls[r["intended_label"]]["total"] += 1
            stats["total"] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stat_out = {
        f"stage{stage}_model": model_used,
        "counts": dict(stats),
        "by_class": {k: dict(v) for k, v in sorted(by_cls.items())},
    }
    if stage == 2:
        (out_dir / "he_stats.json").write_text(json.dumps(stat_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[VERIFY_HE{stage} DONE] {stat_out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["gen", "verify", "gen_he", "verify_he1", "verify_he2", "all"],
        required=True,
    )
    ap.add_argument("--recipes", default="pilot_recipes.jsonl")
    ap.add_argument("--he-recipes", default="human_eval_recipes.jsonl")
    ap.add_argument("--out-dir", default="out_synth")
    ap.add_argument("--pool-filter", default="")
    ap.add_argument("--max-recipes", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--gen-batch-sz", type=int, default=2000)
    ap.add_argument("--verify-batch-sz", type=int, default=3000)

    ap.add_argument("--gen-model", default="Qwen/Qwen3-14B-FP8")
    ap.add_argument("--gen-fallback", default="Qwen/Qwen3-14B-AWQ")
    ap.add_argument("--gen-fallback2", default="Qwen/Qwen3-8B")
    ap.add_argument("--gen-backend", default="auto", choices=["auto", "vllm", "transformers"])

    ap.add_argument("--verifier-model", default="microsoft/phi-4")
    ap.add_argument("--verifier-fallback", default="google/gemma-4-E4B-it")
    ap.add_argument("--verifier-fallback2", default="mistralai/Mistral-Nemo-Instruct-2407")
    ap.add_argument("--verifier-backend", default="auto", choices=["auto", "vllm", "transformers"])

    ap.add_argument("--he-gen-model", default="google/gemma-4-E4B-it")
    ap.add_argument("--he-gen-fallback", default="mistralai/Mistral-Nemo-Instruct-2407")
    ap.add_argument("--he-gen-fallback2", default="microsoft/phi-4")
    ap.add_argument("--he-gen-backend", default="auto", choices=["auto", "vllm", "transformers"])

    ap.add_argument("--he-v1-model", default="microsoft/phi-4")
    ap.add_argument("--he-v1-fallback", default="mistralai/Mistral-Nemo-Instruct-2407")
    ap.add_argument("--he-v1-backend", default="auto", choices=["auto", "vllm", "transformers"])

    ap.add_argument("--he-v2-model", default="Qwen/Qwen3-14B-FP8")
    ap.add_argument("--he-v2-fallback", default="Qwen/Qwen3-14B-AWQ")
    ap.add_argument("--he-v2-backend", default="auto", choices=["auto", "vllm", "transformers"])

    args = ap.parse_args()

    if args.phase == "gen":
        phase_gen(args)
    elif args.phase == "verify":
        phase_verify(args)
    elif args.phase == "gen_he":
        phase_gen_he(args)
    elif args.phase == "verify_he1":
        phase_verify_he(args, 1)
    elif args.phase == "verify_he2":
        phase_verify_he(args, 2)
    elif args.phase == "all":
        script = str(Path(__file__).resolve())
        common = [
            "--recipes", args.recipes,
            "--he-recipes", args.he_recipes,
            "--out-dir", args.out_dir,
            "--max-recipes", str(args.max_recipes),
            "--gen-batch-sz", str(args.gen_batch_sz),
            "--verify-batch-sz", str(args.verify_batch_sz),
            "--gen-model", args.gen_model,
            "--gen-fallback", args.gen_fallback,
            "--gen-fallback2", args.gen_fallback2,
            "--gen-backend", args.gen_backend,
            "--verifier-model", args.verifier_model,
            "--verifier-fallback", args.verifier_fallback,
            "--verifier-fallback2", args.verifier_fallback2,
            "--verifier-backend", args.verifier_backend,
            "--he-gen-model", args.he_gen_model,
            "--he-gen-fallback", args.he_gen_fallback,
            "--he-gen-fallback2", args.he_gen_fallback2,
            "--he-gen-backend", args.he_gen_backend,
            "--he-v1-model", args.he_v1_model,
            "--he-v1-fallback", args.he_v1_fallback,
            "--he-v1-backend", args.he_v1_backend,
            "--he-v2-model", args.he_v2_model,
            "--he-v2-fallback", args.he_v2_fallback,
            "--he-v2-backend", args.he_v2_backend,
        ]
        if args.pool_filter:
            common.extend(["--pool-filter", args.pool_filter])
        if args.overwrite:
            common.append("--overwrite")

        for ph in ["gen", "verify", "gen_he", "verify_he1", "verify_he2"]:
            print(f"=== Subprocess phase: {ph} ===", flush=True)
            subprocess.run([sys.executable, script, "--phase", ph] + common, check=True)
            gc.collect()


if __name__ == "__main__":
    main()
