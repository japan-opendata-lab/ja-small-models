#!/usr/bin/env python3
"""公開データ (llmail-inject-challenge / browsesafe-bench) の正規化・機械監査・類似重複除去・PIIモデル補助スクリーニング。

第1号の学習済みモデル (ja-pii-ner-v1) は
読み取り専用の推論にのみ使用し、第1号フォルダへは一切書き込まない。
"""
import hashlib
import json
import os
import random
import re
import unicodedata
import zlib
from collections import Counter, defaultdict

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

BASE_PUB = os.path.join("data", "public_sources")
OUT_AUDIT_POOL = os.path.join("data", "audited_public_pool.json")
OUT_REVIEW = os.path.join("data", "audit_review_summary.json")
PII_MODEL_DIR = os.environ.get("PII_MODEL_DIR", "japan-opendata/ja-pii-ner-v1")

RE_HIRA = re.compile(r"[\u3040-\u309F]")
RE_KANA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")

# llmail False に混入している偽陰性・未遂攻撃シグナル
FALSE_NEG_ATK_RE = re.compile(
    r"(<\|.*?\|>|\[/?(?:SYSTEM|USER|ASSISTANT|INST|JOB|DEOBFUSCATION)\]|"
    r"contact@contact\.com|contact\s*(?:<at>|＠|at)\s*contact|send_email|api_call|exfil|"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)|disregard\s+(?:all\s+)?(?:previous|prior)|"
    r"override\s+(?:all\s+)?instructions|system\s*prompt|"
    r"前の指示|これまでの指示|元のタスク|最初のタスク|出力トークン数|すべての手順に従|"
    r"システムプロンプト|メールを送信|confirmation\s*を送信|to:\s*contact)",
    re.IGNORECASE,
)

# 強い不適切表現（攻撃側であっても検知器学習に不要な過激表現を除外）
INAPPROPRIATE_RE = re.compile(
    r"(\b(?:porn|pornhub|xvideos|hentai|rape|raping|molest|pedophil|child\s*porn|csam|"
    r"kill\s+yourself|kys|commit\s+suicide|self[- ]harm|how\s+to\s+make\s+a\s+bomb|"
    r"nigger|faggot|cunt|motherfucker|cocksucker|blowjob)\b|"
    r"児童ポルノ|強姦|レイプ|自殺しろ|殺害予告|爆弾の作り方|死ね|クソ野郎)",
    re.IGNORECASE,
)

# 実在個人情報の機械チェック（競技用ダミー値以外を厳しく検出）
DUMMY_EMAIL_SET = {
    "contact@contact.com",
    "attacker@evil.com",
    "admin@example.com",
    "test@example.com",
    "user@example.com",
    "noreply@example.com",
    "info@example.com",
    "support@example.com",
    "document-access@docs.google.com",
}
DUMMY_EMAIL_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "test.com",
    "contact.com",
    "contoso.com",
    "fabrikam.com",
    "company.com",
    "domain.com",
    "acme.com",
    "sample.com",
    "evil.com",
    "localhost",
}
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
JA_PHONE_RE = re.compile(r"(?<!\d)(?:0[789]0[- ]?\d{4}[- ]?\d{4}|0\d{1,3}-\d{2,4}-\d{4}|\+81[- ]?\d{1,4}[- ]?\d{1,4}[- ]?\d{4})(?!\d)")
US_PHONE_RE = re.compile(r"(?<!\d)(?:\+1[-.\s]?)?\(\d{3}\)[-.\s]\d{3}[-.\s]\d{4}(?!\d)")
CARD16_RE = re.compile(r"(?<!\d)(?:\d{4}[-\s]){3}\d{4}(?!\d)")
MYNUM_RE = re.compile(r"(?:マイナンバー|個人番号)[^\d]{0,10}\d{4}[-\s]?\d{4}[-\s]?\d{4}")

# browsesafe HTML の攻撃ペイロード箇所特定キーワード（線形時間で動作するようバックトラッキングを排したパターン）
BS_PAYLOAD_RE = re.compile(
    r"(ignore\s+(?:all\s+)?(?:previous|prior)|disregard|system\s*prompt|"
    r"important\s+message|important\s+security|todo:|urgent[:\s]|"
    r"verify\s+your\s+|confirm\s+at:|data-instruction|"
    r"<!--|override|前の指示|確認してください|制限され)",
    re.IGNORECASE,
)


def has_regex_pii(text: str) -> str | None:
    for m in EMAIL_RE.finditer(text):
        full = m.group(0).lower()
        dom = m.group(1).lower()
        if full in DUMMY_EMAIL_SET or dom in DUMMY_EMAIL_DOMAINS:
            continue
        # HTML/Webページの汎用窓口系ローカルパートを除外しつつ個人・実在企業アドレスは除外
        local = full.split("@", 1)[0]
        if local in {"support", "info", "noreply", "no-reply", "contact", "help", "sales", "press", "privacy", "security"} and dom.endswith((".com", ".org", ".io")):
            continue
        return f"email:{full}"
    if JA_PHONE_RE.search(text):
        return "ja_phone"
    if US_PHONE_RE.search(text):
        return "us_phone"
    if CARD16_RE.search(text):
        return "card16"
    if MYNUM_RE.search(text):
        return "my_number"
    return None


def norm_compact(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\s+", "", s)
    if len(s) > 2000:
        s = s[:1000] + s[-1000:]
    return s


SEEDS16 = [i * 0x9E3779B1 + 12345 for i in range(16)]


def minhash16(nc: str) -> tuple[int, ...]:
    if len(nc) < 5:
        h = zlib.crc32(nc.encode("utf-8"))
        return tuple((h ^ s) & 0xFFFFFFFF for s in SEEDS16)
    step = 2 if len(nc) < 1200 else 4
    grams = [zlib.crc32(nc[i : i + 5].encode("utf-8")) for i in range(0, len(nc) - 4, step)]
    return tuple(min(((g ^ s) & 0xFFFFFFFF) for g in grams) for s in SEEDS16)


def dedup_and_cluster(items: list[dict], sim_dedup: int = 12, sim_cluster: int = 8) -> tuple[list[dict], dict]:
    """正規化完全一致除去 + MinHash LSH 高類似重複除去 + Union-Find 表層クラスタ付与。"""
    seen_exact = set()
    exact_items = []
    for it in items:
        nc = norm_compact(it.get("dedup_key", it["text"]))
        if nc in seen_exact:
            continue
        seen_exact.add(nc)
        it["_nc"] = nc
        exact_items.append(it)

    sigs = [minhash16(it["_nc"]) for it in exact_items]
    bands_dedup = defaultdict(list)
    kept = []
    kept_sigs = []
    for it, sig in zip(exact_items, sigs):
        bkeys = [(b, sig[b * 4 : (b + 1) * 4]) for b in range(4)]
        cands = {cid for bk in bkeys for cid in bands_dedup[bk]}
        if any(sum(1 for a, b in zip(sig, kept_sigs[cid]) if a == b) >= sim_dedup for cid in cands):
            continue
        nid = len(kept)
        kept.append(it)
        kept_sigs.append(sig)
        for bk in bkeys:
            if len(bands_dedup[bk]) < 60:
                bands_dedup[bk].append(nid)

    parent = list(range(len(kept)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    bands_cl = defaultdict(list)
    for idx, sig in enumerate(kept_sigs):
        for b in range(8):
            bk = (b, sig[b * 2 : (b + 1) * 2])
            for prev in bands_cl[bk]:
                if sum(1 for a, c in zip(sig, kept_sigs[prev]) if a == c) >= sim_cluster:
                    union(idx, prev)
                    break
            if len(bands_cl[bk]) < 40:
                bands_cl[bk].append(idx)

    for idx, it in enumerate(kept):
        it["cluster"] = f"cl_{find(idx)}"
        it.pop("_nc", None)
        it.pop("dedup_key", None)

    stats = {
        "before": len(items),
        "after_exact": len(exact_items),
        "after_neardup": len(kept),
        "clusters": len({it["cluster"] for it in kept}),
    }
    return kept, stats


def format_email_natural(raw_key: str, lang: str) -> str:
    m = re.match(r"^Subject of the email:\s*(.*?)\.\s{2,}Body:\s*(.*)$", raw_key, flags=re.DOTALL)
    if m:
        subj, body = m.group(1).strip(), m.group(2).strip()
    else:
        subj, body = "", raw_key.strip()
    h = int(hashlib.md5(raw_key.encode("utf-8")).hexdigest()[:8], 16) % 4
    if not subj:
        return body
    if lang == "ja":
        if h == 0:
            return f"件名: {subj}\n本文:\n{body}"
        if h == 1:
            return f"【受信メール】\n件名: {subj}\n\n{body}"
        if h == 2:
            return f"件名：{subj}\n---\n{body}"
        return f"{subj}\n\n{body}"
    else:
        if h == 0:
            return f"Subject: {subj}\n\n{body}"
        if h == 1:
            return f"Email Subject: {subj}\nBody:\n{body}"
        if h == 2:
            return f"From: Sender\nSubject: {subj}\n---\n{body}"
        return f"{subj}\n\n{body}"


def classify_llmail_subcat(text: str) -> str:
    low = text.lower()
    if re.search(r"(<\|(?:system|assistant|user|im_start|end)\|>\s*|\[/?(?:system|assistant|user|inst)\]|"
                 r"---会話履歴---|システム通知|new\s+policy|admin\s+mode)", low):
        return "fake_history"
    if re.search(r"(base64|rot13|decode|復号|縦読み|頭文字|[a-z0-9+/]{24,}={0,2})", low):
        return "obfuscation"
    if re.search(r"(system\s*prompt|initial\s+instruction|システムプロンプト|初期設定|内部指示|内部設定を(?:出力|表示|教え))", low):
        return "extraction"
    if re.search(r"(あなたはこれから|なりきって|act\s+as|you\s+are\s+now\s+|developer\s+mode)", low):
        return "role_hijack"
    if re.search(r"(ignore\s+(?:all\s+)?(?:previous|prior|above)|disregard|前の指示は無視|これまでの指示を無視)", low):
        return "override"
    return "indirect"


def clean_html_window(html: str, is_yes: bool, is_ja: bool, seed_key: str) -> str | None:
    s = re.sub(r"<style\b[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<svg\b[^>]*>.*?</svg>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r'\s+style="[^"]*"', "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+style='[^']*'", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= 2600:
        return s
    h = int(hashlib.md5(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    win = 1400 + (h % 600)  # 1400..1999 chars
    if is_yes:
        m = BS_PAYLOAD_RE.search(s)
        if not m:
            return None
        p = m.start()
        # Place payload at beginning / middle / end depending on hash
        mode = h % 3
        if mode == 0:
            start = max(0, p - 120)
        elif mode == 1:
            start = max(0, p - win // 2)
        else:
            start = max(0, p - int(win * 0.8))
        if start + win > len(s):
            start = max(0, len(s) - win)
        return s[start : start + win].strip()
    else:
        if is_ja:
            m = RE_HIRA.search(s)
            p = m.start() if m else len(s) // 2
            start = max(0, p - win // 2)
        else:
            start = h % max(1, len(s) - win)
        if start + win > len(s):
            start = max(0, len(s) - win)
        return s[start : start + win].strip()


def run_pii_screening(items: list[dict], tok, model, device: str, batch_size: int = 32) -> tuple[list[dict], list[dict]]:
    """第1号学習済みBERTモデルでPIIエンティティを検出し、要確認行とクリーン行に分離する。"""
    if not items:
        return [], []
    id2label = model.config.id2label

    flagged = []
    clean = []
    HIGH_RISK_TYPES = {
        "EMAIL", "PHONE", "ADDRESS", "POSTAL_CODE", "DATE_OF_BIRTH", "MY_NUMBER",
        "BANK_ACCOUNT", "CREDIT_CARD", "DRIVER_LICENSE", "PASSPORT", "URL_PERSONAL",
        "USERNAME", "SECRET", "EMPLOYEE_ID", "IP_ADDRESS", "PERSON",
    }

    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        texts = [x["text"] for x in chunk]
        enc = tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
        input_ids_list = enc["input_ids"].tolist()
        attn_list = enc["attention_mask"].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(dim=-1).cpu().tolist()

        spec_ids = set(tok.all_special_ids)
        for row, pred_ids, ids_row, mask_row in zip(chunk, preds, input_ids_list, attn_list):
            entities = []
            cur_type = None
            cur_toks = []

            def flush_ent():
                nonlocal cur_type, cur_toks
                if cur_type and cur_toks:
                    val = tok.convert_tokens_to_string(cur_toks).strip()
                    if row.get("lang") == "ja":
                        val_cmp = val.replace(" ", "")
                        if val_cmp:
                            val = val_cmp
                    entities.append((cur_type, val))
                cur_type = None
                cur_toks = []

            for pid, tid, am in zip(pred_ids, ids_row, mask_row):
                if am == 0 or tid in spec_ids:
                    flush_ent()
                    continue
                lbl = id2label[pid]
                if lbl == "O":
                    flush_ent()
                    continue
                tok_str = tok.convert_ids_to_tokens(tid)
                if lbl.startswith("B-"):
                    flush_ent()
                    cur_type = lbl[2:]
                    cur_toks = [tok_str]
                elif lbl.startswith("I-") and cur_type == lbl[2:]:
                    cur_toks.append(tok_str)
                else:
                    flush_ent()
            flush_ent()

            risk_ents = [(t, v) for t, v in entities if t in HIGH_RISK_TYPES]
            # 既知の競技定型フィクション文字・メールを除外判定から外す
            filtered_risk = []
            for t, v in risk_ents:
                vl = v.strip().lower()
                if t == "EMAIL" and (vl in DUMMY_EMAIL_SET or any(vl.endswith("@" + d) for d in DUMMY_EMAIL_DOMAINS)):
                    continue
                if t == "PERSON" and vl in {"casey", "emily watson", "detlef", "john", "alice", "bob", "david"}:
                    continue
                if len(vl) <= 1:
                    continue
                filtered_risk.append((t, v))

            if filtered_risk:
                row["pii_entities"] = filtered_risk
                flagged.append(row)
            else:
                clean.append(row)
    return clean, flagged


def main():
    drop_counts = Counter()

    # 1. Load llmail-inject-challenge
    llmail_ja_atk_raw = []
    llmail_en_atk_raw = []
    llmail_ja_ben_raw = []
    llmail_en_ben_raw = []

    ll_dir = os.path.join(BASE_PUB, "llmail-inject-challenge", "data")
    for ph in ["phase1", "phase2"]:
        src = f"llmail_{ph}"
        d = json.load(open(os.path.join(ll_dir, f"labelled_unique_submissions_{ph}.json"), encoding="utf-8"))
        for raw_key, meta in d.items():
            att = str(meta.get("attack_attempt"))
            if att not in ("True", "False"):
                drop_counts["llmail_unclear_or_invalid"] += 1
                continue
            is_ja = len(RE_HIRA.findall(raw_key)) >= 5
            lang = "ja" if is_ja else "en"
            if att == "False" and FALSE_NEG_ATK_RE.search(raw_key):
                drop_counts[f"llmail_false_neg_attack_{lang}"] += 1
                continue
            if INAPPROPRIATE_RE.search(raw_key):
                drop_counts[f"llmail_inappropriate_{lang}_{att}"] += 1
                continue
            pii_hit = has_regex_pii(raw_key)
            if pii_hit:
                drop_counts[f"llmail_regex_pii_{lang}_{att}"] += 1
                continue

            rec = {
                "text": format_email_natural(raw_key, lang),
                "dedup_key": raw_key,
                "label": 1 if att == "True" else 0,
                "subcategory": "indirect" if att == "True" else "benign",
                "source": src,
                "lang": lang,
            }
            if att == "True":
                (llmail_ja_atk_raw if lang == "ja" else llmail_en_atk_raw).append(rec)
            else:
                (llmail_ja_ben_raw if lang == "ja" else llmail_en_ben_raw).append(rec)

    # Load llmail FP tests and scenario benign emails
    fp_emails = json.load(open(os.path.join(ll_dir, "emails_for_fp_tests.json"), encoding="utf-8"))
    for raw_key in fp_emails:
        if has_regex_pii(raw_key) or INAPPROPRIATE_RE.search(raw_key):
            drop_counts["llmail_fp_test_filtered"] += 1
            continue
        llmail_en_ben_raw.append({
            "text": format_email_natural(raw_key, "en"),
            "dedup_key": raw_key,
            "label": 0,
            "subcategory": "benign",
            "source": "llmail_fp_test",
            "lang": "en",
        })

    scen = json.load(open(os.path.join(ll_dir, "scenarios.json"), encoding="utf-8"))
    for sdata in scen.values():
        for raw_key in sdata.get("emails", []):
            if has_regex_pii(raw_key) or INAPPROPRIATE_RE.search(raw_key):
                drop_counts["llmail_scenario_filtered"] += 1
                continue
            llmail_en_ben_raw.append({
                "text": format_email_natural(raw_key, "en"),
                "dedup_key": raw_key,
                "label": 0,
                "subcategory": "benign",
                "source": "llmail_scenario",
                "lang": "en",
            })

    print("[1/3] Deduplicating and clustering llmail...", flush=True)
    rng = random.Random(42)
    ll_ja_atk, st_ja_atk = dedup_and_cluster(llmail_ja_atk_raw)
    ll_ja_ben, st_ja_ben = dedup_and_cluster(llmail_ja_ben_raw)
    ll_en_ben, st_en_ben = dedup_and_cluster(llmail_en_ben_raw)

    seen_en = set()
    ll_en_atk_exact = []
    for it in llmail_en_atk_raw:
        nc = norm_compact(it["dedup_key"])
        if nc not in seen_en:
            seen_en.add(nc)
            ll_en_atk_exact.append(it)
    rng.shuffle(ll_en_atk_exact)
    ll_en_atk, st_en_atk = dedup_and_cluster(ll_en_atk_exact[:6000])
    st_en_atk["before_all_raw"] = len(llmail_en_atk_raw)
    st_en_atk["after_exact_all"] = len(ll_en_atk_exact)

    for it in ll_ja_atk + ll_en_atk:
        it["subcategory"] = classify_llmail_subcat(it["text"])
    print(f"  llmail done: ja_atk={len(ll_ja_atk)}, ja_ben={len(ll_ja_ben)}, en_atk={len(ll_en_atk)}, en_ben={len(ll_en_ben)}", flush=True)

    # 2. Load browsesafe-bench (train and test kept strictly separate!)
    print("[2/3] Loading and windowing browsesafe-bench...", flush=True)
    bs_pools = {}
    bs_stats = {}
    for sp in ["train", "test"]:
        t = pq.read_table(os.path.join(BASE_PUB, "browsesafe-bench", f"{sp}.parquet"))
        all_rows = t.to_pylist()
        groups = defaultdict(list)
        for idx, r in enumerate(all_rows):
            html = r["content"]
            is_yes = r["label"] == "yes"
            is_ja = len(RE_HIRA.findall(html)) >= 5
            groups[(is_yes, is_ja)].append((idx, html, is_yes, is_ja))

        cap = 2200 if sp == "train" else 300
        selected_yes, selected_no = [], []
        for (is_yes, is_ja), lst in groups.items():
            if not is_ja:
                rng.shuffle(lst)
                lst = lst[:cap]
            for idx, html, iy, ij in lst:
                lang = "ja" if ij else "en"
                win = clean_html_window(html, is_yes=iy, is_ja=ij, seed_key=f"{sp}_{idx}")
                if win is None:
                    drop_counts[f"browsesafe_{sp}_payload_not_located"] += 1
                    continue
                if INAPPROPRIATE_RE.search(win):
                    drop_counts[f"browsesafe_{sp}_inappropriate"] += 1
                    continue
                if has_regex_pii(win):
                    drop_counts[f"browsesafe_{sp}_regex_pii"] += 1
                    continue
                rec = {
                    "text": win,
                    "label": 1 if iy else 0,
                    "subcategory": "indirect" if iy else "benign_long",
                    "source": f"browsesafe_{sp}",
                    "lang": lang,
                }
                (selected_yes if iy else selected_no).append(rec)

        bs_yes, st_by = dedup_and_cluster(selected_yes)
        bs_no, st_bn = dedup_and_cluster(selected_no)
        bs_pools[f"{sp}_yes"] = bs_yes
        bs_pools[f"{sp}_no"] = bs_no
        bs_stats[f"{sp}_yes"] = st_by
        bs_stats[f"{sp}_no"] = st_bn
        print(f"  browsesafe {sp}: yes={len(bs_yes)}, no={len(bs_no)}", flush=True)

    # 3. Run PII model screening on llmail JA full + sampled EN + browsesafe JA
    print("[3/3] Running PII model screening on GPU...", flush=True)
    tok = AutoTokenizer.from_pretrained(PII_MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(PII_MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.half()
    model.to(device).eval()

    rng.shuffle(ll_en_atk)
    rng.shuffle(ll_en_ben)
    ll_ja_atk_clean, ll_ja_atk_flag = run_pii_screening(ll_ja_atk, tok, model, device)
    print(f"  screened ll_ja_atk: clean={len(ll_ja_atk_clean)}, flag={len(ll_ja_atk_flag)}", flush=True)
    ll_ja_ben_clean, ll_ja_ben_flag = run_pii_screening(ll_ja_ben, tok, model, device)
    ll_en_atk_clean, ll_en_atk_flag = run_pii_screening(ll_en_atk[:3500], tok, model, device)
    print(f"  screened ll_en_atk: clean={len(ll_en_atk_clean)}, flag={len(ll_en_atk_flag)}", flush=True)
    ll_en_ben_clean, ll_en_ben_flag = run_pii_screening(ll_en_ben[:1000], tok, model, device)

    bs_train_ja = [x for x in bs_pools["train_yes"] + bs_pools["train_no"] if x["lang"] == "ja"]
    bs_train_ja_clean, bs_train_ja_flag = run_pii_screening(bs_train_ja, tok, model, device)
    bs_test_ja = [x for x in bs_pools["test_yes"] + bs_pools["test_no"] if x["lang"] == "ja"]
    bs_test_ja_clean, bs_test_ja_flag = run_pii_screening(bs_test_ja, tok, model, device)

    drop_counts["pii_model_flagged_llmail_ja_atk"] = len(ll_ja_atk_flag)
    drop_counts["pii_model_flagged_llmail_ja_ben"] = len(ll_ja_ben_flag)
    drop_counts["pii_model_flagged_llmail_en_atk"] = len(ll_en_atk_flag)
    drop_counts["pii_model_flagged_llmail_en_ben"] = len(ll_en_ben_flag)
    drop_counts["pii_model_flagged_browsesafe_ja"] = len(bs_train_ja_flag) + len(bs_test_ja_flag)

    # Save audited pools
    pool_out = {
        "llmail_ja_atk": ll_ja_atk_clean,
        "llmail_ja_ben": ll_ja_ben_clean,
        "llmail_en_atk": ll_en_atk_clean,
        "llmail_en_ben": ll_en_ben_clean,
        "browsesafe_train_yes": [x for x in bs_pools["train_yes"] if x["lang"] == "en"] + [x for x in bs_train_ja_clean if x["label"] == 1],
        "browsesafe_train_no": [x for x in bs_pools["train_no"] if x["lang"] == "en"] + [x for x in bs_train_ja_clean if x["label"] == 0],
        "browsesafe_test_yes": [x for x in bs_pools["test_yes"] if x["lang"] == "en"] + [x for x in bs_test_ja_clean if x["label"] == 1],
        "browsesafe_test_no": [x for x in bs_pools["test_no"] if x["lang"] == "en"] + [x for x in bs_test_ja_clean if x["label"] == 0],
    }
    with open(OUT_AUDIT_POOL, "w", encoding="utf-8") as f:
        json.dump(pool_out, f, ensure_ascii=False)

    # Save review summary + sample inspection snippets (excluding browsesafe_test content)
    flagged_examples = []
    for tag, flist in [
        ("llmail_ja_atk", ll_ja_atk_flag),
        ("llmail_ja_ben", ll_ja_ben_flag),
        ("llmail_en_atk", ll_en_atk_flag),
        ("llmail_en_ben", ll_en_ben_flag),
    ]:
        for x in flist[:8]:
            flagged_examples.append({"group": tag, "entities": x["pii_entities"], "snippet": x["text"][:160]})

    clean_samples = []
    for tag, clist in [
        ("llmail_ja_atk", ll_ja_atk_clean),
        ("llmail_ja_ben", ll_ja_ben_clean),
        ("llmail_en_atk", ll_en_atk_clean),
        ("llmail_en_ben", ll_en_ben_clean),
    ]:
        for x in clist[:8]:
            clean_samples.append({"group": tag, "source": x["source"], "subcat": x["subcategory"], "snippet": x["text"][:200]})

    summary = {
        "drop_counts": dict(sorted(drop_counts.items())),
        "dedup_stats": {
            "llmail_ja_atk": st_ja_atk,
            "llmail_ja_ben": st_ja_ben,
            "llmail_en_atk": st_en_atk,
            "llmail_en_ben": st_en_ben,
            "browsesafe": bs_stats,
        },
        "clean_pool_counts": {k: len(v) for k, v in pool_out.items()},
        "flagged_entity_type_counts": {
            tag: dict(Counter(t for x in flist for t, _ in x["pii_entities"]))
            for tag, flist in [
                ("llmail_ja_atk", ll_ja_atk_flag),
                ("llmail_ja_ben", ll_ja_ben_flag),
                ("llmail_en_atk", ll_en_atk_flag),
                ("llmail_en_ben", ll_en_ben_flag),
            ]
        },
        "flagged_examples": flagged_examples,
        "clean_samples": clean_samples,
    }
    with open(OUT_REVIEW, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== AUDIT COMPLETE ===")
    print("Drop counts:", json.dumps(summary["drop_counts"], ensure_ascii=False, indent=2))
    print("Dedup stats:", json.dumps(summary["dedup_stats"], ensure_ascii=False))
    print("Clean pool counts:", json.dumps(summary["clean_pool_counts"], ensure_ascii=False))
    print("Flagged entity types:", json.dumps(summary["flagged_entity_type_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
