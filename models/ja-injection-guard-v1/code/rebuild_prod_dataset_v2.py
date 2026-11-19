#!/usr/bin/env python3
"""第2号 本番データセット v2 再構築スクリプト。

変更要件（v1 最終評価の教訓を反映）:
1. 既存の v1 封印テスト (data/prod/sealed-test.jsonl, 2,500件) は先に
   data/prod/diagnostic-test.jsonl へ「診断用セット」として格下げ保存する（冪等）。
2. train (45,000件) と dev (5,000件) について、実在ソースおよび synth の非長文・
   browsesafe 由来の benign_long（train: 35,434件 / dev: 3,908件）は完全保持しつつ、
   synth 由来の benign_long（train: 9,566件 / dev: 1,092件）を新しい大規模段落プール
   (code/benign_long_pool.py: 8文体×50=400基底、具体段落9,900本、段落あたり上限5件) で再生成する。
3. 新しい封印テスト data/prod/sealed-test.jsonl (2,500件) を完全分離構築する:
   - 実在保持分 (300件): browsesafe_test EN (50 indirect + 80 benign_long)
     + llmail 未使用・クラスタ非重複 EN (130攻撃 + 40 benign)
   - v2 封印専用 3層合成 + sealed_v2 段落プール (2,200件)
   - 構成比を維持: 攻撃 900件 (36%, 6サブカテゴリ各150件) / benign 500件 (20%)
     / benign_hard 500件 (20%) / benign_long 600件 (24%)、日86.0%・英14.0%
4. 新しい sealed-test.jsonl のテキスト本文は一切標準出力・ログ・統計JSONへ出力しない。
"""
import json
import os
import random
import re
import shutil
import sys
import unicodedata
import zlib
from collections import Counter, defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_public_data as apd
import benign_long_pool as blp
import synth_3layer as s3

BASE_PUB = os.path.join("data", "public_sources")
PROD_DIR = os.path.join("data", "prod")
TRAIN_PATH = os.path.join(PROD_DIR, "train.jsonl")
DEV_PATH = os.path.join(PROD_DIR, "dev.jsonl")
SEALED_PATH = os.path.join(PROD_DIR, "sealed-test.jsonl")
DIAG_PATH = os.path.join(PROD_DIR, "diagnostic-test.jsonl")
STATS_PATH = os.path.join("data", "prod_build_stats_v2.json")

SEEDS16 = [i * 0x9E3779B1 + 12345 for i in range(16)]
US_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")

# v1 build_prod_dataset.py と同等の厳格 EN benign フィルタ
WEIRD_EN_BEN_RE = re.compile(
    r"(<[a-z/!?]|\[/?[a-z_]+\]|====|####|confirmation|ignore|bypass|secret|password|admin|"
    r"assistant|system|user:|defense\.|undetected|barrel roll|test email|summarize all|confirm me|"
    r"[^\x00-\x7F\u2018-\u201F\u2026—–])",
    re.IGNORECASE,
)


def norm_text(s: str) -> str:
    """NFKC正規化・小文字化・空白除去による比較用テキスト正規化。"""
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", "", s)


def minhash16_fast(s: str) -> tuple[int, ...]:
    """5-gram ベースの 16ハッシュ MinHash 署名を計算する。"""
    nc = norm_text(s)
    if len(nc) > 1600:
        nc = nc[:800] + nc[-800:]
    if len(nc) < 5:
        h = zlib.crc32(nc.encode("utf-8"))
        return tuple((h ^ sd) & 0xFFFFFFFF for sd in SEEDS16)
    step = max(2, len(nc) // 160)
    grams = [zlib.crc32(nc[i : i + 5].encode("utf-8")) for i in range(0, len(nc) - 4, step)]
    return tuple(min(((g ^ sd) & 0xFFFFFFFF) for g in grams) for sd in SEEDS16)


def load_jsonl(path: str) -> list[dict]:
    """JSONL ファイルを全行読み込む。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: list[dict]) -> None:
    """レコード列を UTF-8 JSONL として保存する。"""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_lsh_index(rows: list[dict], max_bucket: int = 200) -> tuple[list[tuple[int, ...]], dict]:
    """既存分割に対する 4バンド×4ハッシュ LSH インデックスを構築する。"""
    sigs = [minhash16_fast(r["text"]) for r in rows]
    bands: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for b in range(4):
            bk = (b, sig[b * 4 : (b + 1) * 4])
            if len(bands[bk]) < max_bucket:
                bands[bk].append(idx)
    return sigs, bands


def has_lsh_hit(sig: tuple[int, ...], ref_sigs: list[tuple[int, ...]], bands: dict, thresh: int = 13) -> bool:
    """LSH インデックス上で一致数 >= thresh の近傍があるか判定する。"""
    cands = {cid for b in range(4) for cid in bands.get((b, sig[b * 4 : (b + 1) * 4]), ())}
    for cid in cands:
        if sum(1 for a, b in zip(sig, ref_sigs[cid]) if a == b) >= thresh:
            return True
    return False


def has_en_exact_sim(sig: tuple[int, ...], en_sigs: list[tuple[int, ...]], thresh: int = 8) -> bool:
    """既存 EN 全件（数千件規模）に対して MinHash 一致数 >= thresh を直接走査しクラスタリークを防ぐ。"""
    for rs in en_sigs:
        if sum(1 for a, b in zip(sig, rs) if a == b) >= thresh:
            return True
    return False


def extract_heldout_browsesafe_en(
    rng: random.Random,
    seen_all_norms: set[str],
    existing_en_sigs: list[tuple[int, ...]],
    existing_sigs: list[tuple[int, ...]],
    existing_bands: dict,
    need_yes: int = 50,
    need_no: int = 80,
) -> tuple[list[dict], list[dict], dict]:
    """browsesafe-bench test.parquet から既存分割と非重複のクリーン EN データを抽出する。"""
    path = os.path.join(BASE_PUB, "browsesafe-bench", "test.parquet")
    t = pq.read_table(path)
    all_rows = t.to_pylist()
    raw_yes, raw_no = [], []
    drops = Counter()

    for idx, r in enumerate(all_rows):
        html = r["content"]
        is_yes = r["label"] == "yes"
        if len(apd.RE_HIRA.findall(html)) >= 5 or len(apd.RE_KANA.findall(html)) >= 5:
            continue
        (raw_yes if is_yes else raw_no).append((idx, html, is_yes))

    rng.shuffle(raw_yes)
    rng.shuffle(raw_no)

    def _filter_group(raw_list: list[tuple], target_subcat: str, limit_scan: int = 1200) -> list[dict]:
        clean = []
        for idx, html, is_yes in raw_list[:limit_scan]:
            win = apd.clean_html_window(html, is_yes=is_yes, is_ja=False, seed_key=f"v2test_{idx}")
            if win is None or len(win) < 80:
                drops[f"bs_{target_subcat}_window_drop"] += 1
                continue
            if apd.INAPPROPRIATE_RE.search(win):
                drops[f"bs_{target_subcat}_inappropriate"] += 1
                continue
            if apd.has_regex_pii(win) or US_SSN_RE.search(win):
                drops[f"bs_{target_subcat}_pii"] += 1
                continue
            if sum(ord(c) < 128 for c in win) / max(1, len(win)) < 0.90:
                drops[f"bs_{target_subcat}_non_ascii"] += 1
                continue
            nt = norm_text(win)
            if nt in seen_all_norms:
                drops[f"bs_{target_subcat}_exact_existing"] += 1
                continue
            sig = minhash16_fast(win)
            if has_en_exact_sim(sig, existing_en_sigs, thresh=8) or has_lsh_hit(sig, existing_sigs, existing_bands, thresh=13):
                drops[f"bs_{target_subcat}_lsh_existing"] += 1
                continue
            clean.append({
                "text": win,
                "label": 1 if is_yes else 0,
                "subcategory": target_subcat,
                "source": "browsesafe_test",
                "lang": "en",
            })
        return clean

    cand_yes = _filter_group(raw_yes, "indirect", limit_scan=1200)
    cand_no = _filter_group(raw_no, "benign_long", limit_scan=1500)

    yes_cl, _ = apd.dedup_and_cluster(cand_yes, sim_dedup=12, sim_cluster=8)
    no_cl, _ = apd.dedup_and_cluster(cand_no, sim_dedup=12, sim_cluster=8)

    def _pick_cluster_diverse(items: list[dict], k: int) -> list[dict]:
        by_c = defaultdict(list)
        for it in items:
            by_c[it["cluster"]].append(it)
        keys = list(by_c.keys())
        rng.shuffle(keys)
        for ck in keys:
            rng.shuffle(by_c[ck])
        picked = []
        round_idx = 0
        while len(picked) < k:
            progressed = False
            for ck in keys:
                if round_idx < len(by_c[ck]):
                    cand = by_c[ck][round_idx]
                    picked.append(cand)
                    progressed = True
                    if len(picked) == k:
                        break
            if not progressed:
                break
            round_idx += 1
        assert len(picked) == k, f"browsesafe_test pool insufficient: need {k}, got {len(picked)}"
        return picked

    picked_yes = _pick_cluster_diverse(yes_cl, need_yes)
    picked_no = _pick_cluster_diverse(no_cl, need_no)
    return picked_yes, picked_no, dict(drops)


def extract_heldout_llmail_en(
    rng: random.Random,
    seen_all_norms: set[str],
    existing_en_sigs: list[tuple[int, ...]],
    existing_sigs: list[tuple[int, ...]],
    existing_bands: dict,
    need_atk: int = 130,
    need_ben: int = 40,
    subcat_caps: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """llmail phase1/phase2 から既存分割と非重複・非類似クラスタの EN データを抽出する。"""
    ll_dir = os.path.join(BASE_PUB, "llmail-inject-challenge", "data")
    raw_atk, raw_ben = [], []
    drops = Counter()

    for ph in ["phase1", "phase2"]:
        src = f"llmail_{ph}"
        d = json.load(open(os.path.join(ll_dir, f"labelled_unique_submissions_{ph}.json"), encoding="utf-8"))
        for raw_key, meta in d.items():
            att = str(meta.get("attack_attempt"))
            if att not in ("True", "False"):
                continue
            if len(apd.RE_HIRA.findall(raw_key)) >= 5 or len(apd.RE_KANA.findall(raw_key)) >= 5:
                continue
            if apd.INAPPROPRIATE_RE.search(raw_key) or apd.has_regex_pii(raw_key) or US_SSN_RE.search(raw_key):
                drops[f"ll_{att}_pii_or_inapp"] += 1
                continue
            txt = apd.format_email_natural(raw_key, "en")
            if apd.has_regex_pii(txt) or US_SSN_RE.search(txt) or apd.INAPPROPRIATE_RE.search(txt):
                drops[f"ll_{att}_fmt_pii_or_inapp"] += 1
                continue
            nt = norm_text(txt)
            if nt in seen_all_norms:
                drops[f"ll_{att}_exact_existing"] += 1
                continue
            if att == "False":
                if len(txt) < 120 or apd.FALSE_NEG_ATK_RE.search(raw_key) or WEIRD_EN_BEN_RE.search(txt):
                    drops["ll_False_strict_reject"] += 1
                    continue
                raw_ben.append({
                    "text": txt,
                    "dedup_key": raw_key,
                    "label": 0,
                    "subcategory": "benign",
                    "source": src,
                    "lang": "en",
                })
            else:
                if len(txt) < 50 or (sum(ord(c) < 128 for c in txt) / max(1, len(txt)) < 0.95):
                    drops["ll_True_non_ascii_or_short"] += 1
                    continue
                raw_atk.append({
                    "text": txt,
                    "dedup_key": raw_key,
                    "label": 1,
                    "subcategory": apd.classify_llmail_subcat(txt),
                    "source": src,
                    "lang": "en",
                })

    rng.shuffle(raw_atk)
    rng.shuffle(raw_ben)

    def _filter_disjoint(candidates: list[dict], target_scan: int) -> list[dict]:
        kept = []
        seen_local = set()
        for it in candidates:
            nc = norm_text(it["text"])
            if nc in seen_local:
                continue
            seen_local.add(nc)
            sig = minhash16_fast(it["text"])
            if has_en_exact_sim(sig, existing_en_sigs, thresh=8) or has_lsh_hit(sig, existing_sigs, existing_bands, thresh=13):
                drops[f"ll_lsh_existing_{it['label']}"] += 1
                continue
            kept.append(it)
            if len(kept) >= target_scan:
                break
        return kept

    filt_atk = _filter_disjoint(raw_atk, target_scan=1800)
    filt_ben = _filter_disjoint(raw_ben, target_scan=700)

    cl_atk, _ = apd.dedup_and_cluster(filt_atk, sim_dedup=12, sim_cluster=8)
    cl_ben, _ = apd.dedup_and_cluster(filt_ben, sim_dedup=12, sim_cluster=8)

    # クラスタラウンドロビン + サブカテゴリ上限付きで攻撃を抽出
    by_c_atk = defaultdict(list)
    for it in cl_atk:
        by_c_atk[it["cluster"]].append(it)
    keys_atk = list(by_c_atk.keys())
    rng.shuffle(keys_atk)
    for ck in keys_atk:
        rng.shuffle(by_c_atk[ck])

    caps = dict(subcat_caps or {})
    picked_atk = []
    sub_cnt = Counter()
    for r_i in range(4):
        for ck in keys_atk:
            if len(picked_atk) >= need_atk:
                break
            if r_i < len(by_c_atk[ck]):
                cand = by_c_atk[ck][r_i]
                sc = cand["subcategory"]
                if sc in caps and sub_cnt[sc] >= caps[sc]:
                    continue
                picked_atk.append(cand)
                sub_cnt[sc] += 1
        if len(picked_atk) >= need_atk:
            break
    assert len(picked_atk) == need_atk, f"llmail atk insufficient: {len(picked_atk)} < {need_atk}"

    by_c_ben = defaultdict(list)
    for it in cl_ben:
        by_c_ben[it["cluster"]].append(it)
    keys_ben = list(by_c_ben.keys())
    rng.shuffle(keys_ben)
    picked_ben = []
    for ck in keys_ben:
        if len(picked_ben) >= need_ben:
            break
        picked_ben.append(rng.choice(by_c_ben[ck]))
    assert len(picked_ben) == need_ben, f"llmail ben insufficient: {len(picked_ben)} < {need_ben}"

    return picked_atk, picked_ben, dict(drops)


def clean_record(r: dict) -> dict:
    """最終出力 JSONL の 5 フィールドのみへ正規化する。"""
    return {
        "text": r["text"],
        "label": int(r["label"]),
        "subcategory": r["subcategory"],
        "source": r["source"],
        "lang": r["lang"],
    }


def summarize_split(rows: list[dict]) -> dict:
    """分割ごとの統計（テキスト本文は一切含めない）を集計する。"""
    n = len(rows)
    lens = sorted(len(r["text"]) for r in rows)
    norms = {norm_text(r["text"]) for r in rows}
    sub_lbl = Counter(f"{r['subcategory']} (label={r['label']})" for r in rows)
    return {
        "total": n,
        "unique_exact_norm": len(norms),
        "dup_rate_pct": round((1.0 - len(norms) / max(1, n)) * 100, 4),
        "labels": {"benign_0": sum(1 for r in rows if r["label"] == 0), "attack_1": sum(1 for r in rows if r["label"] == 1)},
        "subcategories": dict(sorted(Counter(r["subcategory"] for r in rows).items())),
        "subcategories_with_label": dict(sorted(sub_lbl.items())),
        "sources": dict(sorted(Counter(r["source"] for r in rows).items())),
        "langs": dict(sorted(Counter(r["lang"] for r in rows).items())),
        "n_gt512chars": sum(1 for x in lens if x > 512),
        "pct_gt512chars": round(sum(1 for x in lens if x > 512) / max(1, n) * 100, 2),
        "char_len": {"min": lens[0], "p50": lens[n // 2], "p90": lens[int(n * 0.9)], "max": lens[-1]},
    }


def main():
    rng = random.Random(20261119)

    # 1. 開封済み v1 sealed-test.jsonl を診断用セット diagnostic-test.jsonl へ退避（冪等）
    if not os.path.exists(DIAG_PATH):
        assert os.path.exists(SEALED_PATH), f"Missing {SEALED_PATH}"
        shutil.copy2(SEALED_PATH, DIAG_PATH)
        print(f"Demoted v1 sealed-test -> {DIAG_PATH}", flush=True)
    else:
        print(f"Using existing diagnostic set: {DIAG_PATH}", flush=True)

    diag_rows = load_jsonl(DIAG_PATH)
    assert len(diag_rows) == 2500, f"Expected 2500 rows in diagnostic-test, got {len(diag_rows)}"

    # 2. 現行 train / dev から synth benign_long 以外を保持
    orig_train = load_jsonl(TRAIN_PATH)
    orig_dev = load_jsonl(DEV_PATH)
    assert len(orig_train) == 45000 and len(orig_dev) == 5000

    kept_train = [r for r in orig_train if not (r["subcategory"] == "benign_long" and r["source"].startswith("synth"))]
    kept_dev = [r for r in orig_dev if not (r["subcategory"] == "benign_long" and r["source"].startswith("synth"))]
    need_tr_long = 45000 - len(kept_train)
    need_dv_long = 5000 - len(kept_dev)
    assert need_tr_long == 9566 and need_dv_long == 1092, f"Unexpected kept counts: {need_tr_long}, {need_dv_long}"

    # 3. 段落プール (8文体 × 50基底 = 400、具体段落 9,900本) を構築し train/dev の benign_long を再生成
    print("Building v2 paragraph pools (8 styles x 50 templates = 400 bases)...", flush=True)
    pools, pool_stats = blp.build_paragraph_pools(seed=20261119)
    assert pool_stats["total_base_templates"] == 400
    assert pool_stats["cross_split_paragraph_exact_overlap"] == {"train_dev": 0, "train_sealed_v2": 0, "dev_sealed_v2": 0}

    seen_doc_norms = {norm_text(r["text"]) for r in kept_train + kept_dev + diag_rows}

    tr_long_docs, tr_long_stats = blp.generate_benign_long_split(
        "train", need_tr_long, pools["train"], rng, seen_doc_norms, max_docs_per_para=5
    )
    dv_long_docs, dv_long_stats = blp.generate_benign_long_split(
        "dev", need_dv_long, pools["dev"], rng, seen_doc_norms, max_docs_per_para=5
    )
    assert tr_long_stats["max_docs_per_paragraph_actual"] <= 5
    assert dv_long_stats["max_docs_per_paragraph_actual"] <= 5

    for r in tr_long_docs + dv_long_docs:
        r["source"] = "synth_ja_v2"

    train_rows = [clean_record(r) for r in kept_train + tr_long_docs]
    dev_rows = [clean_record(r) for r in kept_dev + dv_long_docs]
    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    assert len(train_rows) == 45000 and len(dev_rows) == 5000

    # 4. 既存 3 分割 (train + dev + diagnostic) の正規化セット・MinHash/LSH インデックスを作成
    existing_all = train_rows + dev_rows + diag_rows
    seen_all_norms = {norm_text(r["text"]) for r in existing_all}
    assert len(seen_all_norms) == len(existing_all) == 52500

    print("Indexing train+dev+diagnostic LSH for held-out isolation...", flush=True)
    existing_sigs, existing_bands = build_lsh_index(existing_all, max_bucket=200)
    existing_en_sigs = [existing_sigs[i] for i, r in enumerate(existing_all) if r["lang"] == "en"]

    # 5. 新しい封印テスト (2,500件) の実在保持分 (300件) を抽出
    print("Extracting clean cluster-disjoint real sources for new sealed-test...", flush=True)
    bs_yes_sv2, bs_no_sv2, bs_drops = extract_heldout_browsesafe_en(
        rng, seen_all_norms, existing_en_sigs, existing_sigs, existing_bands,
        need_yes=50, need_no=80,
    )
    ll_atk_sv2, ll_ben_sv2, ll_drops = extract_heldout_llmail_en(
        rng, seen_all_norms, existing_en_sigs, existing_sigs, existing_bands,
        need_atk=130, need_ben=40,
        subcat_caps={"indirect": 100, "override": 150, "extraction": 150, "role_hijack": 150, "fake_history": 150, "obfuscation": 150},
    )

    real_sealed = [clean_record(r) for r in (bs_yes_sv2 + bs_no_sv2 + ll_atk_sv2 + ll_ben_sv2)]
    for r in real_sealed:
        nt = norm_text(r["text"])
        assert nt not in seen_all_norms
        seen_all_norms.add(nt)
        seen_doc_norms.add(nt)

    # 6. 残り 2,200 件を v2 封印専用合成 + sealed_v2 段落プールで構築（比率: 攻撃36% / benign 20% / benign_hard 20% / benign_long 24%）
    sealed_rows = list(real_sealed)
    real_atk_counts = Counter(r["subcategory"] for r in real_sealed if r["label"] == 1)

    def _fill_synth_disjoint(need: int, gen_fn, label: int, subcat: str, source: str, lang: str, tag: str) -> None:
        added = 0
        attempts = 0
        max_attempts = max(2000, need * 60)
        while added < need and attempts < max_attempts:
            attempts += 1
            txt = gen_fn()
            nt = norm_text(txt)
            if not nt or nt in seen_all_norms:
                continue
            sig = minhash16_fast(txt)
            if has_lsh_hit(sig, existing_sigs, existing_bands, thresh=13):
                continue
            seen_all_norms.add(nt)
            seen_doc_norms.add(nt)
            sealed_rows.append({
                "text": txt,
                "label": int(label),
                "subcategory": subcat,
                "source": source,
                "lang": lang,
            })
            added += 1
        assert added == need, f"Failed to synthesize {need} items for {tag}: got {added}"

    for cat in s3.ATTACK_GENS:
        need_cat = 150 - real_atk_counts[cat]
        assert need_cat >= 0
        _fill_synth_disjoint(
            need_cat,
            lambda c=cat: s3.ATTACK_GENS[c](rng, "sealed_v2"),
            label=1, subcat=cat, source="synth_ja_v2", lang="ja", tag=f"sealed_v2_{cat}",
        )

    need_ben_ja = 500 - sum(1 for r in real_sealed if r["subcategory"] == "benign")
    _fill_synth_disjoint(
        need_ben_ja,
        lambda: s3.gen_benign_normal(rng, "sealed_v2"),
        label=0, subcat="benign", source="synth_ja_v2", lang="ja", tag="sealed_v2_benign_ja",
    )

    _fill_synth_disjoint(
        50,
        lambda: s3.gen_en_benign_hard_sealed_v2(rng),
        label=0, subcat="benign_hard", source="synth_en_v2", lang="en", tag="sealed_v2_benign_hard_en",
    )
    _fill_synth_disjoint(
        450,
        lambda: s3.gen_benign_hard(rng, "sealed_v2"),
        label=0, subcat="benign_hard", source="synth_ja_v2", lang="ja", tag="sealed_v2_benign_hard_ja",
    )

    need_sv2_long = 600 - sum(1 for r in real_sealed if r["subcategory"] == "benign_long")
    assert need_sv2_long == 520
    sv2_long_docs, sv2_long_stats = blp.generate_benign_long_split(
        "sealed_v2", need_sv2_long, pools["sealed_v2"], rng, seen_doc_norms, max_docs_per_para=5
    )
    assert sv2_long_stats["max_docs_per_paragraph_actual"] <= 5
    for r in sv2_long_docs:
        nt = norm_text(r["text"])
        assert nt not in seen_all_norms
        seen_all_norms.add(nt)
        r["source"] = "synth_ja_v2"
        sealed_rows.append(clean_record(r))

    rng.shuffle(sealed_rows)
    assert len(sealed_rows) == 2500, f"Expected 2500 sealed_rows, got {len(sealed_rows)}"

    # 7. JSONL 書き出し
    save_jsonl(TRAIN_PATH, train_rows)
    save_jsonl(DEV_PATH, dev_rows)
    save_jsonl(SEALED_PATH, sealed_rows)

    # 8. 全 4 分割・6 ペアの完全一致重複＆LSH 近傍リーク検査
    print("Running full 4-split cross-leakage exact & LSH audit (no sealed text output)...", flush=True)
    split_map = {
        "train": train_rows,
        "dev": dev_rows,
        "sealed_test": sealed_rows,
        "diagnostic_test": diag_rows,
    }
    norm_sets = {k: {norm_text(r["text"]) for r in v} for k, v in split_map.items()}
    exact_cross = {}
    keys_order = ["train", "dev", "sealed_test", "diagnostic_test"]
    for i in range(len(keys_order)):
        for j in range(i + 1, len(keys_order)):
            ka, kb = keys_order[i], keys_order[j]
            ov = len(norm_sets[ka] & norm_sets[kb])
            exact_cross[f"{ka}_vs_{kb}"] = ov
            assert ov == 0, f"Exact overlap between {ka} and {kb}: {ov}"

    sigs_by_split = {}
    bands_by_split = {}
    for k, v in split_map.items():
        sigs_by_split[k], bands_by_split[k] = build_lsh_index(v, max_bucket=200)

    def _lsh_pair(target_k: str, ref_k: str, thresh: int = 13) -> dict:
        hits = Counter()
        for idx, r in enumerate(split_map[target_k]):
            sig = sigs_by_split[target_k][idx]
            if has_lsh_hit(sig, sigs_by_split[ref_k], bands_by_split[ref_k], thresh=thresh):
                hits[r["source"]] += 1
        return {"total_hits": sum(hits.values()), "by_source": dict(sorted(hits.items()))}

    lsh_cross = {
        "dev_vs_train": _lsh_pair("dev", "train", 13),
        "sealed_test_vs_train": _lsh_pair("sealed_test", "train", 13),
        "sealed_test_vs_dev": _lsh_pair("sealed_test", "dev", 13),
        "sealed_test_vs_diagnostic_test": _lsh_pair("sealed_test", "diagnostic_test", 13),
        "diagnostic_test_vs_train": _lsh_pair("diagnostic_test", "train", 13),
        "diagnostic_test_vs_dev": _lsh_pair("diagnostic_test", "dev", 13),
    }

    stats_out = {
        "version": "v2",
        "paragraph_pool": pool_stats,
        "benign_long_generation": {
            "train": tr_long_stats,
            "dev": dv_long_stats,
            "sealed_v2": sv2_long_stats,
        },
        "heldout_real_filter_drops": {
            "browsesafe_test": bs_drops,
            "llmail_en": ll_drops,
        },
        "cross_split_exact_overlap": exact_cross,
        "cross_split_lsh_sim_ge_13_of_16": lsh_cross,
        "splits": {k: summarize_split(v) for k, v in split_map.items()},
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2)

    print("=== V2 DATASET REBUILD COMPLETE ===", flush=True)
    print(f"Stats written to {STATS_PATH}", flush=True)
    print("Cross-split exact overlap:", exact_cross, flush=True)
    print("Cross-split LSH >=13/16 totals:", {k: v["total_hits"] for k, v in lsh_cross.items()}, flush=True)


if __name__ == "__main__":
    main()
