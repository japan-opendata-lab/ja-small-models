#!/usr/bin/env python3
"""第2号 本番データセット (data/prod/train.jsonl, dev.jsonl, sealed-test.jsonl) 構築スクリプト。

DATA-BUILD-v1.md と目視監査確定ルールに完全準拠:
- llmail_ja_ben (42件) は目視監査により未遂攻撃・スパム混入のため全件除外
- llmail_en_ben は公式 emails_for_fp_tests / scenarios を最優先し、厳格フィルタ済 False + 英語合成で補完
- llmail_ja_atk は 45文字未満短文除外・巨大クラスタ cl_1261 を train 限定(上限350件)・残り230クラスタを非重複分割
- browsesafe train/test は split 分離、英語上限に合わせて総量制限
- NotInject (339件) と XSTest safe (250件) は学習に入れず、dev と sealed-test に層化折半
- 自前 3層合成は train / dev / sealed で構文骨格・ファミリーを完全分離し、重複率 0.00% を保証
- sealed-test.jsonl のテキスト本文は一切標準出力・ログに出さない
"""
import csv
import json
import os
import random
import re
import sys
import unicodedata
import zlib
from collections import Counter, defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_public_data as apd
import synth_3layer as s3

BASE_PUB = os.path.join("data", "public_sources")
OUT_DIR = os.path.join("data", "prod")
STATS_PATH = os.path.join("data", "prod_build_stats.json")


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", "", s)


SEEDS16 = [i * 0x9E3779B1 + 12345 for i in range(16)]


def minhash16_fast(s: str) -> tuple[int, ...]:
    nc = norm_text(s)
    if len(nc) > 1600:
        nc = nc[:800] + nc[-800:]
    if len(nc) < 5:
        h = zlib.crc32(nc.encode("utf-8"))
        return tuple((h ^ sd) & 0xFFFFFFFF for sd in SEEDS16)
    step = max(2, len(nc) // 160)
    grams = [zlib.crc32(nc[i : i + 5].encode("utf-8")) for i in range(0, len(nc) - 4, step)]
    return tuple(min(((g ^ sd) & 0xFFFFFFFF) for g in grams) for sd in SEEDS16)


def split_clusters_target(items: list[dict], target_counts: dict[str, int], rng: random.Random) -> dict[str, list[dict]]:
    """クラスタ単位で重複なく split ごとに配分する（split 順序どおり目標件数まで割り当て）。"""
    by_cl = defaultdict(list)
    for it in items:
        by_cl[it["cluster"]].append(it)
    cl_keys = list(by_cl.keys())
    rng.shuffle(cl_keys)

    out = {k: [] for k in target_counts}
    order = list(target_counts.keys())
    idx = 0
    for ck in cl_keys:
        # Find next split that still needs items
        assigned = False
        for offset in range(len(order)):
            sp = order[(idx + offset) % len(order)]
            if len(out[sp]) < target_counts[sp]:
                out[sp].extend(by_cl[ck])
                idx = (idx + offset + 1) % len(order)
                assigned = True
                break
        if not assigned:
            # Overflow goes to first split if unlimited, else drop
            break
    # Trim exact overflow inside each split (within same clusters, never crosses clusters!)
    for sp, cap in target_counts.items():
        if len(out[sp]) > cap:
            rng.shuffle(out[sp])
            out[sp] = out[sp][:cap]
    return out


def load_strict_llmail_en_benign(rng: random.Random) -> tuple[list[dict], dict]:
    """公式 fp_test / scenarios を優先し、phase1/2 False からは厳格基準でクリーン英語メールのみ抽出する。"""
    ll_dir = os.path.join(BASE_PUB, "llmail-inject-challenge", "data")
    official = []
    fp_emails = json.load(open(os.path.join(ll_dir, "emails_for_fp_tests.json"), encoding="utf-8"))
    for rk in fp_emails:
        if not apd.has_regex_pii(rk) and not apd.INAPPROPRIATE_RE.search(rk):
            official.append({
                "text": apd.format_email_natural(rk, "en"),
                "dedup_key": rk,
                "label": 0,
                "subcategory": "benign",
                "source": "llmail_fp_test",
                "lang": "en",
            })
    scen = json.load(open(os.path.join(ll_dir, "scenarios.json"), encoding="utf-8"))
    for sd in scen.values():
        for rk in sd.get("emails", []):
            if not apd.has_regex_pii(rk) and not apd.INAPPROPRIATE_RE.search(rk):
                official.append({
                    "text": apd.format_email_natural(rk, "en"),
                    "dedup_key": rk,
                    "label": 0,
                    "subcategory": "benign",
                    "source": "llmail_scenario",
                    "lang": "en",
                })

    strict_sub = []
    WEIRD_RE = re.compile(
        r"(<[a-z/!?]|\[/?[a-z_]+\]|====|####|confirmation|ignore|bypass|secret|password|admin|"
        r"assistant|system|user:|defense\.|undetected|barrel roll|test email|summarize all|confirm me|[^\x00-\x7F\u2018-\u201F\u2026—–])",
        re.IGNORECASE,
    )
    pool = json.load(open("data/audited_public_pool.json", encoding="utf-8"))
    dropped_extra = 0
    for it in pool["llmail_en_ben"]:
        if it["source"] in ("llmail_fp_test", "llmail_scenario"):
            continue
        if WEIRD_RE.search(it["text"]) or len(it["text"]) < 120:
            dropped_extra += 1
            continue
        it2 = dict(it)
        it2["dedup_key"] = it["text"]
        strict_sub.append(it2)

    combined, st = apd.dedup_and_cluster(official + strict_sub)
    st["manual_strict_dropped_en_false"] = dropped_extra
    return combined, st


def load_notinject_halves(rng: random.Random) -> tuple[list[dict], list[dict]]:
    """NotInject 339件をサブセット・カテゴリ層化で前半169 (dev) / 後半170 (sealed) に折半する。"""
    ni_dir = os.path.join(BASE_PUB, "notinject", "data")
    files = [
        "NotInject_one-00000-of-00001.parquet",
        "NotInject_two-00000-of-00001.parquet",
        "NotInject_three-00000-of-00001.parquet",
    ]
    rows_all = []
    for fn in files:
        t = pq.read_table(os.path.join(ni_dir, fn))
        for r in t.to_pylist():
            lang = "other" if r["category"] == "Multilingual" else "en"
            rows_all.append({
                "text": r["prompt"].strip(),
                "label": 0,
                "subcategory": "benign_hard",
                "source": "notinject",
                "lang": lang,
                "_cat": r["category"],
                "_split_src": fn,
            })
    by_grp = defaultdict(list)
    for r in rows_all:
        by_grp[(r["_split_src"], r["_cat"])].append(r)
    dev_out, sealed_out = [], []
    toggle = 0
    for gk in sorted(by_grp.keys()):
        grp = by_grp[gk]
        rng.shuffle(grp)
        for item in grp:
            item.pop("_cat", None)
            item.pop("_split_src", None)
            if toggle % 2 == 0:
                dev_out.append(item)
            else:
                sealed_out.append(item)
            toggle += 1
    # Adjust exact 169 / 170
    if len(dev_out) == 170 and len(sealed_out) == 169:
        sealed_out.append(dev_out.pop())
    assert len(dev_out) == 169 and len(sealed_out) == 170, (len(dev_out), len(sealed_out))
    return dev_out, sealed_out


def load_xstest_safe_halves(rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Paul/XSTest の safe 250件のみをタイプ層化で dev 125 / sealed 125 に折半する（unsafe は不使用）。"""
    path = os.path.join(BASE_PUB, "xstest", "xstest_prompts.csv")
    by_type = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["label"] != "safe":
                continue
            by_type[r["type"]].append({
                "text": r["prompt"].strip(),
                "label": 0,
                "subcategory": "benign_hard",
                "source": "xstest_safe",
                "lang": "en",
            })
    dev_out, sealed_out = [], []
    toggle = 0
    for tk in sorted(by_type.keys()):
        grp = by_type[tk]
        rng.shuffle(grp)
        for item in grp:
            if toggle % 2 == 0:
                dev_out.append(item)
            else:
                sealed_out.append(item)
            toggle += 1
    assert len(dev_out) == 125 and len(sealed_out) == 125, (len(dev_out), len(sealed_out))
    return dev_out, sealed_out


def main():
    rng = random.Random(20261119)
    os.makedirs(OUT_DIR, exist_ok=True)
    audit_extra_drops = Counter()

    pool = json.load(open("data/audited_public_pool.json", encoding="utf-8"))
    audit_extra_drops["manual_audit_llmail_ja_false_contaminated"] = len(pool["llmail_ja_ben"])  # 42件全除外

    # 1. llmail_ja_atk: drop <45 chars, isolate cl_1261 to train(<=350), split remaining 230 clusters across sealed/dev/train
    ja_atk_all = []
    for it in pool["llmail_ja_atk"]:
        if len(it["text"]) < 45:
            audit_extra_drops["manual_audit_llmail_ja_atk_too_short"] += 1
            continue
        ja_atk_all.append(it)

    cl1261 = [x for x in ja_atk_all if x["cluster"] == "cl_1261"]
    rng.shuffle(cl1261)
    cl1261_train = cl1261[:350]
    audit_extra_drops["llmail_ja_atk_cl1261_diversity_cap"] = max(0, len(cl1261) - len(cl1261_train))

    ja_atk_other = [x for x in ja_atk_all if x["cluster"] != "cl_1261"]
    ja_atk_split = split_clusters_target(
        ja_atk_other, {"sealed": 155, "dev": 115, "train": 9999}, rng
    )
    ll_ja_atk_train = cl1261_train + ja_atk_split["train"]
    ll_ja_atk_dev = ja_atk_split["dev"]
    ll_ja_atk_sealed = ja_atk_split["sealed"]

    # 2. llmail_en_atk: cluster split into train(1950) and dev(150) so JA synth indirect also gets ~400+ rows in train
    ll_en_atk_split = split_clusters_target(
        pool["llmail_en_atk"], {"dev": 150, "train": 1950}, rng
    )

    # 3. llmail_en_ben: official fp+scenario prioritized + strict phase1/2 False
    en_ben_pool, st_en_ben_strict = load_strict_llmail_en_benign(rng)
    audit_extra_drops["manual_audit_llmail_en_false_strict"] = st_en_ben_strict["manual_strict_dropped_en_false"]
    ll_en_ben_split = split_clusters_target(en_ben_pool, {"dev": 50, "train": 500}, rng)

    # 4. browsesafe_train_yes / no: split JA and EN separately by cluster into train / dev
    bs_tr_yes_ja = [x for x in pool["browsesafe_train_yes"] if x["lang"] == "ja"]
    bs_tr_yes_en = [x for x in pool["browsesafe_train_yes"] if x["lang"] == "en"]
    bs_tr_no_ja = [x for x in pool["browsesafe_train_no"] if x["lang"] == "ja"]
    bs_tr_no_en = [x for x in pool["browsesafe_train_no"] if x["lang"] == "en"]

    bs_yes_ja_sp = split_clusters_target(bs_tr_yes_ja, {"dev": max(1, len(bs_tr_yes_ja) // 5), "train": 9999}, rng)
    bs_yes_en_sp = split_clusters_target(bs_tr_yes_en, {"dev": 70, "train": 700}, rng)
    bs_no_ja_sp = split_clusters_target(bs_tr_no_ja, {"dev": max(1, len(bs_tr_no_ja) // 5), "train": 9999}, rng)
    bs_no_en_sp = split_clusters_target(bs_tr_no_en, {"dev": 100, "train": 1200}, rng)

    # 5. browsesafe_test_yes / no -> sealed-test ONLY
    bs_te_yes_ja = [x for x in pool["browsesafe_test_yes"] if x["lang"] == "ja"]
    bs_te_yes_en = [x for x in pool["browsesafe_test_yes"] if x["lang"] == "en"]
    bs_te_no_ja = [x for x in pool["browsesafe_test_no"] if x["lang"] == "ja"]
    bs_te_no_en = [x for x in pool["browsesafe_test_no"] if x["lang"] == "en"]
    rng.shuffle(bs_te_yes_en)
    rng.shuffle(bs_te_no_en)
    bs_yes_sealed = bs_te_yes_ja + bs_te_yes_en[:35]
    bs_no_sealed = bs_te_no_ja + bs_te_no_en[:35]

    # 6. NotInject & XSTest safe -> dev & sealed-test ONLY
    ni_dev, ni_sealed = load_notinject_halves(rng)
    xs_dev, xs_sealed = load_xstest_safe_halves(rng)

    # Global deduplication registry (ensures 0.00% exact normalized overlap across ALL 52,500 items)
    seen_global = set()
    dup_skipped = Counter()

    def try_add(target_list: list[dict], rec: dict, tag: str) -> bool:
        nc = norm_text(rec["text"])
        if not nc or nc in seen_global:
            dup_skipped[tag] += 1
            return False
        seen_global.add(nc)
        target_list.append({
            "text": rec["text"],
            "label": int(rec["label"]),
            "subcategory": rec["subcategory"],
            "source": rec["source"],
            "lang": rec["lang"],
        })
        return True

    def fill_synth(target_list: list[dict], need: int, gen_fn, label: int, subcat: str, source: str, lang: str, tag: str):
        added = 0
        attempts = 0
        max_attempts = max(1000, need * 40)
        while added < need and attempts < max_attempts:
            attempts += 1
            txt = gen_fn()
            if try_add(target_list, {"text": txt, "label": label, "subcategory": subcat, "source": source, "lang": lang}, tag):
                added += 1
        assert added == need, f"Failed to generate {need} unique items for {tag} (got {added} in {attempts} tries)"

    def build_split(
        split_name: str,
        total_n: int,
        pub_atk: list[dict],
        pub_en_ben: list[dict],
        pub_hard: list[dict],
        pub_long: list[dict],
        en_hard_n: int,
        en_ben_cap: int,
    ) -> list[dict]:
        rows = []
        target_atk = int(round(total_n * 0.36))
        target_ben = int(round(total_n * 0.20))
        target_hard = int(round(total_n * 0.20))
        target_long = total_n - target_atk - target_ben - target_hard
        assert target_atk + target_ben + target_hard + target_long == total_n

        # A. Attack (36%)
        atk_rows = []
        for r in pub_atk:
            if len(atk_rows) < target_atk:
                try_add(atk_rows, r, f"{split_name}_pub_atk")
        need_synth_atk = target_atk - len(atk_rows)
        # Distribute synthetic attacks to balance the 6 subcategories toward target_atk / 6 each
        cats = list(s3.ATTACK_GENS.keys())
        cur_cat_cnt = Counter(r["subcategory"] for r in atk_rows)
        ideal_each = target_atk / len(cats)
        deficits = {c: max(0.0, ideal_each - cur_cat_cnt[c]) for c in cats}
        tot_def = sum(deficits.values()) or 1.0
        alloc = {c: int(need_synth_atk * deficits[c] / tot_def) for c in cats}
        rem = need_synth_atk - sum(alloc.values())
        for c in sorted(cats, key=lambda k: cur_cat_cnt[k] + alloc[k]):
            if rem <= 0:
                break
            alloc[c] += 1
            rem -= 1
        for c in cats:
            if alloc[c] > 0:
                fill_synth(
                    atk_rows, alloc[c],
                    lambda cat=c: s3.ATTACK_GENS[cat](rng, split_name),
                    1, c, "synth_ja_v1", "ja", f"{split_name}_synth_{c}"
                )
        rows.extend(atk_rows)

        # B. Benign normal (20%)
        ben_rows = []
        for r in pub_en_ben:
            if len(ben_rows) < en_ben_cap:
                try_add(ben_rows, r, f"{split_name}_pub_en_ben")
        en_ben_cur = sum(1 for r in ben_rows if r["lang"] == "en")
        if en_ben_cur < en_ben_cap:
            fill_synth(
                ben_rows, en_ben_cap - en_ben_cur,
                lambda: s3.gen_en_benign_normal(rng, split_name),
                0, "benign", "synth_en_v1", "en", f"{split_name}_synth_en_benign"
            )
        fill_synth(
            ben_rows, target_ben - len(ben_rows),
            lambda: s3.gen_benign_normal(rng, split_name),
            0, "benign", "synth_ja_v1", "ja", f"{split_name}_synth_ja_benign"
        )
        rows.extend(ben_rows)

        # C. Benign hard (20%)
        hard_rows = []
        for r in pub_hard:
            try_add(hard_rows, r, f"{split_name}_pub_hard")
        if en_hard_n > 0:
            fill_synth(
                hard_rows, en_hard_n,
                lambda: s3.gen_en_benign_hard(rng),
                0, "benign_hard", "synth_en_v1", "en", f"{split_name}_synth_en_hard"
            )
        fill_synth(
            hard_rows, target_hard - len(hard_rows),
            lambda: s3.gen_benign_hard(rng, split_name),
            0, "benign_hard", "synth_ja_v1", "ja", f"{split_name}_synth_ja_hard"
        )
        rows.extend(hard_rows)

        # D. Benign long (24%)
        long_rows = []
        for r in pub_long:
            if len(long_rows) < target_long:
                try_add(long_rows, r, f"{split_name}_pub_long")
        fill_synth(
            long_rows, target_long - len(long_rows),
            lambda: s3.gen_benign_long(rng, split_name),
            0, "benign_long", "synth_ja_v1", "ja", f"{split_name}_synth_ja_long"
        )
        rows.extend(long_rows)

        rng.shuffle(rows)
        assert len(rows) == total_n
        return rows

    train_rows = build_split(
        "train", 45000,
        pub_atk=ll_ja_atk_train + bs_yes_ja_sp["train"] + ll_en_atk_split["train"] + bs_yes_en_sp["train"],
        pub_en_ben=ll_en_ben_split["train"],
        pub_hard=[],
        pub_long=bs_no_ja_sp["train"] + bs_no_en_sp["train"],
        en_hard_n=400,
        en_ben_cap=500,
    )
    dev_rows = build_split(
        "dev", 5000,
        pub_atk=ll_ja_atk_dev + bs_yes_ja_sp["dev"] + ll_en_atk_split["dev"] + bs_yes_en_sp["dev"],
        pub_en_ben=ll_en_ben_split["dev"],
        pub_hard=ni_dev + xs_dev,
        pub_long=bs_no_ja_sp["dev"] + bs_no_en_sp["dev"],
        en_hard_n=0,
        en_ben_cap=50,
    )
    sealed_rows = build_split(
        "sealed", 2500,
        pub_atk=ll_ja_atk_sealed + bs_yes_sealed,
        pub_en_ben=[],
        pub_hard=ni_sealed + xs_sealed,
        pub_long=bs_no_sealed,
        en_hard_n=0,
        en_ben_cap=0,
    )

    # Save JSONL files
    for fn, subset in [("train.jsonl", train_rows), ("dev.jsonl", dev_rows), ("sealed-test.jsonl", sealed_rows)]:
        out_p = os.path.join(OUT_DIR, fn)
        with open(out_p, "w", encoding="utf-8", newline="\n") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Cross-split near-duplicate LSH audit (stat only, never print sealed-test text!)
    print("Running cross-split exact & LSH leakage verification...", flush=True)
    tr_norms = {norm_text(r["text"]) for r in train_rows}
    dv_norms = {norm_text(r["text"]) for r in dev_rows}
    sl_norms = {norm_text(r["text"]) for r in sealed_rows}
    assert len(tr_norms) == 45000 and len(dv_norms) == 5000 and len(sl_norms) == 2500
    assert not (tr_norms & dv_norms) and not (tr_norms & sl_norms) and not (dv_norms & sl_norms)

    # Build LSH index on train_rows to count any high-sim (>=13/16) neighbors in dev or sealed
    tr_sigs = [minhash16_fast(r["text"]) for r in train_rows]
    bands = defaultdict(list)
    for idx, sig in enumerate(tr_sigs):
        for b in range(4):
            bk = (b, sig[b * 4 : (b + 1) * 4])
            if len(bands[bk]) < 50:
                bands[bk].append(idx)

    def count_lsh_hits(subset: list[dict], thresh: int = 13) -> dict:
        hits_by_src = Counter()
        for r in subset:
            sig = minhash16_fast(r["text"])
            cands = {cid for b in range(4) for cid in bands.get((b, sig[b * 4 : (b + 1) * 4]), ())}
            if any(sum(1 for a, b in zip(sig, tr_sigs[cid]) if a == b) >= thresh for cid in cands):
                hits_by_src[r["source"]] += 1
        return dict(hits_by_src)

    lsh_dev_vs_train = count_lsh_hits(dev_rows, 13)
    lsh_sealed_vs_train = count_lsh_hits(sealed_rows, 13)

    def summarize(subset: list[dict]) -> dict:
        n = len(subset)
        lens = sorted(len(r["text"]) for r in subset)
        return {
            "total": n,
            "unique_exact_norm": len({norm_text(r["text"]) for r in subset}),
            "dup_rate_pct": round((1.0 - len({norm_text(r["text"]) for r in subset}) / n) * 100, 4),
            "labels": dict(Counter(r["label"] for r in subset)),
            "subcategories": dict(sorted(Counter(r["subcategory"] for r in subset).items())),
            "sources": dict(sorted(Counter(r["source"] for r in subset).items())),
            "langs": dict(sorted(Counter(r["lang"] for r in subset).items())),
            "char_len": {"min": lens[0], "p50": lens[n // 2], "p90": lens[int(n * 0.9)], "max": lens[-1]},
        }

    stats = {
        "audit_extra_drops": dict(audit_extra_drops),
        "dup_skipped_during_build": dict(dup_skipped),
        "cross_split_exact_overlap": {"train_dev": 0, "train_sealed": 0, "dev_sealed": 0},
        "cross_split_lsh_sim_ge_13_of_16": {
            "dev_vs_train_by_source": lsh_dev_vs_train,
            "sealed_vs_train_by_source": lsh_sealed_vs_train,
        },
        "splits": {
            "train": summarize(train_rows),
            "dev": summarize(dev_rows),
            "sealed_test": summarize(sealed_rows),
        },
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=== PRODUCTION BUILD COMPLETE ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
