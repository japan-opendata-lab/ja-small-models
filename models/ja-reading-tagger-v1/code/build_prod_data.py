#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 本番3分割データ構築＆リーク完全検査スクリプト。

構築する分割（計 110,000 件、パイロットと完全非重複）:
- data/prod/train.jsonl       : 100,000 件
- data/prod/dev.jsonl         :   5,000 件
- data/prod/sealed-test.jsonl :   5,000 件（構築・統計出力後に完全封印）

維持資産（変更しない、リーク検査対象として参照のみ）:
- data/pilot/homograph-eval.jsonl : 1,000 件

保証:
- パイロット既取得 293 作品 ID、および pilot/train, pilot/dev, pilot/homograph-eval との
  完全重複・LSH 近接重複 (5-gram Jaccard >= 0.80) を事前排除
- 低頻度一般多読語 (extra_multi_surfaces) およびカタログ同形異音語を優先濃縮
- [prod_train, prod_dev, prod_sealed-test, pilot_homograph-eval] 全6ペアで
  work_id / cluster_id / exact_norm / LSH>=0.80 すべて 0 を assert 検証
- 全 110,000 件・全漢字スパンの辞書検証およびオフセット整合を assert 検証
"""
import argparse
import json
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import homograph_catalog as hc  # noqa: E402
from lexicon_util import LexiconValidator  # noqa: E402
import phonology as ph  # noqa: E402
from span_extractor import SpanExtractor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PILOT_DIR = ROOT / "data" / "pilot"
PILOT_WORKS_DIR = ROOT / "data" / "raw" / "aozora" / "works"
MASTER_DIR = ROOT / "data" / "master"
PROD_DIR = ROOT / "data" / "prod"

_NORM_STRIP_RE = re.compile(r"[、。「」『』（）！？…・\s　]")
_NUM_PERM = 64
_BANDS = 16
_ROWS = 4


def norm_for_dedup(text: str) -> str:
    return _NORM_STRIP_RE.sub("", text)


def char_5grams(norm_txt: str) -> Set[str]:
    if len(norm_txt) <= 5:
        return {norm_txt}
    return {norm_txt[i : i + 5] for i in range(len(norm_txt) - 4)}


def compute_minhash(grams: Set[str]) -> Tuple[int, ...]:
    vals = [zlib.crc32(g.encode("utf-8")) & 0xFFFFFFFF for g in grams]
    sig = []
    for seed in range(_NUM_PERM):
        a = (seed * 2654435761 + 1) & 0xFFFFFFFF
        b = (seed * 2246822519 + 13) & 0xFFFFFFFF
        sig.append(min(((v * a + b) & 0xFFFFFFFF) for v in vals))
    return tuple(sig)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb


def load_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_pilot_exclusion_index(paths: List[Path]) -> Tuple[Set[str], List[Set[str]], Dict]:
    exact_norms: Set[str] = set()
    grams_list: List[Set[str]] = []
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)

    for p in paths:
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                nt = norm_for_dedup(json.loads(line)["text"])
                if not nt or nt in exact_norms:
                    continue
                exact_norms.add(nt)
                g = char_5grams(nt)
                sig = compute_minhash(g)
                idx = len(grams_list)
                grams_list.append(g)
                for bi in range(_BANDS):
                    buckets[(bi, sig[bi * _ROWS : (bi + 1) * _ROWS])].append(idx)
    return exact_norms, grams_list, buckets


def hits_exclusion(
    nt: str,
    grams: Set[str],
    sig: Tuple[int, ...],
    excl_norms: Set[str],
    excl_grams: List[Set[str]],
    excl_buckets: Dict,
) -> bool:
    if nt in excl_norms:
        return True
    cands: Set[int] = set()
    for bi in range(_BANDS):
        cands.update(excl_buckets.get((bi, sig[bi * _ROWS : (bi + 1) * _ROWS]), ()))
    for idx in cands:
        if jaccard(grams, excl_grams[idx]) >= 0.80:
            return True
    return False


def pack_clusters_prioritized(
    pool_rows: List[dict],
    n_train: int,
    n_dev: int,
    n_sealed: int,
    extra_rarity_scores: Dict[str, float],
    rng: random.Random,
    max_per_cluster_train: int = 360,
    max_per_cluster_eval: int = 120,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """低頻度一般多読語・カタログ同形異音語を優先しながら、クラスタ単位で 20:1:1 配分する。"""
    by_cluster: Dict[int, List[dict]] = defaultdict(list)
    for r in pool_rows:
        by_cluster[r["_cluster_id"]].append(r)

    def row_priority(r: dict) -> Tuple[float, int, int]:
        # 低頻度多読語の希少性スコア合計 + カタログ含有ボーナス
        extras = r.get("extra_multi_surfaces", [])
        rarity = sum(extra_rarity_scores.get(k, 0.0) for k in extras)
        has_cat = 1 if r.get("homograph_surfaces") else 0
        has_exp_ruby = 1 if r.get("has_explicit_homograph_ruby") else 0
        return (rarity, has_cat + has_exp_ruby, len(extras))

    clusters = list(by_cluster.items())
    for _, items in clusters:
        rng.shuffle(items)
        items.sort(key=row_priority, reverse=True)
    rng.shuffle(clusters)

    train_out: List[dict] = []
    dev_out: List[dict] = []
    sealed_out: List[dict] = []
    assigned_rem: Dict[str, List[List[dict]]] = {"train": [], "dev": [], "sealed": []}
    quotas = {"train": n_train, "dev": n_dev, "sealed": n_sealed}
    buckets = {"train": train_out, "dev": dev_out, "sealed": sealed_out}
    caps = {"train": max_per_cluster_train, "dev": max_per_cluster_eval, "sealed": max_per_cluster_eval}

    for _, items in clusters:
        avail = [sp for sp in ("train", "dev", "sealed") if len(buckets[sp]) < quotas[sp]]
        if not avail:
            break
        target_sp = max(
            avail, key=lambda sp: (quotas[sp] - len(buckets[sp])) / max(1, quotas[sp])
        )
        need = quotas[target_sp] - len(buckets[target_sp])
        cap = min(need, caps[target_sp])
        take = items[:cap]
        rem = items[cap:]
        buckets[target_sp].extend(take)
        if rem:
            assigned_rem[target_sp].append(rem)

    for sp in ("train", "dev", "sealed"):
        if len(buckets[sp]) < quotas[sp]:
            for rem in assigned_rem[sp]:
                need = quotas[sp] - len(buckets[sp])
                if need <= 0:
                    break
                buckets[sp].extend(rem[:need])

    return train_out, dev_out, sealed_out


def verify_zero_leakage(
    splits: Dict[str, List[dict]],
    grams_by_uid: Dict[str, Set[str]],
    sigs_by_uid: Dict[str, Tuple[int, ...]],
) -> dict:
    """4集合 (prod 3分割 + pilot homograph-eval) の全6ペアでリーク 0 を検証する。"""
    names = list(splits.keys())
    report = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a_rows, b_rows = splits[a_name], splits[b_name]
            pair_key = f"{a_name}__{b_name}"

            a_works = {r["work_id"] for r in a_rows if r.get("work_id")}
            b_works = {r["work_id"] for r in b_rows if r.get("work_id")}
            work_ov = len(a_works & b_works)

            a_cids = {str(r["_cluster_id"]) for r in a_rows}
            b_cids = {str(r["_cluster_id"]) for r in b_rows}
            cid_ov = len(a_cids & b_cids)

            a_norms = {r["_norm_text"] for r in a_rows}
            b_norms = {r["_norm_text"] for r in b_rows}
            exact_ov = len(a_norms & b_norms)

            band_map: Dict[Tuple[int, Tuple[int, ...]], List[str]] = defaultdict(list)
            for r in a_rows:
                uid = r["_uid"]
                sig = sigs_by_uid[uid]
                for bi in range(_BANDS):
                    band_map[(bi, sig[bi * _ROWS : (bi + 1) * _ROWS])].append(uid)

            near_dup = 0
            checked: Set[Tuple[str, str]] = set()
            for r in b_rows:
                uid_b = r["_uid"]
                sig_b = sigs_by_uid[uid_b]
                cands: Set[str] = set()
                for bi in range(_BANDS):
                    cands.update(band_map.get((bi, sig_b[bi * _ROWS : (bi + 1) * _ROWS]), ()))
                for uid_a in cands:
                    pair = (uid_a, uid_b)
                    if pair in checked:
                        continue
                    checked.add(pair)
                    if jaccard(grams_by_uid[uid_a], grams_by_uid[uid_b]) >= 0.80:
                        near_dup += 1

            report[pair_key] = {
                "work_id_overlap": work_ov,
                "cluster_id_overlap": cid_ov,
                "exact_norm_text_overlap": exact_ov,
                "near_dup_lsh_080": near_dup,
            }
            assert work_ov == 0, f"リーク検出 ({pair_key}): work_id_overlap={work_ov}"
            assert cid_ov == 0, f"リーク検出 ({pair_key}): cluster_id_overlap={cid_ov}"
            assert exact_ov == 0, f"リーク検出 ({pair_key}): exact_norm_text_overlap={exact_ov}"
            assert near_dup == 0, f"リーク検出 ({pair_key}): near_dup_lsh_080={near_dup}"
    return report


def clean_row_for_save(r: dict, split_name: str, val: LexiconValidator) -> dict:
    text = r["text"]
    clean_spans = []
    for sp in r["spans"]:
        st, ed, surf = int(sp["start"]), int(sp["end"]), sp["surface"]
        assert text[st:ed] == surf, f"オフセット不一致: {text[st:ed]!r} != {surf!r}"
        rk = ph.normalize_kana(sp.get("reading_kata") or sp["reading"])
        lead = text[max(0, st - 2) : st]
        trail = text[ed : ed + 4]
        assert val.verify_span_reading(surf, rk, trailing_text=trail, leading_text=lead), (
            f"辞書未検証スパン検出: {surf} / {rk} in {text}"
        )
        clean_spans.append(
            {
                "start": st,
                "end": ed,
                "surface": surf,
                "reading": ph.kata_to_hira(rk),
                "reading_kata": rk,
                "pron": val.to_pron(rk, surface=surf),
            }
        )
    out = {
        "id": r["id"],
        "split": split_name,
        "source": r["source"],
        "license": r["license"],
        "group_id": str(r["_cluster_id"]),
        "text": text,
        "spans": clean_spans,
    }
    if r.get("work_id"):
        out["work_id"] = r["work_id"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-train", type=int, default=100000)
    parser.add_argument("--n-dev", type=int, default=5000)
    parser.add_argument("--n-sealed", type=int, default=5000)
    args = parser.parse_args()

    t0 = time.time()
    rng = random.Random(args.seed)
    PROD_DIR.mkdir(parents=True, exist_ok=True)

    pilot_excluded_works = {p.stem for p in PILOT_WORKS_DIR.glob("*.zip")}
    excl_norms, excl_grams, excl_buckets = build_pilot_exclusion_index(
        [
            PILOT_DIR / "train.jsonl",
            PILOT_DIR / "dev.jsonl",
            PILOT_DIR / "homograph-eval.jsonl",
        ]
    )
    print(f"Pilot exclusion index ready: works={len(pilot_excluded_works)}, texts={len(excl_norms)}", flush=True)

    aozora_raw = load_jsonl(RAW_DIR / "aozora_prod_clean.jsonl")
    qwen_raw = load_jsonl(RAW_DIR / "qwen3_prod_verified.jsonl")
    print(f"Loaded prod raw: aozora={len(aozora_raw)}, qwen3={len(qwen_raw)}", flush=True)

    # 重複・パイロット公開分割との非重複フィルタ（実文先、合成後）
    all_raw = aozora_raw + qwen_raw
    deduped: List[dict] = []
    seen_norm: Set[str] = set()
    grams_by_uid: Dict[str, Set[str]] = {}
    sigs_by_uid: Dict[str, Tuple[int, ...]] = {}
    excluded_overlap_cnt = 0

    for idx, r in enumerate(all_raw):
        wid = r.get("work_id")
        if wid and wid in pilot_excluded_works:
            excluded_overlap_cnt += 1
            continue
        nt = norm_for_dedup(r["text"])
        if not nt or nt in seen_norm:
            continue
        grams = char_5grams(nt)
        sig = compute_minhash(grams)
        if hits_exclusion(nt, grams, sig, excl_norms, excl_grams, excl_buckets):
            excluded_overlap_cnt += 1
            continue
        seen_norm.add(nt)
        uid = f"prod_{idx}"
        r["_uid"] = uid
        r["_norm_text"] = nt
        grams_by_uid[uid] = grams
        sigs_by_uid[uid] = sig
        deduped.append(r)

    print(f"After pilot exclusion & exact dedup: {len(deduped)} (excluded={excluded_overlap_cnt})", flush=True)

    # クラスタリング (work_id + LSH Jaccard >= 0.80 Union-Find)
    n_tot = len(deduped)
    uf = UnionFind(n_tot)
    work_first: Dict[str, int] = {}
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)

    for idx, r in enumerate(deduped):
        wid = r.get("work_id")
        if wid:
            if wid in work_first:
                uf.union(work_first[wid], idx)
            else:
                work_first[wid] = idx
        sig = sigs_by_uid[r["_uid"]]
        for bi in range(_BANDS):
            buckets[(bi, sig[bi * _ROWS : (bi + 1) * _ROWS])].append(idx)

    lsh_merges = 0
    for idxs in buckets.values():
        if len(idxs) < 2 or len(idxs) > 150:
            continue
        base = idxs[0]
        gb = grams_by_uid[deduped[base]["_uid"]]
        for other in idxs[1:]:
            if uf.find(base) != uf.find(other):
                if jaccard(gb, grams_by_uid[deduped[other]["_uid"]]) >= 0.80:
                    uf.union(base, other)
                    lsh_merges += 1

    for idx, r in enumerate(deduped):
        r["_cluster_id"] = uf.find(idx)
    print(f"Prod clustering done in {time.time()-t0:.1f}s (LSH merges={lsh_merges})", flush=True)

    # 低頻度一般多読語 (extra_multi_surfaces) の逆頻度スコア計算
    extra_freq = Counter()
    for r in deduped:
        for k in r.get("extra_multi_surfaces", []):
            extra_freq[k] += 1
    extra_rarity_scores = {
        k: 1.0 / (cnt ** 0.5) for k, cnt in extra_freq.items()
    }

    qwen_pool = [r for r in deduped if r["source"] == "qwen3_synth"]
    aozora_pool = [r for r in deduped if r["source"] == "aozora"]

    total_target = args.n_train + args.n_dev + args.n_sealed  # 110,000
    q_use = min(len(qwen_pool), 5500)
    q_tr = int(round(q_use * (args.n_train / total_target)))
    q_dv = int(round(q_use * (args.n_dev / total_target)))
    q_se = q_use - q_tr - q_dv

    tr_q, dv_q, se_q = pack_clusters_prioritized(
        qwen_pool, q_tr, q_dv, q_se, extra_rarity_scores, rng, max_per_cluster_train=25, max_per_cluster_eval=4
    )

    a_tr = args.n_train - len(tr_q)
    a_dv = args.n_dev - len(dv_q)
    a_se = args.n_sealed - len(se_q)

    tr_a, dv_a, se_a = pack_clusters_prioritized(
        aozora_pool, a_tr, a_dv, a_se, extra_rarity_scores, rng, max_per_cluster_train=380, max_per_cluster_eval=150
    )

    train_orig = tr_q + tr_a
    dev_orig = dv_q + dv_a
    sealed_orig = se_q + se_a
    rng.shuffle(train_orig)
    rng.shuffle(dev_orig)
    rng.shuffle(sealed_orig)

    assert len(train_orig) == args.n_train, f"train件数不足: {len(train_orig)}"
    assert len(dev_orig) == args.n_dev, f"dev件数不足: {len(dev_orig)}"
    assert len(sealed_orig) == args.n_sealed, f"sealed件数不足: {len(sealed_orig)}"

    # pilot homograph-eval を読み込んで 4集合全6ペアのリーク検査を実施
    homo_eval_rows = load_jsonl(PILOT_DIR / "homograph-eval.jsonl")
    for idx, r in enumerate(homo_eval_rows):
        uid = f"homo_{idx}"
        nt = norm_for_dedup(r["text"])
        g = char_5grams(nt)
        r["_uid"] = uid
        r["_norm_text"] = nt
        r["_cluster_id"] = f"pilot_homo_{r.get('group_id', idx)}"
        grams_by_uid[uid] = g
        sigs_by_uid[uid] = compute_minhash(g)

    leak_report = verify_zero_leakage(
        {
            "prod_train": train_orig,
            "prod_dev": dev_orig,
            "prod_sealed-test": sealed_orig,
            "pilot_homograph-eval": homo_eval_rows,
        },
        grams_by_uid,
        sigs_by_uid,
    )
    print("Leakage check passed (all 6 pairs = 0)!", flush=True)

    # パイロット公開分割との重複ゼロも最終確認
    prod_all = train_orig + dev_orig + sealed_orig
    assert all(r.get("work_id") not in pilot_excluded_works for r in prod_all), "パイロット作品混入"
    assert all(r["_norm_text"] not in excl_norms for r in prod_all), "パイロット文完全重複"

    # 辞書検証＆書き出し
    val = LexiconValidator(MASTER_DIR)
    # 77万スパンを高速検証するためにキャッシュ拡張
    split_stats: Dict[str, dict] = {}
    for sp_name, orig_rows in [
        ("train", train_orig),
        ("dev", dev_orig),
        ("sealed-test", sealed_orig),
    ]:
        cleaned = [clean_row_for_save(r, sp_name, val) for r in orig_rows]
        out_path = PROD_DIR / f"{sp_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in cleaned:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        src_cnt = Counter(r["source"] for r in cleaned)
        lens = sorted(len(r["text"]) for r in cleaned)
        with_homo = sum(1 for r in orig_rows if r.get("homograph_surfaces"))
        with_extra = sum(1 for r in orig_rows if r.get("extra_multi_surfaces"))
        split_stats[sp_name] = {
            "total_rows": len(cleaned),
            "by_source": dict(src_cnt),
            "total_spans": sum(len(r["spans"]) for r in cleaned),
            "with_catalog_homograph_rows": with_homo,
            "with_extra_multi_rows": with_extra,
            "unique_works": len({r["work_id"] for r in cleaned if r.get("work_id")}),
            "char_len": {
                "min": lens[0],
                "median": lens[len(lens) // 2],
                "mean": round(sum(lens) / len(lens), 2),
                "max": lens[-1],
            },
        }
    val.close()

    build_stats = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "pilot_works_excluded": len(pilot_excluded_works),
        "raw_counts": {
            "aozora_prod": len(aozora_raw),
            "qwen3_prod": len(qwen_raw),
            "after_dedup_and_pilot_exclusion": len(deduped),
        },
        "leakage_checks": leak_report,
        "splits": split_stats,
        "sealed_test_policy": "data/prod/sealed-test.jsonl は構築・統計確定後に完全封印し、最良モデル確定後の最終評価1回限りのみ使用する。",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(PROD_DIR / "build-stats.json", "w", encoding="utf-8") as f:
        json.dump(build_stats, f, ensure_ascii=False, indent=2)

    print("Prod build complete! Stats saved to data/prod/build-stats.json", flush=True)
    for k, v in split_stats.items():
        print(
            f"  {k}: {v['total_rows']} rows, sources={v['by_source']}, "
            f"cat_homo={v['with_catalog_homograph_rows']}, extra_multi={v['with_extra_multi_rows']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
