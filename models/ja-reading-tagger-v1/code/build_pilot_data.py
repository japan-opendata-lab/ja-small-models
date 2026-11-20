#!/usr/bin/env python
"""第9号 ja-reading-tagger-v1 パイロット4分割データ構築＆リーク完全検査スクリプト。

構築する分割（計 25,000 件）:
- data/pilot/train.jsonl          : 20,000 件
- data/pilot/dev.jsonl            :  2,000 件
- data/pilot/sealed-test.jsonl    :  2,000 件（構築・統計出力後に完全封印）
- data/pilot/homograph-eval.jsonl :  1,000 件（dev 独立の同形異音語評価セット：実文＋現代語合成）

分割・検証保証:
- 文字 5-gram MinHash LSH (実測 Jaccard >= 0.80) による Union-Find クラスタ統合
- 青空文庫の work_id（作品ID）単位グループ分割
- 4分割全6ペアで work_id_overlap == 0, group_id_overlap == 0,
  exact_norm_text_overlap == 0, near_dup_lsh_080 == 0 を assert 検証
- 全採用スパンの辞書検証および文字オフセット整合 (text[start:end] == surface) を assert 検証
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import homograph_catalog as hc  # noqa: E402
from lexicon_util import LexiconValidator  # noqa: E402
import phonology as ph  # noqa: E402
from span_extractor import SpanExtractor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MASTER_DIR = ROOT / "data" / "master"
PILOT_DIR = ROOT / "data" / "pilot"

_NORM_STRIP_RE = re.compile(r"[、。「」『』（）！？…・\s　]")
_NUM_PERM = 64
_BANDS = 16
_ROWS = 4  # 16 * 4 = 64


def norm_for_dedup(text: str) -> str:
    return _NORM_STRIP_RE.sub("", text)


def char_5grams(norm_txt: str) -> Set[str]:
    if len(norm_txt) <= 5:
        return {norm_txt}
    return {norm_txt[i : i + 5] for i in range(len(norm_txt) - 4)}


def compute_minhash(grams: Set[str]) -> Tuple[int, ...]:
    """決定的な高速 64 次元 MinHash シグネチャ（xxh/md5/crc32 ベース混合ハッシュ）。"""
    import zlib

    vals = [zlib.crc32(g.encode("utf-8")) & 0xFFFFFFFF for g in grams]
    sig = []
    # 64 個の線形合同法ハッシュ（Mersenne prime 2^31-1 風の高速 32bit 乗算混合）
    for seed in range(_NUM_PERM):
        a = (seed * 2654435761 + 1) & 0xFFFFFFFF
        b = (seed * 2246822519 + 13) & 0xFFFFFFFF
        m = min(((v * a + b) & 0xFFFFFFFF) for v in vals)
        sig.append(m)
    return tuple(sig)


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


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def load_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def annotate_extractor_info(rows: List[dict], extractor: SpanExtractor) -> None:
    """各文に対して SpanExtractor を通し、多読語の有無・実文同形異音ターゲットを特定する。"""
    catalog_keys = {item["key"] for item in hc.HOMOGRAPH_CATALOG}
    for r in rows:
        ext = extractor.extract_spans(r["text"])
        multi_ext = [sp for sp in ext if len(sp["candidates"]) >= 2]
        r["_ext_multi_count"] = len(multi_ext)
        r["_has_catalog_multi"] = any(sp["key"] in catalog_keys for sp in multi_ext)

        # homograph-eval 用ターゲット認定（明示ルビ または 合成ターゲットが SpanExtractor 多読スパンと一致）
        if r["source"] == "qwen3_synth":
            tsp = r["target_span"]
            matched = [
                sp
                for sp in multi_ext
                if sp["start"] == tsp["start"]
                and sp["end"] == tsp["end"]
                and tsp["reading_kata"] in sp["candidates"]
            ]
            if matched:
                t_copy = dict(tsp)
                t_copy["key"] = matched[0]["key"]
                r["_eval_targets"] = [t_copy]
            else:
                r["_eval_targets"] = []
        elif r["source"] == "rohan4600":
            # ROHAN は全漢字スパンが人手ルビ正解！その中で SpanExtractor 上で多読候補 >= 2 となるものをターゲットにできる
            targets = []
            gold_by_pos = {(sp["start"], sp["end"]): sp for sp in r["spans"]}
            for sp in multi_ext:
                gsp = gold_by_pos.get((sp["start"], sp["end"]))
                if gsp and gsp["reading_kata"] in sp["candidates"] and sp["key"] in catalog_keys:
                    t_copy = dict(gsp)
                    t_copy["key"] = sp["key"]
                    targets.append(t_copy)
            r["_eval_targets"] = targets
        elif r["source"] == "aozora":
            # 青空文庫は explicit_homograph_targets（底本明示ルビ付き）のみを同形異音評価ターゲットにする
            targets = []
            exp_map = {(sp["start"], sp["end"]): sp for sp in r.get("explicit_homograph_targets", [])}
            for sp in multi_ext:
                gsp = exp_map.get((sp["start"], sp["end"]))
                if gsp and gsp["reading_kata"] in sp["candidates"] and sp["key"] in catalog_keys:
                    t_copy = dict(gsp)
                    t_copy["key"] = sp["key"]
                    targets.append(t_copy)
            r["_eval_targets"] = targets
        else:
            r["_eval_targets"] = []


def round_robin_pick(
    candidates: List[dict],
    target_n: int,
    used_clusters: Set[int],
    rng: random.Random,
) -> List[dict]:
    """(key, reading_kata) の多様性を最大化するラウンドロビン抽出（同一クラスタ排他）。"""
    by_pair: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in candidates:
        t = r["_eval_targets"][0]
        by_pair[(t["key"], t["reading_kata"])].append(r)

    keys = sorted(by_pair.keys())
    for k in keys:
        rng.shuffle(by_pair[k])

    picked: List[dict] = []
    # 青空文庫などで1作品が複数件占めないよう、評価セットでは同一クラスタ最大1件または同作品上限3件に抑制
    cluster_picked: Counter = Counter()

    progress = True
    while len(picked) < target_n and progress:
        progress = False
        rng.shuffle(keys)
        for k in keys:
            if len(picked) >= target_n:
                break
            bucket = by_pair[k]
            while bucket:
                cand = bucket.pop()
                cid = cand["_cluster_id"]
                if cid in used_clusters:
                    continue
                # 評価セットの多様性のため1クラスタ（1作品）から最大2文まで
                if cluster_picked[cid] >= 2:
                    continue
                picked.append(cand)
                cluster_picked[cid] += 1
                progress = True
                break

    for p in picked:
        used_clusters.add(p["_cluster_id"])
    return picked


def pack_clusters_to_splits(
    pool_rows: List[dict],
    n_train: int,
    n_dev: int,
    n_sealed: int,
    rng: random.Random,
    max_per_cluster_train: int = 80,
    max_per_cluster_eval: int = 10,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """クラスタ境界を守りながら train / dev / sealed-test へ 10:1:1 で層化配分する。"""
    by_cluster: Dict[int, List[dict]] = defaultdict(list)
    for r in pool_rows:
        by_cluster[r["_cluster_id"]].append(r)

    clusters = list(by_cluster.items())
    # クラスタ内部をシャッフルし、同形異音語・多読語を含む文を先頭に寄せて高品質文を優先採用
    for _, items in clusters:
        rng.shuffle(items)
        items.sort(key=lambda x: (1 if x["_has_catalog_multi"] else 0, x["_ext_multi_count"]), reverse=True)
    rng.shuffle(clusters)

    train_out: List[dict] = []
    dev_out: List[dict] = []
    sealed_out: List[dict] = []

    # 各クラスタの全アイテムがどの分割に属するかを決めて割り当て（まずキャップ付きで詰め、不足なら同分割割当クラスタの残りで補充）
    assigned_clusters: Dict[str, List[List[dict]]] = {"train": [], "dev": [], "sealed": []}
    quotas = {"train": n_train, "dev": n_dev, "sealed": n_sealed}
    buckets = {"train": train_out, "dev": dev_out, "sealed": sealed_out}
    caps = {"train": max_per_cluster_train, "dev": max_per_cluster_eval, "sealed": max_per_cluster_eval}

    for cid, items in clusters:
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
            assigned_clusters[target_sp].append(rem)

    # 万一目標数に達していない場合は同一分割へ割り当て済みのクラスタの残余から補充（分割間リーク 0 維持）
    for sp in ("train", "dev", "sealed"):
        if len(buckets[sp]) < quotas[sp]:
            for rem in assigned_clusters[sp]:
                need = quotas[sp] - len(buckets[sp])
                if need <= 0:
                    break
                buckets[sp].extend(rem[:need])

    return train_out, dev_out, sealed_out


def verify_zero_leakage(
    splits: Dict[str, List[dict]],
    grams_by_idx: List[Set[str]],
    sigs_by_idx: List[Tuple[int, ...]],
) -> dict:
    """全分割の全ペア間で work_id / cluster_id / exact_norm_text / LSH Jaccard>=0.80 リークが 0 件か検査する。"""
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

            a_cids = {r["_cluster_id"] for r in a_rows}
            b_cids = {r["_cluster_id"] for r in b_rows}
            cid_ov = len(a_cids & b_cids)

            a_norms = {r["_norm_text"] for r in a_rows}
            b_norms = {r["_norm_text"] for r in b_rows}
            exact_ov = len(a_norms & b_norms)

            # LSH 近接ペア検査（a_rows の LSH バンドに登録し b_rows を照合）
            band_map: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
            for r in a_rows:
                gid = r["_global_idx"]
                sig = sigs_by_idx[gid]
                for bi in range(_BANDS):
                    band_map[(bi, sig[bi * _ROWS : (bi + 1) * _ROWS])].append(gid)

            near_dup = 0
            checked_pairs: Set[Tuple[int, int]] = set()
            for r in b_rows:
                gid_b = r["_global_idx"]
                sig_b = sigs_by_idx[gid_b]
                cands: Set[int] = set()
                for bi in range(_BANDS):
                    cands.update(band_map.get((bi, sig_b[bi * _ROWS : (bi + 1) * _ROWS]), ()))
                for gid_a in cands:
                    pair = (gid_a, gid_b)
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)
                    if jaccard(grams_by_idx[gid_a], grams_by_idx[gid_b]) >= 0.80:
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
    """内部作業フィールドを除き、全スパンの文字列位置・辞書実在を再確認して保存レコードを作る。"""
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
    if split_name == "homograph-eval" and r.get("_eval_targets"):
        out["eval_subset"] = "qwen3_synth" if r["source"] == "qwen3_synth" else "real_ruby"
        out["target_spans"] = [
            {
                "start": int(t["start"]),
                "end": int(t["end"]),
                "surface": t["surface"],
                "key": t.get("key", t["surface"]),
                "reading": ph.kata_to_hira(t["reading_kata"]),
                "reading_kata": t["reading_kata"],
                "pron": val.to_pron(t["reading_kata"], surface=t["surface"]),
            }
            for t in r["_eval_targets"]
        ]
    return out


def summarize_split(rows: List[dict], ext_rows_orig: List[dict]) -> dict:
    src_cnt = Counter(r["source"] for r in rows)
    lens = sorted(len(r["text"]) for r in rows)
    total_spans = sum(len(r["spans"]) for r in rows)
    with_multi = sum(1 for r in ext_rows_orig if r["_ext_multi_count"] >= 1)
    with_cat = sum(1 for r in ext_rows_orig if r["_has_catalog_multi"])
    res = {
        "total_rows": len(rows),
        "by_source": dict(src_cnt),
        "total_spans": total_spans,
        "with_multi_reading_span_rows": with_multi,
        "with_catalog_homograph_rows": with_cat,
        "char_len": {
            "min": lens[0] if lens else 0,
            "median": lens[len(lens) // 2] if lens else 0,
            "mean": round(sum(lens) / max(1, len(lens)), 2),
            "max": lens[-1] if lens else 0,
        },
    }
    if rows and rows[0].get("target_spans") is not None:
        t_keys = Counter()
        t_pairs = Counter()
        sub_cnt = Counter(r.get("eval_subset", "unknown") for r in rows)
        for r in rows:
            for t in r.get("target_spans", []):
                t_keys[t["key"]] += 1
                t_pairs[(t["key"], t["reading_kata"])] += 1
        res["eval_subset_counts"] = dict(sub_cnt)
        res["unique_target_keys"] = len(t_keys)
        res["unique_target_reading_pairs"] = len(t_pairs)
        res["top_target_keys"] = t_keys.most_common(15)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-dev", type=int, default=2000)
    parser.add_argument("--n-sealed", type=int, default=2000)
    parser.add_argument("--n-homo-eval", type=int, default=1000)
    parser.add_argument("--homo-real-target", type=int, default=450)
    args = parser.parse_args()

    t0 = time.time()
    rng = random.Random(args.seed)
    PILOT_DIR.mkdir(parents=True, exist_ok=True)

    rohan_rows = load_jsonl(RAW_DIR / "rohan_clean.jsonl")
    aozora_rows = load_jsonl(RAW_DIR / "aozora_clean.jsonl")
    qwen_rows = load_jsonl(RAW_DIR / "qwen3_verified.jsonl")
    print(f"Loaded raw: rohan={len(rohan_rows)}, aozora={len(aozora_rows)}, qwen3={len(qwen_rows)}")

    extractor = SpanExtractor(MASTER_DIR)
    val = LexiconValidator(MASTER_DIR)

    all_raw = rohan_rows + aozora_rows + qwen_rows
    # 正規化テキストで完全重複排除（実文優先：rohan / aozora -> qwen3 の順）
    deduped: List[dict] = []
    seen_norm: Set[str] = set()
    for r in all_raw:
        nt = norm_for_dedup(r["text"])
        if not nt or nt in seen_norm:
            continue
        seen_norm.add(nt)
        r["_norm_text"] = nt
        deduped.append(r)
    print(f"After exact norm dedup: {len(deduped)}")

    # SpanExtractor メタ情報・ターゲット判定を付与
    annotate_extractor_info(deduped, extractor)

    # 5-gram MinHash LSH + work_id による Union-Find クラスタリング
    n_total = len(deduped)
    uf = UnionFind(n_total)
    grams_list: List[Set[str]] = []
    sigs_list: List[Tuple[int, ...]] = []
    work_first_idx: Dict[str, int] = {}
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)

    for idx, r in enumerate(deduped):
        r["_global_idx"] = idx
        grams = char_5grams(r["_norm_text"])
        sig = compute_minhash(grams)
        grams_list.append(grams)
        sigs_list.append(sig)

        wid = r.get("work_id")
        if wid:
            if wid in work_first_idx:
                uf.union(work_first_idx[wid], idx)
            else:
                work_first_idx[wid] = idx

        for bi in range(_BANDS):
            buckets[(bi, sig[bi * _ROWS : (bi + 1) * _ROWS])].append(idx)

    # LSH 候補ペアの Jaccard >= 0.80 を統合
    lsh_merged = 0
    for idxs in buckets.values():
        if len(idxs) < 2 or len(idxs) > 100:
            continue
        base = idxs[0]
        for other in idxs[1:]:
            if uf.find(base) != uf.find(other):
                if jaccard(grams_list[base], grams_list[other]) >= 0.80:
                    uf.union(base, other)
                    lsh_merged += 1

    for idx, r in enumerate(deduped):
        r["_cluster_id"] = uf.find(idx)
    print(f"Clustering complete in {time.time()-t0:.1f}s (LSH merges={lsh_merged})")

    used_clusters: Set[int] = set()

    # 1. homograph-eval (1,000件) の構築
    real_homo_cands = [
        r
        for r in deduped
        if r["source"] in ("rohan4600", "aozora") and len(r["_eval_targets"]) >= 1
    ]
    synth_homo_cands = [
        r for r in deduped if r["source"] == "qwen3_synth" and len(r["_eval_targets"]) >= 1
    ]
    n_real_eval = min(args.homo_real_target, len(real_homo_cands))
    n_synth_eval = args.n_homo_eval - n_real_eval

    picked_real_eval = round_robin_pick(real_homo_cands, n_real_eval, used_clusters, rng)
    picked_synth_eval = round_robin_pick(synth_homo_cands, n_synth_eval, used_clusters, rng)
    homo_eval_orig = picked_real_eval + picked_synth_eval
    rng.shuffle(homo_eval_orig)
    print(
        f"homograph-eval picked: total={len(homo_eval_orig)} "
        f"(real={len(picked_real_eval)}, synth={len(picked_synth_eval)})"
    )

    # 2. 残りクラスタから train / dev / sealed-test をソース別に 10:1:1 で層化構築
    rem_rohan = [
        r for r in deduped if r["source"] == "rohan4600" and r["_cluster_id"] not in used_clusters
    ]
    rem_qwen = [
        r for r in deduped if r["source"] == "qwen3_synth" and r["_cluster_id"] not in used_clusters
    ]
    rem_aozora = [
        r for r in deduped if r["source"] == "aozora" and r["_cluster_id"] not in used_clusters
    ]

    # 目標配分計算（rohan ~13%, qwen ~17-20% 在庫に応じ, 残りを aozora で埋める）
    total_main = args.n_train + args.n_dev + args.n_sealed  # 24,000
    rohan_total_use = min(len(rem_rohan), 3120)  # 2600 + 260 + 260
    r_tr = int(round(rohan_total_use * (args.n_train / total_main)))
    r_dv = int(round(rohan_total_use * (args.n_dev / total_main)))
    r_se = rohan_total_use - r_tr - r_dv

    qwen_total_use = min(len(rem_qwen), 4320)  # 最大 3600 + 360 + 360
    q_tr = int(round(qwen_total_use * (args.n_train / total_main)))
    q_dv = int(round(qwen_total_use * (args.n_dev / total_main)))
    q_se = qwen_total_use - q_tr - q_dv

    tr_r, dv_r, se_r = pack_clusters_to_splits(rem_rohan, r_tr, r_dv, r_se, rng, max_per_cluster_train=5, max_per_cluster_eval=3)
    tr_q, dv_q, se_q = pack_clusters_to_splits(rem_qwen, q_tr, q_dv, q_se, rng, max_per_cluster_train=20, max_per_cluster_eval=4)

    a_tr = args.n_train - len(tr_r) - len(tr_q)
    a_dv = args.n_dev - len(dv_r) - len(dv_q)
    a_se = args.n_sealed - len(se_r) - len(se_q)

    tr_a, dv_a, se_a = pack_clusters_to_splits(
        rem_aozora, a_tr, a_dv, a_se, rng, max_per_cluster_train=220, max_per_cluster_eval=120
    )

    train_orig = tr_r + tr_q + tr_a
    dev_orig = dv_r + dv_q + dv_a
    sealed_orig = se_r + se_q + se_a
    rng.shuffle(train_orig)
    rng.shuffle(dev_orig)
    rng.shuffle(sealed_orig)

    assert len(train_orig) == args.n_train, f"train件数不足: {len(train_orig)} != {args.n_train}"
    assert len(dev_orig) == args.n_dev, f"dev件数不足: {len(dev_orig)} != {args.n_dev}"
    assert len(sealed_orig) == args.n_sealed, f"sealed件数不足: {len(sealed_orig)} != {args.n_sealed}"
    assert len(homo_eval_orig) == args.n_homo_eval, f"homo-eval件数不足: {len(homo_eval_orig)} != {args.n_homo_eval}"

    # 3. 4分割全6ペアのリーク検査
    splits_orig = {
        "train": train_orig,
        "dev": dev_orig,
        "sealed-test": sealed_orig,
        "homograph-eval": homo_eval_orig,
    }
    leak_report = verify_zero_leakage(splits_orig, grams_list, sigs_list)
    print("Leakage verification passed (all 6 pairs = 0)!")

    # 4. 全スパンオフセット・辞書検証＆書き出し
    clean_splits: Dict[str, List[dict]] = {}
    split_stats: Dict[str, dict] = {}
    for sp_name, rows in splits_orig.items():
        cleaned = [clean_row_for_save(r, sp_name, val) for r in rows]
        clean_splits[sp_name] = cleaned
        split_stats[sp_name] = summarize_split(cleaned, rows)
        out_path = PILOT_DIR / f"{sp_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in cleaned:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    val.close()

    build_stats = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "raw_counts": {
            "rohan4600": len(rohan_rows),
            "aozora": len(aozora_rows),
            "qwen3_synth": len(qwen_rows),
            "after_exact_norm_dedup": len(deduped),
        },
        "leakage_checks": leak_report,
        "splits": split_stats,
        "sealed_test_policy": "封印テスト (sealed-test.jsonl) は構築・統計確定後に完全封印し、パイロット工程では一切開封・参照・評価しない。",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(PILOT_DIR / "build-stats.json", "w", encoding="utf-8") as f:
        json.dump(build_stats, f, ensure_ascii=False, indent=2)

    print("Build pilot data complete! Stats saved to data/pilot/build-stats.json")
    for k, v in split_stats.items():
        print(f"  {k}: {v['total_rows']} rows, sources={v['by_source']}, catalog_homo={v['with_catalog_homograph_rows']}")


if __name__ == "__main__":
    main()
