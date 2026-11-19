#!/usr/bin/env python3
"""第4号 ja-doc-classify-v1 パイロット用データ構築。

Nemotron-PII Japan の data.parquet から、種別で層化した文書単位の分割を作る。
出力: train.jsonl / dev.jsonl / sealed-test.jsonl / build-stats.json
  各行: {"text", "domain", "document_type", "document_format"}
検査: 完全一致の重複除去、文字5-gram の MinHash LSH で分割間の
      クロスリーク（類似 0.81 以上）が 0 件であることを確認する。
あわせて risk_tiers_v0.csv の domain 欄と実データの突き合わせ結果を出す。

使い方:
  python build_cls_data.py --parquet data.parquet --out data/pilot
"""
import argparse, hashlib, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path


def shingles(text, n=5):
    return {text[i:i + n] for i in range(max(0, len(text) - n + 1))}


def minhash_coeffs(num_perm=64, seed=42):
    """MinHash 用の線形ハッシュ係数テーブルを生成する。"""
    rng = random.Random(seed)
    m = (1 << 61) - 1
    return m, [(rng.randrange(1, m), rng.randrange(0, m)) for _ in range(num_perm)]


def minhash(sig_shingles, num_perm=64, seed=42, coeffs=None, m=(1 << 61) - 1):
    if coeffs is None:
        m, coeffs = minhash_coeffs(num_perm=num_perm, seed=seed)
    sig = [m] * len(coeffs)
    for sh in sig_shingles:
        h = int(hashlib.md5(sh.encode()).hexdigest()[:16], 16)
        for i, (a, b) in enumerate(coeffs):
            v = (a * h + b) % m
            if v < sig[i]:
                sig[i] = v
    return sig


def cluster_near_dups(rows, bands=16, rows_per=4, threshold=0.81):
    """MinHash LSH + Jaccard >= threshold で近傍重複の連結成分を返す。"""
    n = len(rows)
    parent = list(range(n))
    size = [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]

    for b in range(bands):
        buckets = defaultdict(list)
        s_idx, e_idx = b * rows_per, (b + 1) * rows_per
        for i, r in enumerate(rows):
            buckets[tuple(r["_sig"][s_idx:e_idx])].append(i)
        for idxs in buckets.values():
            if len(idxs) < 2:
                continue
            for p in range(len(idxs)):
                i = idxs[p]
                sh_i = rows[i]["_sh"]
                for q in range(p + 1, len(idxs)):
                    j = idxs[q]
                    if find(i) == find(j):
                        continue
                    sh_j = rows[j]["_sh"]
                    union_len = len(sh_i | sh_j)
                    if union_len and len(sh_i & sh_j) / union_len >= threshold:
                        union(i, j)

    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    return list(comps.values())


def lsh_pairs(rows_a, rows_b, bands=16, rows_per=4):
    """類似候補ペアの数を数える（バンド一致を候補とし、Jaccard で確定）。"""
    index = defaultdict(list)
    for j, r in enumerate(rows_b):
        sig = r["_sig"]
        for b in range(bands):
            key = (b, tuple(sig[b * rows_per:(b + 1) * rows_per]))
            index[key].append(j)
    hits = 0
    for r in rows_a:
        cand = set()
        sig = r["_sig"]
        for b in range(bands):
            key = (b, tuple(sig[b * rows_per:(b + 1) * rows_per]))
            cand.update(index.get(key, []))
        for j in cand:
            inter = len(r["_sh"] & rows_b[j]["_sh"])
            union = len(r["_sh"] | rows_b[j]["_sh"])
            if union and inter / union >= 0.81:
                hits += 1
                break
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", default="data/pilot")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    t = pq.read_table(args.parquet,
                      columns=["text", "domain", "document_type",
                               "document_format"]).to_pylist()
    # 完全一致の重複除去
    seen, rows = set(), []
    for r in t:
        h = hashlib.md5(r["text"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        rows.append(r)
    m, coeffs = minhash_coeffs(num_perm=64, seed=42)
    for r in rows:
        r["_sh"] = shingles(r["text"])
        r["_sig"] = minhash(r["_sh"], coeffs=coeffs, m=m)
    clusters = cluster_near_dups(rows, bands=16, rows_per=4, threshold=0.81)
    rng = random.Random(args.seed)
    by_type = defaultdict(list)
    multi_doc_clusters = 0
    multi_doc_rows = 0
    max_cluster_size = 0
    cross_type_clusters = 0
    for comp in clusters:
        clen = len(comp)
        if clen > 1:
            multi_doc_clusters += 1
            multi_doc_rows += clen
        if clen > max_cluster_size:
            max_cluster_size = clen
        tc = Counter(rows[i]["document_type"] for i in comp)
        if len(tc) > 1:
            cross_type_clusters += 1
        rep_type = min(tc.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        by_type[rep_type].append(comp)
    splits = {"train": [], "dev": [], "sealed-test": []}
    for ty, c_list in by_type.items():
        rng.shuffle(c_list)
        n = sum(len(c) for c in c_list)
        n_sealed = int(n * 0.10)
        n_dev = int(n * 0.05)
        s_len, d_len = 0, 0
        for c in c_list:
            clen = len(c)
            if s_len < n_sealed and abs((s_len + clen) - n_sealed) < abs(s_len - n_sealed):
                splits["sealed-test"].extend(rows[i] for i in c)
                s_len += clen
            elif d_len < n_dev and abs((d_len + clen) - n_dev) < abs(d_len - n_dev):
                splits["dev"].extend(rows[i] for i in c)
                d_len += clen
            else:
                splits["train"].extend(rows[i] for i in c)
    leaks = {}
    for a, b in [("train", "dev"), ("train", "sealed-test"),
                 ("dev", "sealed-test")]:
        leaks[f"{a}__{b}"] = lsh_pairs(splits[a], splits[b])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cluster_stats = {
        "clusters_total": len(clusters),
        "multi_doc_clusters": multi_doc_clusters,
        "multi_doc_rows": multi_doc_rows,
        "max_cluster_size": max_cluster_size,
        "cross_type_clusters": cross_type_clusters,
    }
    (out / "cluster-stats.json").write_text(
        json.dumps(cluster_stats, ensure_ascii=False, indent=1), encoding="utf-8")
    stats = {"rows_after_dedup": len(rows), "leaks": leaks, "splits": {}}
    for name, part in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(
                    {"text": r["text"], "domain": r["domain"],
                     "document_type": r["document_type"],
                     "document_format": r["document_format"]},
                    ensure_ascii=False) + "\n")
        c = Counter(r["document_type"] for r in part)
        stats["splits"][name] = {"rows": len(part),
                                 "type_min": min(c.values()),
                                 "type_max": max(c.values())}
    # risk_tiers_v0.csv の domain 突き合わせ
    csv_path = Path(__file__).resolve().parent.parent / "risk_tiers_v0.csv"
    if csv_path.exists():
        actual = {}
        for r in rows:
            actual[r["document_type"]] = r["domain"]
        mismatch = []
        for line in csv_path.read_text(encoding="utf-8").splitlines()[1:]:
            if not line.strip():
                continue
            ty, dom, _tier = line.split(",")
            if actual.get(ty) != dom:
                mismatch.append({"type": ty, "csv": dom,
                                 "actual": actual.get(ty)})
        stats["risk_tier_domain_mismatch"] = mismatch
    (out / "build-stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
