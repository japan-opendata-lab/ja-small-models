#!/usr/bin/env python
"""第6号 ja-error-detect-v1 本番データ構築スクリプト。

HuggingFaceFW/fineweb-2 の日本語サブセット (jpn_Jpan, ODC-By 1.0) の
row group 8 以降（パイロット用 row group 0〜7 と完全非重複）のみを使用し、
文書・ヘッダ・URL・MinHash LSH (Jaccard >= 0.80) のグループ単位で
train (100,000件) / dev (5,000件) / sealed-test (5,000件) に分割し、
正常文 60% / 誤り文 40%（6種別均等・1誤り 85% / 2誤り 15%）のデータを構築する。

使い方:
  python code/build_prod_data.py --out data/prod --seed 1042
"""
import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.parse import urlsplit

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import error_injector as ei  # noqa: E402

RE_BAD_LINE = re.compile(
    r"https?://|www\.|<[^>]+>|[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]|^[・※■◆●▲▼▶★☆\-*#>]"
)
RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_REPEAT4 = re.compile(r"(.)\1{3,}")


def _is_jp_char(ch: str) -> bool:
    c = ord(ch)
    return (
        0x3040 <= c <= 0x309F
        or 0x30A0 <= c <= 0x30FF
        or 0x4E00 <= c <= 0x9FFF
        or ch in "、。？！「」『』（）・ー〜"
    )


def _split_sentences(line: str) -> List[str]:
    """括弧・引用符の深さを追跡して句末記号（。？！）で文分割する。"""
    sents: List[str] = []
    buf: List[str] = []
    d1 = d2 = d3 = 0
    for ch in line:
        buf.append(ch)
        if ch == "「":
            d1 += 1
        elif ch == "」" and d1 > 0:
            d1 -= 1
        elif ch == "『":
            d2 += 1
        elif ch == "』" and d2 > 0:
            d2 -= 1
        elif ch == "（":
            d3 += 1
        elif ch == "）" and d3 > 0:
            d3 -= 1
        elif ch in ("。", "？", "！") and d1 == 0 and d2 == 0 and d3 == 0:
            s = "".join(buf).strip()
            if s:
                sents.append(s)
            buf = []
    if buf:
        s = "".join(buf).strip()
        if s:
            sents.append(s)
    return sents


def _ok_sentence(s: str) -> bool:
    """自然日本語文の品質・長さ・文字種・括弧対応フィルタ。"""
    n = len(s)
    if not (20 <= n <= 160):
        return False
    if s[-1] not in ("。", "？", "！"):
        return False
    if RE_BAD_LINE.search(s) or RE_EMAIL.search(s):
        return False
    if RE_REPEAT4.search(s):
        return False
    if (
        s.count("「") != s.count("」")
        or s.count("『") != s.count("』")
        or s.count("（") != s.count("）")
    ):
        return False
    jp = sum(1 for c in s if _is_jp_char(c))
    hira = sum(1 for c in s if "\u3041" <= c <= "\u3096")
    kanji = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    dig = sum(1 for c in s if c.isdigit())
    return jp / n >= 0.75 and hira >= 2 and kanji >= 2 and dig / n <= 0.25


def _norm_url(url: str) -> str:
    if not url:
        return ""
    try:
        sp = urlsplit(url)
        return f"{sp.netloc.lower()}{sp.path}"
    except Exception:
        return url.split("?")[0].split("#")[0]


def _shingles(text: str, n: int = 5) -> Set[str]:
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}


def _minhash_coeffs(num_perm: int = 64, seed: int = 42):
    rng = random.Random(seed)
    m = (1 << 61) - 1
    return m, [(rng.randrange(1, m), rng.randrange(0, m)) for _ in range(num_perm)]


def _minhash(sh: Set[str], coeffs, m: int, sh_hash_cache: Dict[str, int]) -> List[int]:
    """5-gram ハッシュキャッシュを用いて高速に MinHash 署名を生成する。"""
    sig = [m] * len(coeffs)
    for g in sh:
        h = sh_hash_cache.get(g)
        if h is None:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:16], 16)
            sh_hash_cache[g] = h
        for i, (a, b) in enumerate(coeffs):
            v = (a * h + b) % m
            if v < sig[i]:
                sig[i] = v
    return sig


def _count_lsh_pairs(
    rows_a: List[dict],
    rows_b: List[dict],
    bands: int = 16,
    rows_per: int = 4,
    thr: float = 0.80,
) -> int:
    """2 分割間で Jaccard >= thr の近傍重複件数を厳格検査する。"""
    index: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
    for j, r in enumerate(rows_b):
        sig = r["_sig"]
        for b in range(bands):
            index[(b, tuple(sig[b * rows_per : (b + 1) * rows_per]))].append(j)
    hits = 0
    for r in rows_a:
        cand: Set[int] = set()
        sig = r["_sig"]
        for b in range(bands):
            cand.update(index.get((b, tuple(sig[b * rows_per : (b + 1) * rows_per])), ()))
        sh_a = r["_sh"]
        for j in cand:
            sh_b = rows_b[j]["_sh"]
            u = len(sh_a | sh_b)
            if u and len(sh_a & sh_b) / u >= thr:
                hits += 1
                break
    return hits


def _fetch_candidates(
    target_sents: int, seed: int, start_row_group: int = 8
) -> Tuple[List[dict], dict]:
    """FineWeb-2 jpn_Jpan Parquet の start_row_group 以降から候補文を収集する。"""
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    parquet_path = "datasets/HuggingFaceFW/fineweb-2/data/jpn_Jpan/train/000_00000.parquet"
    fs = HfFileSystem()

    candidates: List[dict] = []
    seen_exact: Set[str] = set()
    docs_scanned = 0
    docs_accepted = 0
    raw_sents = 0
    cur_rg = start_row_group
    rg_read = 0

    for attempt in range(5):
        try:
            with fs.open(parquet_path, "rb") as f:
                pf = pq.ParquetFile(f)
                num_rgs = pf.num_row_groups
                for rg_idx in range(cur_rg, num_rgs):
                    tbl = pf.read_row_group(
                        rg_idx, columns=["text", "id", "url", "language_score"]
                    )
                    cur_rg = rg_idx + 1
                    rg_read += 1
                    texts = tbl["text"].to_pylist()
                    ids = tbl["id"].to_pylist()
                    urls = tbl["url"].to_pylist()
                    scores = tbl["language_score"].to_pylist()

                    for txt, doc_id, url, sc in zip(texts, ids, urls, scores):
                        docs_scanned += 1
                        if sc is None or sc < 0.95 or not txt:
                            continue
                        norm_doc = re.sub(r"\s+", " ", txt).strip()
                        if len(norm_doc) < 40:
                            continue
                        header_key = hashlib.md5(norm_doc[:200].encode("utf-8")).hexdigest()[:16]
                        url_key = hashlib.md5(_norm_url(url or "").encode("utf-8")).hexdigest()[:16]
                        doc_sents: List[str] = []
                        for line in txt.splitlines():
                            line_clean = re.sub(r"\s+", " ", line).strip()
                            if (
                                len(line_clean) < 15
                                or RE_BAD_LINE.search(line_clean)
                                or RE_EMAIL.search(line_clean)
                            ):
                                continue
                            for s in _split_sentences(line_clean):
                                if _ok_sentence(s):
                                    doc_sents.append(s)
                        if not doc_sents:
                            continue
                        docs_accepted += 1
                        rng.shuffle(doc_sents)
                        for s in doc_sents[:5]:
                            raw_sents += 1
                            eh = hashlib.md5(s.encode("utf-8")).hexdigest()
                            if eh in seen_exact:
                                continue
                            seen_exact.add(eh)
                            candidates.append(
                                {
                                    "clean_text": s,
                                    "exact_hash": eh,
                                    "doc_id": str(doc_id),
                                    "header_key": header_key,
                                    "url_key": url_key,
                                }
                            )
                    if rg_read % 5 == 0 or len(candidates) >= target_sents:
                        print(
                            f"  rg={rg_idx} (read={rg_read}) | docs={docs_accepted}/{docs_scanned} | uniq_sents={len(candidates)}"
                        )
                    if len(candidates) >= target_sents:
                        break
            break
        except Exception as exc:
            if attempt == 4:
                raise
            print(f"[warn] row group {cur_rg} 読込リトライ ({attempt + 1}/5): {exc}")
            time.sleep(3)

    meta = {
        "parquet_path": parquet_path,
        "start_row_group": start_row_group,
        "end_row_group_exclusive": cur_rg,
        "row_groups_read": rg_read,
        "docs_scanned": docs_scanned,
        "docs_accepted": docs_accepted,
        "candidate_sentences_raw": raw_sents,
        "candidate_sentences_after_exact_dedup": len(candidates),
    }
    return candidates, meta


def _group_and_split(
    candidates: List[dict],
    seed: int,
    dev_pool_min: int,
    sealed_pool_min: int,
) -> Tuple[Dict[str, List[dict]], int]:
    """doc_id・header_key・url_key・MinHash LSH (Jaccard>=0.80) を Union-Find で連結し、
    グループ単位で train / dev / sealed-test に分割する。
    """
    n = len(candidates)
    parent = list(range(n))
    size = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]

    # 1. doc_id / header_key / url_key で連結
    for key_field in ("doc_id", "header_key", "url_key"):
        first_idx: Dict[str, int] = {}
        for i, r in enumerate(candidates):
            val = r[key_field]
            if not val:
                continue
            if val in first_idx:
                union(first_idx[val], i)
            else:
                first_idx[val] = i

    # 2. MinHash LSH (5-gram, 64 perm, 16x4, Jaccard>=0.80) で近傍重複も連結
    m_prime, coeffs = _minhash_coeffs(num_perm=64, seed=seed)
    sh_hash_cache: Dict[str, int] = {}
    for r in candidates:
        sh = _shingles(r["clean_text"], n=5)
        r["_sh"] = sh
        r["_sig"] = _minhash(sh, coeffs, m_prime, sh_hash_cache)

    bands, rows_per = 16, 4
    for b in range(bands):
        buckets: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        s_i, e_i = b * rows_per, (b + 1) * rows_per
        for i, r in enumerate(candidates):
            buckets[tuple(r["_sig"][s_i:e_i])].append(i)
        for idxs in buckets.values():
            if len(idxs) < 2:
                continue
            # 既に同一連結成分に属する要素を代表1件ずつに間引いてペア比較を高速化
            seen_roots: Set[int] = set()
            rep_idxs: List[int] = []
            for idx in idxs:
                rt = find(idx)
                if rt not in seen_roots:
                    seen_roots.add(rt)
                    rep_idxs.append(idx)
            if len(rep_idxs) < 2:
                continue
            limit_p = min(len(rep_idxs), 200)
            for p in range(limit_p):
                i = rep_idxs[p]
                sh_i = candidates[i]["_sh"]
                for q in range(p + 1, len(rep_idxs)):
                    j = rep_idxs[q]
                    if find(i) == find(j):
                        continue
                    sh_j = candidates[j]["_sh"]
                    u = len(sh_i | sh_j)
                    if u and len(sh_i & sh_j) / u >= 0.80:
                        union(i, j)

    comps: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)

    groups = list(comps.values())
    rng = random.Random(seed)
    rng.shuffle(groups)

    splits: Dict[str, List[dict]] = {"dev": [], "sealed-test": [], "train": []}
    for g_idx, idxs in enumerate(groups):
        g_hash = hashlib.md5(f"prod-grp-{g_idx}-{idxs[0]}".encode()).hexdigest()[:12]
        items = []
        for i in idxs:
            candidates[i]["doc_group"] = g_hash
            items.append(candidates[i])
        if len(splits["dev"]) < dev_pool_min and len(items) <= 250:
            splits["dev"].extend(items)
        elif len(splits["sealed-test"]) < sealed_pool_min and len(items) <= 250:
            splits["sealed-test"].extend(items)
        else:
            splits["train"].extend(items)

    return splits, len(groups)


def _build_split_rows(
    split_name: str,
    pool: List[dict],
    n_clean: int,
    n_err: int,
    injector: ei.ErrorInjector,
    seed: int,
) -> Tuple[List[dict], dict]:
    """分割プールから 正常文 n_clean 件・誤り文 n_err 件を均等 6 種別で生成する。"""
    rng = random.Random(seed)
    items = list(pool)
    rng.shuffle(items)

    n_two_target = round(n_err * 0.15)
    n_one_target = n_err - n_two_target

    type_counts: Counter = Counter({t: 0 for t in ei.ERROR_TYPES})
    err_rows: List[dict] = []
    unused_clean_pool: List[dict] = []

    one_done = 0
    two_done = 0

    for idx_item, item in enumerate(items):
        if len(err_rows) >= n_err:
            unused_clean_pool.append(item)
            continue

        clean_txt = item["clean_text"]
        cands_by_type = injector.collect_all_candidates(clean_txt)
        avail_types = [t for t in ei.ERROR_TYPES if cands_by_type[t]]
        if not avail_types:
            unused_clean_pool.append(item)
            continue

        want_two = two_done < n_two_target and (
            one_done >= n_one_target
            or rng.random() < (n_two_target - two_done) / max(1, n_err - len(err_rows))
        )

        ordered_types = sorted(avail_types, key=lambda t: (type_counts[t], rng.random()))

        if want_two:
            primary = ordered_types[0]
            secondary_candidates = [t for t in ordered_types if t != primary] + [primary]
            chosen_pair = None
            for sec in secondary_candidates:
                cor_txt, spans = injector.inject_errors(clean_txt, [primary, sec], rng, min_gap=4)
                if len(spans) == 2:
                    chosen_pair = (cor_txt, spans)
                    break
            if chosen_pair is not None:
                cor_txt, spans = chosen_pair
                two_done += 1
            else:
                cor_txt, spans = injector.inject_errors(clean_txt, [primary], rng)
                if not spans:
                    unused_clean_pool.append(item)
                    continue
                one_done += 1
        else:
            primary = ordered_types[0]
            cor_txt, spans = injector.inject_errors(clean_txt, [primary], rng)
            if not spans:
                unused_clean_pool.append(item)
                continue
            one_done += 1

        char_labels = [0] * len(cor_txt)
        for sp in spans:
            type_counts[sp["type"]] += 1
            char_labels[sp["start"]] = 1
            for k in range(sp["start"] + 1, sp["end"]):
                char_labels[k] = 2

        err_rows.append(
            {
                "doc_id": item["doc_id"],
                "doc_group": item["doc_group"],
                "text": cor_txt,
                "clean_text": clean_txt,
                "has_error": True,
                "spans": spans,
                "char_labels": char_labels,
                "_sh": item["_sh"],
                "_sig": item["_sig"],
            }
        )

        if len(err_rows) % 10000 == 0:
            print(f"    [{split_name}] 誤り注入 {len(err_rows)}/{n_err} 完了")

    assert len(err_rows) == n_err, f"{split_name} 誤り文不足: {len(err_rows)} < {n_err}"
    assert len(unused_clean_pool) >= n_clean, (
        f"{split_name} 正常文不足: {len(unused_clean_pool)} < {n_clean}"
    )

    clean_rows: List[dict] = []
    for item in unused_clean_pool[:n_clean]:
        txt = item["clean_text"]
        clean_rows.append(
            {
                "doc_id": item["doc_id"],
                "doc_group": item["doc_group"],
                "text": txt,
                "clean_text": txt,
                "has_error": False,
                "spans": [],
                "char_labels": [0] * len(txt),
                "_sh": item["_sh"],
                "_sig": item["_sig"],
            }
        )

    all_rows = err_rows + clean_rows
    rng.shuffle(all_rows)
    for idx, r in enumerate(all_rows, 1):
        r["id"] = f"{split_name}-{idx:06d}"

    total_spans = sum(type_counts.values())
    type_ratios = {t: round(type_counts[t] / total_spans, 4) for t in ei.ERROR_TYPES}
    for t, ratio in type_ratios.items():
        assert 0.14 <= ratio <= 0.19, f"{split_name} 種別比率逸脱: {t}={ratio} ({type_counts})"

    lens = [len(r["text"]) for r in all_rows]
    stats = {
        "total_rows": len(all_rows),
        "clean_rows": len(clean_rows),
        "error_rows": len(err_rows),
        "clean_ratio": round(len(clean_rows) / len(all_rows), 4),
        "unique_docs": len({r["doc_id"] for r in all_rows}),
        "unique_groups": len({r["doc_group"] for r in all_rows}),
        "error_1_span_rows": one_done,
        "error_2_span_rows": two_done,
        "total_spans": total_spans,
        "span_type_counts": {t: int(type_counts[t]) for t in ei.ERROR_TYPES},
        "span_type_ratios": type_ratios,
        "char_len_min": min(lens),
        "char_len_median": float(statistics.median(lens)),
        "char_len_mean": round(statistics.mean(lens), 2),
        "char_len_max": max(lens),
        "total_chars": sum(lens),
        "clean_total_chars": sum(len(r["text"]) for r in clean_rows),
    }
    return all_rows, stats


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            assert len(r["char_labels"]) == len(r["text"]), f"char_labels長不整合: {r['id']}"
            for sp in r["spans"]:
                assert r["text"][sp["start"] : sp["end"]] == sp["corrupted"], (
                    f"スパン文字列不整合: {r['id']} {sp}"
                )
            out = {
                "id": r["id"],
                "doc_id": r["doc_id"],
                "doc_group": r["doc_group"],
                "text": r["text"],
                "clean_text": r["clean_text"],
                "has_error": r["has_error"],
                "spans": r["spans"],
                "char_labels": r["char_labels"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/prod")
    ap.add_argument("--seed", type=int, default=1042)
    ap.add_argument("--start-row-group", type=int, default=8)
    ap.add_argument("--train-n", type=int, default=100000)
    ap.add_argument("--dev-n", type=int, default=5000)
    ap.add_argument("--sealed-n", type=int, default=5000)
    ap.add_argument("--clean-ratio", type=float, default=0.60)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_total = int((args.train_n + args.dev_n + args.sealed_n) * 1.25)
    print(
        f"[1/4] FineWeb-2 (jpn_Jpan, rg>={args.start_row_group}) から候補文を収集中 "
        f"(目標 {target_total} 文)..."
    )
    candidates, fetch_meta = _fetch_candidates(
        target_sents=target_total,
        seed=args.seed,
        start_row_group=args.start_row_group,
    )
    print(
        f"  取得完了: row_groups={fetch_meta['row_groups_read']} "
        f"(rg {fetch_meta['start_row_group']}..{fetch_meta['end_row_group_exclusive'] - 1}), "
        f"docs={fetch_meta['docs_accepted']}/{fetch_meta['docs_scanned']}, "
        f"uniq_sents={len(candidates)} ({time.time() - t0:.1f}s)"
    )

    print("[2/4] 文書・ヘッダ・URL・MinHash LSH (0.80) でグループ分割中...")
    splits_pool, n_groups = _group_and_split(
        candidates,
        seed=args.seed,
        dev_pool_min=int(args.dev_n * 1.25),
        sealed_pool_min=int(args.sealed_n * 1.25),
    )
    print(
        f"  グループ数={n_groups} | 候補配分 train={len(splits_pool['train'])}, "
        f"dev={len(splits_pool['dev'])}, sealed-test={len(splits_pool['sealed-test'])} ({time.time() - t0:.1f}s)"
    )

    print("[3/4] 6種別のプログラム的誤り注入と分割生成中...")
    injector = ei.ErrorInjector()
    built_rows: Dict[str, List[dict]] = {}
    split_stats: Dict[str, dict] = {}
    targets = {
        "train": args.train_n,
        "dev": args.dev_n,
        "sealed-test": args.sealed_n,
    }
    for offset_seed, sname in enumerate(("train", "dev", "sealed-test")):
        n_total = targets[sname]
        n_clean = round(n_total * args.clean_ratio)
        n_err = n_total - n_clean
        rows, st = _build_split_rows(
            sname,
            splits_pool[sname],
            n_clean=n_clean,
            n_err=n_err,
            injector=injector,
            seed=args.seed + offset_seed + 100,
        )
        built_rows[sname] = rows
        split_stats[sname] = st
        print(
            f"  {sname}: total={st['total_rows']} (clean={st['clean_rows']}, err={st['error_rows']}, "
            f"spans={st['total_spans']}) ratios={st['span_type_ratios']}"
        )

    print("[4/4] 分割間リーク検査と JSONL 書き出し中...")
    pairs = [("train", "dev"), ("train", "sealed-test"), ("dev", "sealed-test")]
    leak_checks = {
        "doc_overlap": {},
        "group_overlap": {},
        "exact_clean_text_overlap": {},
        "near_dup_lsh_080": {},
    }
    for a, b in pairs:
        pkey = f"{a}__{b}"
        ra, rb = built_rows[a], built_rows[b]
        doc_ov = len({r["doc_id"] for r in ra} & {r["doc_id"] for r in rb})
        grp_ov = len({r["doc_group"] for r in ra} & {r["doc_group"] for r in rb})
        txt_ov = len({r["clean_text"] for r in ra} & {r["clean_text"] for r in rb})
        lsh_ov = _count_lsh_pairs(ra, rb, thr=0.80)
        assert doc_ov == 0, f"doc_overlap != 0 on {pkey}: {doc_ov}"
        assert grp_ov == 0, f"group_overlap != 0 on {pkey}: {grp_ov}"
        assert txt_ov == 0, f"exact_overlap != 0 on {pkey}: {txt_ov}"
        assert lsh_ov == 0, f"near_dup_overlap != 0 on {pkey}: {lsh_ov}"
        leak_checks["doc_overlap"][pkey] = doc_ov
        leak_checks["group_overlap"][pkey] = grp_ov
        leak_checks["exact_clean_text_overlap"][pkey] = txt_ov
        leak_checks["near_dup_lsh_080"][pkey] = lsh_ov

    for sname in ("train", "dev", "sealed-test"):
        _write_jsonl(out_dir / f"{sname}.jsonl", built_rows[sname])

    build_stats = {
        "source": "HuggingFaceFW/fineweb-2 (jpn_Jpan, train)",
        "license": "ODC-By 1.0",
        "seed": args.seed,
        "clean_ratio_target": args.clean_ratio,
        "elapsed_sec": round(time.time() - t0, 2),
        "fetch": fetch_meta,
        "groups_total": n_groups,
        "splits": split_stats,
        "leak_checks": leak_checks,
    }
    (out_dir / "build-stats.json").write_text(
        json.dumps(build_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完了 ({time.time() - t0:.1f}s) -> {out_dir / 'build-stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
