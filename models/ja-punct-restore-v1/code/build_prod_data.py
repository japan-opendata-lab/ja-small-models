#!/usr/bin/env python3
"""第3号 ja-punct-restore-v1 本番データセット構築・リーク検査スクリプト。

DATA-BUILD-v1.md および SPEC-v1.md §4・§5・§7.5 に準拠し、以下を実行する:
1. FineWeb-2 日本語 (jpn_Jpan) から有効出典文書 60,000 件をストリーミング取得し、
   1文書あたり上限 5 ウィンドウ（偏り抑制）で 300〜600 文字のウィンドウを構築する。
2. 口語合成データ (data/spoken-train.jsonl) を読み込み、学習側 (train) にのみ混ぜる。
3. グローバル完全一致重複除去、doc_id 単位の 3 分割 (sealed-test / dev / train)、
   文字 5-gram 64-hash MinHash LSH (類似 81% 以上) による分割間クロスリーク除去を行う。
4. 出力ファイルを再読込して独立検証 (doc_overlap=0, exact=0, MinHash>=0.81=0) を通過させ、
   封印テストの本文は一切出力せず統計のみ build-stats.json に記録する。
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import build_punct_data as bpd  # noqa: E402
import generate_spoken as gsp  # noqa: E402

PARQUET_SHARDS = [
    "datasets/HuggingFaceFW/fineweb-2/data/jpn_Jpan/train/000_00000.parquet",
    "datasets/HuggingFaceFW/fineweb-2/data/jpn_Jpan/train/000_00001.parquet",
]
MAX_WINS_PER_DOC = 5
NUM_PERM = 64
LSH_BANDS = 16
LSH_ROWS = 4
SIM_THRESHOLD = 0.81

# 64個の決定論的シードと奇数乗数（64bit Universal Hashing 用）
_SEEDS64 = np.array(
    [((i + 1) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF for i in range(NUM_PERM)],
    dtype=np.uint64,
)
_HASH_MULT = np.uint64(0xBF58476D1CE4E5B9)


def norm_for_lsh(text: str) -> str:
    """空白除去および NFKC 正規化（小文字化）を行う。"""
    s = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", "", s)


def char_5grams(text: str) -> set[int]:
    """正規化済み文字列から CRC32 ベースの文字 5-gram 整数集合を返す。"""
    nc = norm_for_lsh(text)
    if len(nc) < 5:
        return {zlib.crc32(nc.encode("utf-8"))}
    return {zlib.crc32(nc[i : i + 5].encode("utf-8")) for i in range(len(nc) - 4)}


def compute_minhash_batch(texts: list[str]) -> np.ndarray:
    """複数テキストの 64-hash MinHash シグネチャ行列 (N, 64) uint32 を返す。"""
    n = len(texts)
    sigs = np.empty((n, NUM_PERM), dtype=np.uint32)
    for idx, text in enumerate(texts):
        nc = norm_for_lsh(text)
        if len(nc) < 5:
            grams = np.array([zlib.crc32(nc.encode("utf-8"))], dtype=np.uint64)
        else:
            arr = [zlib.crc32(nc[i : i + 5].encode("utf-8")) for i in range(len(nc) - 4)]
            grams = np.array(arr, dtype=np.uint64)
        h = ((grams[:, None] ^ _SEEDS64[None, :]) * _HASH_MULT) >> np.uint64(32)
        sigs[idx] = h.min(axis=0).astype(np.uint32)
    return sigs


def compute_minhash_parallel(texts: list[str], workers: int = 8) -> np.ndarray:
    """マルチプロセスで全テキストの MinHash シグネチャを計算する。"""
    if len(texts) < 2000 or workers <= 1:
        return compute_minhash_batch(texts)
    chunk_size = max(1000, (len(texts) + workers - 1) // workers)
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(compute_minhash_batch, chunks))
    return np.vstack(parts)


def build_lsh_index(sigs: np.ndarray) -> list[dict[bytes, list[int]]]:
    """16 バンド × 4 行の LSH バケット辞書リストを構築する。"""
    buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(LSH_BANDS)]
    for idx in range(sigs.shape[0]):
        row = sigs[idx]
        for b in range(LSH_BANDS):
            key = row[b * LSH_ROWS : (b + 1) * LSH_ROWS].tobytes()
            buckets[b][key].append(idx)
    return buckets


def find_cross_leaks(
    query_texts: list[str],
    query_sigs: np.ndarray,
    ref_texts: list[str],
    ref_sigs: np.ndarray,
    ref_buckets: list[dict[bytes, list[int]]],
    threshold: float = SIM_THRESHOLD,
) -> dict[int, tuple[int, float, float]]:
    """参照側 LSH インデックスに対し、類似度 threshold 以上のクエリ行を返す。

    戻り値: {query_idx: (matched_ref_idx, est_sim, exact_jaccard)}
    """
    leaks: dict[int, tuple[int, float, float]] = {}
    ref_gram_cache: dict[int, set[int]] = {}
    for qi in range(query_sigs.shape[0]):
        q_row = query_sigs[qi]
        cands = set()
        for b in range(LSH_BANDS):
            key = q_row[b * LSH_ROWS : (b + 1) * LSH_ROWS].tobytes()
            hit = ref_buckets[b].get(key)
            if hit:
                cands.update(hit)
        if not cands:
            continue
        cand_arr = np.fromiter(cands, dtype=np.int64)
        ests = np.mean(ref_sigs[cand_arr] == q_row, axis=1)
        high_pos = np.flatnonzero(ests >= 0.65)
        if high_pos.size == 0:
            continue
        q_grams = None
        for idx_pos in high_pos:
            ri = int(cand_arr[idx_pos])
            est = float(ests[idx_pos])
            if q_grams is None:
                q_grams = char_5grams(query_texts[qi])
            r_grams = ref_gram_cache.get(ri)
            if r_grams is None:
                r_grams = char_5grams(ref_texts[ri])
                ref_gram_cache[ri] = r_grams
            inter = len(q_grams & r_grams)
            union = len(q_grams) + len(r_grams) - inter
            exact = inter / union if union else 1.0
            if est >= threshold or exact >= threshold:
                leaks[qi] = (ri, est, exact)
                break
    return leaks


def sample_windows_evenly(wins: list, cap: int) -> list:
    """文書内ウィンドウ数が cap を超える場合、等間隔インデックスで cap 件抽出する。"""
    if len(wins) <= cap:
        return wins
    if cap == 1:
        return [wins[0]]
    idxs = [(i * (len(wins) - 1)) // (cap - 1) for i in range(cap)]
    return [wins[i] for i in idxs]


def stream_fineweb2(target_docs: int, max_wins_per_doc: int) -> tuple[list[dict], dict]:
    """FineWeb-2 jpn_Jpan Parquet シャードから必要 row group のみを取得しウィンドウ化する。"""
    fs = HfFileSystem()
    rows: list[dict] = []
    doc_count = 0
    raw_wins_total = 0
    capped_docs = 0
    row_groups_read = 0
    raw_rows_scanned = 0

    for shard in PARQUET_SHARDS:
        if doc_count >= target_docs:
            break
        with fs.open(shard, "rb") as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                if doc_count >= target_docs:
                    break
                # 通信エラー等に備え row group 単位でリトライ
                table = None
                for attempt in range(4):
                    try:
                        table = pf.read_row_group(rg, columns=["text"])
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(2 ** attempt)
                row_groups_read += 1
                col_texts = table["text"].to_pylist()
                raw_rows_scanned += len(col_texts)
                for raw_text in col_texts:
                    if doc_count >= target_docs:
                        break
                    text = bpd.normalize(raw_text or "")
                    if len(text) < 350:
                        continue
                    doc_id = hashlib.md5(text[:200].encode()).hexdigest()[:12]
                    wins = bpd.windows_from_doc(text, lo=300, hi=600)
                    if not wins:
                        continue
                    valid = []
                    for w in wins:
                        inp, labels = bpd.make_labels(w)
                        if len(inp) >= 100:
                            valid.append((inp, labels))
                    if not valid:
                        continue
                    doc_count += 1
                    raw_wins_total += len(valid)
                    if len(valid) > max_wins_per_doc:
                        capped_docs += 1
                    chosen = sample_windows_evenly(valid, max_wins_per_doc)
                    for inp, labels in chosen:
                        rows.append(
                            {
                                "text": inp,
                                "labels": labels,
                                "doc": doc_id,
                                "source": "fineweb2",
                            }
                        )
                if row_groups_read % 10 == 0 or doc_count >= target_docs:
                    print(
                        f"[FineWeb-2] rg={row_groups_read} scanned={raw_rows_scanned} "
                        f"docs={doc_count}/{target_docs} wins={len(rows)}",
                        flush=True,
                    )

    meta = {
        "target_docs": target_docs,
        "docs_collected": doc_count,
        "raw_rows_scanned": raw_rows_scanned,
        "row_groups_read": row_groups_read,
        "raw_windows_before_cap": raw_wins_total,
        "docs_capped": capped_docs,
        "max_wins_per_doc": max_wins_per_doc,
        "windows_after_cap": len(rows),
    }
    return rows, meta


def compute_split_stats(rows: list[dict]) -> dict:
    """1 つの分割（または材料スライス）の件数・記号分布・文字長分布を算出する。"""
    n = len(rows)
    if n == 0:
        return {"windows": 0}
    lens = sorted(len(r["text"]) for r in rows)
    c = Counter()
    for r in rows:
        c.update(r["labels"])
    total_chars = sum(lens)
    total_punct = sum(c.get(k, 0) for k in (1, 2, 3, 4))
    label_names = {"0": "O", "1": "COMMA", "2": "PERIOD", "3": "QUESTION", "4": "EXCLAIM"}
    counts = {str(k): int(c.get(k, 0)) for k in range(5)}
    all_pcts = {str(k): round(100.0 * c.get(k, 0) / total_chars, 4) for k in range(5)}
    punct_pcts = {
        str(k): round(100.0 * c.get(k, 0) / total_punct, 4) if total_punct else 0.0
        for k in (1, 2, 3, 4)
    }
    punct_pcts_named = {label_names[str(k)]: punct_pcts[str(k)] for k in (1, 2, 3, 4)}
    gt510 = sum(1 for x in lens if x > 510)
    gt600 = sum(1 for x in lens if x > 600)
    uniq_texts = len({hashlib.md5(r["text"].encode()).hexdigest() for r in rows})
    return {
        "windows": n,
        "unique_windows": uniq_texts,
        "unique_rate": round(uniq_texts / n, 6),
        "unique_docs": len({r["doc"] for r in rows}),
        "total_chars": total_chars,
        "total_punct": total_punct,
        "label_counts": counts,
        "all_char_pcts": all_pcts,
        "punct_pcts": punct_pcts,
        "punct_pcts_named": punct_pcts_named,
        "char_len_dist": {
            "min": lens[0],
            "median": float(np.median(lens)),
            "mean": round(float(np.mean(lens)), 2),
            "p90": int(lens[int(n * 0.90)]),
            "p95": int(lens[int(n * 0.95)]),
            "max": lens[-1],
            "gt_510_count": gt510,
            "gt_510_pct": round(100.0 * gt510 / n, 2),
            "gt_600_count": gt600,
            "gt_600_pct": round(100.0 * gt600 / n, 2),
        },
    }


def analyze_spoken_diversity(spoken_rows: list[dict], sigs: np.ndarray) -> dict:
    """口語合成の文単位ユニーク数（フィラー込み/抜き）と内部類似度を診断する。"""
    id_to_ch = {1: "、", 2: "。", 3: "？", 4: "！"}
    fillers = sorted([f for f in gsp.FILLERS if f], key=len, reverse=True)

    all_sents_with_filler: list[tuple[str, str]] = []
    all_sents_no_filler: list[tuple[str, str]] = []
    by_end = Counter()
    by_end_uniq_wf: dict[str, set[str]] = defaultdict(set)
    by_end_uniq_nf: dict[str, set[str]] = defaultdict(set)

    for r in spoken_rows:
        buf = []
        for ch, lb in zip(r["text"], r["labels"]):
            buf.append(ch)
            if lb:
                buf.append(id_to_ch[lb])
            if lb in (2, 3, 4):
                sent = "".join(buf)
                buf = []
                end_sym = id_to_ch[lb]
                by_end[end_sym] += 1
                all_sents_with_filler.append((end_sym, sent))
                by_end_uniq_wf[end_sym].add(sent)
                stripped = sent
                for fl in fillers:
                    if stripped.startswith(fl):
                        stripped = stripped[len(fl) :]
                        break
                all_sents_no_filler.append((end_sym, stripped))
                by_end_uniq_nf[end_sym].add(stripped)

    # 口語内部での MinHash LSH (sim >= 0.81) 自己一致を除く近傍類似ウィンドウ数を調査
    buckets = build_lsh_index(sigs)
    sim_windows = 0
    for i in range(sigs.shape[0]):
        row = sigs[i]
        cands = set()
        for b in range(LSH_BANDS):
            key = row[b * LSH_ROWS : (b + 1) * LSH_ROWS].tobytes()
            cands.update(buckets[b][key])
        cands.discard(i)
        if not cands:
            continue
        cand_arr = np.fromiter(cands, dtype=np.int64)
        if np.any(np.mean(sigs[cand_arr] == row, axis=1) >= SIM_THRESHOLD):
            sim_windows += 1

    return {
        "template_counts": {
            "S_DECL": len(gsp.S_DECL),
            "S_QUEST": len(gsp.S_QUEST),
            "S_EXCL": len(gsp.S_EXCL),
            "S_TALK": len(gsp.S_TALK),
            "FILLERS_non_empty": len(fillers),
        },
        "total_sentences_generated": len(all_sents_with_filler),
        "unique_sentences_with_filler": len({s for _, s in all_sents_with_filler}),
        "unique_sentences_without_filler": len({s for _, s in all_sents_no_filler}),
        "by_sentence_end": {
            sym: {
                "total": int(by_end[sym]),
                "unique_with_filler": len(by_end_uniq_wf[sym]),
                "unique_without_filler": len(by_end_uniq_nf[sym]),
            }
            for sym in ("。", "？", "！")
        },
        "internal_minhash_ge_081_windows": sim_windows,
        "internal_minhash_ge_081_pct": round(100.0 * sim_windows / len(spoken_rows), 2)
        if spoken_rows
        else 0.0,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def verify_written_outputs(out_dir: Path, workers: int) -> dict:
    """書き出した JSONL を再読込してスキーマ・文書重複・完全重複・LSH クロスリークを独立検証する。"""
    loaded: dict[str, list[dict]] = {}
    for sp in ("train", "dev", "sealed-test"):
        fp = out_dir / f"{sp}.jsonl"
        part = []
        with open(fp, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                assert isinstance(r["text"], str) and len(r["text"]) > 0
                assert isinstance(r["labels"], list) and len(r["labels"]) == len(r["text"])
                assert all(lb in (0, 1, 2, 3, 4) for lb in r["labels"])
                if sp in ("dev", "sealed-test"):
                    assert r.get("source") != "spoken", f"spoken leaked into {sp}!"
                part.append(r)
        loaded[sp] = part

    docs = {sp: {r["doc"] for r in loaded[sp]} for sp in loaded}
    doc_overlaps = {
        "train_dev": len(docs["train"] & docs["dev"]),
        "train_sealed": len(docs["train"] & docs["sealed-test"]),
        "dev_sealed": len(docs["dev"] & docs["sealed-test"]),
    }
    assert all(v == 0 for v in doc_overlaps.values()), f"doc_overlap found: {doc_overlaps}"

    hashes = {
        sp: [hashlib.md5(r["text"].encode()).hexdigest() for r in loaded[sp]]
        for sp in loaded
    }
    for sp, hs in hashes.items():
        assert len(hs) == len(set(hs)), f"Duplicate text inside {sp}!"
    hsets = {sp: set(hs) for sp, hs in hashes.items()}
    exact_overlaps = {
        "train_dev": len(hsets["train"] & hsets["dev"]),
        "train_sealed": len(hsets["train"] & hsets["sealed-test"]),
        "dev_sealed": len(hsets["dev"] & hsets["sealed-test"]),
    }
    assert all(v == 0 for v in exact_overlaps.values()), f"exact_overlap found: {exact_overlaps}"

    texts = {sp: [r["text"] for r in loaded[sp]] for sp in loaded}
    sigs = {sp: compute_minhash_parallel(texts[sp], workers=workers) for sp in loaded}
    idx_dev = build_lsh_index(sigs["dev"])
    idx_sealed = build_lsh_index(sigs["sealed-test"])

    leak_td = find_cross_leaks(texts["train"], sigs["train"], texts["dev"], sigs["dev"], idx_dev)
    leak_ts = find_cross_leaks(
        texts["train"], sigs["train"], texts["sealed-test"], sigs["sealed-test"], idx_sealed
    )
    leak_ds = find_cross_leaks(
        texts["dev"], sigs["dev"], texts["sealed-test"], sigs["sealed-test"], idx_sealed
    )
    minhash_cross_leaks = {
        "train_dev": len(leak_td),
        "train_sealed": len(leak_ts),
        "dev_sealed": len(leak_ds),
    }
    assert all(v == 0 for v in minhash_cross_leaks.values()), (
        f"MinHash cross leak found: {minhash_cross_leaks}"
    )

    return {
        "doc_overlaps": doc_overlaps,
        "exact_overlaps": exact_overlaps,
        "minhash_cross_leaks": minhash_cross_leaks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=60000)
    ap.add_argument("--max-wins-per-doc", type=int, default=MAX_WINS_PER_DOC)
    ap.add_argument("--spoken", default="data/spoken-train.jsonl")
    ap.add_argument("--out", default="data/prod")
    ap.add_argument("--sealed-target", type=int, default=5000)
    ap.add_argument("--dev-target", type=int, default=8050)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t_start = time.time()
    print("[1/5] Streaming FineWeb-2 jpn_Jpan...", flush=True)
    fw_rows, fw_meta = stream_fineweb2(args.docs, args.max_wins_per_doc)

    print("[2/5] Loading spoken synthetic rows & global exact dedup...", flush=True)
    spoken_raw = []
    with open(args.spoken, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["source"] = "spoken"
            spoken_raw.append(r)

    seen_hashes: set[str] = set()
    fw_uniq: list[dict] = []
    fw_exact_dropped = 0
    for r in fw_rows:
        h = hashlib.md5(r["text"].encode()).hexdigest()
        if h in seen_hashes:
            fw_exact_dropped += 1
            continue
        seen_hashes.add(h)
        fw_uniq.append(r)

    spoken_uniq: list[dict] = []
    spoken_exact_dropped = 0
    for r in spoken_raw:
        h = hashlib.md5(r["text"].encode()).hexdigest()
        if h in seen_hashes:
            spoken_exact_dropped += 1
            continue
        seen_hashes.add(h)
        spoken_uniq.append(r)

    print(
        f"Exact dedup: fw {len(fw_rows)}->{len(fw_uniq)} (drop={fw_exact_dropped}), "
        f"spoken {len(spoken_raw)}->{len(spoken_uniq)} (drop={spoken_exact_dropped})",
        flush=True,
    )

    print("[3/5] Grouping by doc_id and splitting into sealed-test / dev / train...", flush=True)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in fw_uniq:
        by_doc[r["doc"]].append(r)
    doc_keys = sorted(by_doc.keys())
    rng = random.Random(args.seed)
    rng.shuffle(doc_keys)

    sealed_rows: list[dict] = []
    dev_rows: list[dict] = []
    train_fw_rows: list[dict] = []

    for dk in doc_keys:
        grp = by_doc[dk]
        if len(sealed_rows) < args.sealed_target:
            sealed_rows.extend(grp)
        elif len(dev_rows) < args.dev_target:
            dev_rows.extend(grp)
        else:
            train_fw_rows.extend(grp)

    print(
        f"Initial doc-split: sealed={len(sealed_rows)}, dev={len(dev_rows)}, "
        f"train_fw={len(train_fw_rows)}, spoken={len(spoken_uniq)}",
        flush=True,
    )

    print("[4/5] Computing 64-hash MinHash & filtering cross-split leaks (>=0.81)...", flush=True)
    sealed_texts = [r["text"] for r in sealed_rows]
    dev_texts = [r["text"] for r in dev_rows]
    train_all_pre = train_fw_rows + spoken_uniq
    train_texts = [r["text"] for r in train_all_pre]

    sig_sealed = compute_minhash_parallel(sealed_texts, workers=args.workers)
    sig_dev = compute_minhash_parallel(dev_texts, workers=args.workers)
    sig_train = compute_minhash_parallel(train_texts, workers=args.workers)

    # 4-1: dev vs sealed-test
    idx_sealed = build_lsh_index(sig_sealed)
    dev_leaks = find_cross_leaks(dev_texts, sig_dev, sealed_texts, sig_sealed, idx_sealed)
    dev_clean = [r for i, r in enumerate(dev_rows) if i not in dev_leaks]
    dev_clean_texts = [r["text"] for r in dev_clean]
    sig_dev_clean = sig_dev[[i for i in range(len(dev_rows)) if i not in dev_leaks]]

    # 4-2: train vs (dev_clean + sealed)
    eval_ref_texts = dev_clean_texts + sealed_texts
    eval_ref_sigs = np.vstack([sig_dev_clean, sig_sealed])
    idx_eval = build_lsh_index(eval_ref_sigs)
    train_leaks = find_cross_leaks(
        train_texts, sig_train, eval_ref_texts, eval_ref_sigs, idx_eval
    )

    train_clean: list[dict] = []
    train_fw_leaked = 0
    train_spoken_leaked = 0
    for i, r in enumerate(train_all_pre):
        if i in train_leaks:
            if r["source"] == "spoken":
                train_spoken_leaked += 1
            else:
                train_fw_leaked += 1
            continue
        train_clean.append(r)

    rng.shuffle(train_clean)
    rng.shuffle(dev_clean)
    rng.shuffle(sealed_rows)

    out_dir = Path(args.out)
    write_jsonl(out_dir / "train.jsonl", train_clean)
    write_jsonl(out_dir / "dev.jsonl", dev_clean)
    write_jsonl(out_dir / "sealed-test.jsonl", sealed_rows)

    print("[5/5] Independently verifying written files & computing stats...", flush=True)
    verify_res = verify_written_outputs(out_dir, workers=args.workers)

    train_fw_clean = [r for r in train_clean if r["source"] == "fineweb2"]
    train_sp_clean = [r for r in train_clean if r["source"] == "spoken"]
    sp_keep_indices = [
        i
        for i, r in enumerate(train_all_pre)
        if i >= len(train_fw_rows) and i not in train_leaks
    ]
    sig_spoken_clean = (
        sig_train[sp_keep_indices] if sp_keep_indices else np.empty((0, NUM_PERM), dtype=np.uint32)
    )

    train_stats = compute_split_stats(train_clean)
    q_pct = train_stats["punct_pcts"]["3"]
    e_pct = train_stats["punct_pcts"]["4"]

    report_stats = {
        "build_params": {
            "docs_target": args.docs,
            "max_wins_per_doc": args.max_wins_per_doc,
            "seed": args.seed,
            "lsh": {"num_perm": NUM_PERM, "bands": LSH_BANDS, "rows": LSH_ROWS, "threshold": SIM_THRESHOLD},
            "elapsed_sec": round(time.time() - t_start, 2),
        },
        "fineweb2_collection": {
            **fw_meta,
            "exact_dedup_dropped": fw_exact_dropped,
            "unique_windows": len(fw_uniq),
            "unique_doc_ids": len(doc_keys),
        },
        "spoken_collection": {
            "raw_windows": len(spoken_raw),
            "exact_dedup_dropped": spoken_exact_dropped,
            "minhash_cross_leaks_dropped": train_spoken_leaked,
            "final_train_windows": len(train_sp_clean),
            "diversity_diagnosis": analyze_spoken_diversity(train_sp_clean, sig_spoken_clean),
        },
        "dedup_and_leak_checks": {
            "pre_lsh_counts": {
                "train_fineweb2": len(train_fw_rows),
                "train_spoken": len(spoken_uniq),
                "dev": len(dev_rows),
                "sealed_test": len(sealed_rows),
            },
            "lsh_dropped": {
                "dev_vs_sealed": len(dev_leaks),
                "train_fineweb2_vs_eval": train_fw_leaked,
                "train_spoken_vs_eval": train_spoken_leaked,
            },
            "post_write_verification": verify_res,
        },
        "splits": {
            "train": train_stats,
            "dev": compute_split_stats(dev_clean),
            "sealed-test": compute_split_stats(sealed_rows),
        },
        "train_by_source": {
            "fineweb2": compute_split_stats(train_fw_clean),
            "spoken": compute_split_stats(train_sp_clean),
        },
        "train_target_check": {
            "question_pct_in_punct": q_pct,
            "question_ge_5pct": bool(q_pct >= 5.0),
            "exclaim_pct_in_punct": e_pct,
            "exclaim_ge_5pct": bool(e_pct >= 5.0),
        },
        "output_files_bytes": {
            fn: (out_dir / fn).stat().st_size
            for fn in ("train.jsonl", "dev.jsonl", "sealed-test.jsonl")
        },
    }

    (out_dir / "build-stats.json").write_text(
        json.dumps(report_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DONE! Elapsed:", report_stats["build_params"]["elapsed_sec"], "s")
    print(
        json.dumps(
            {
                "train_windows": report_stats["splits"]["train"]["windows"],
                "dev_windows": report_stats["splits"]["dev"]["windows"],
                "sealed_windows": report_stats["splits"]["sealed-test"]["windows"],
                "verification": verify_res,
                "train_target_check": report_stats["train_target_check"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
