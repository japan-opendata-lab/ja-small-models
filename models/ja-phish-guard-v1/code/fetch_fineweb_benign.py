#!/usr/bin/env python3
"""FineWeb-2 日本語 (jpn_Jpan, ODC-By 1.0) から通常文サンプル benign_web.jsonl を構築する。"""

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

PARQUET_SHARDS = [
    "datasets/HuggingFaceFW/fineweb-2/data/jpn_Jpan/train/000_00000.parquet",
    "datasets/HuggingFaceFW/fineweb-2/data/jpn_Jpan/train/000_00001.parquet",
]

JP_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
BAD_RE = re.compile(
    r"(今だけ|期間限定|特別価格|限定価格|送料無料|半額|初回限定|最安値|激安|格安|"
    r"ご購入はこちら|購入はこちら|お申し込みはこちら|詳細はこちら|無料登録|登録無料|"
    r"LINE登録|友だち追加|アフィリエイト|割引クーポン|最大\d+%OFF|無料体験|今すぐ登録|"
    r"月利|元本保証|高収入|日給|簡単作業|スマホだけ|未経験歓迎|現金プレゼント|"
    r"当選しました|懸賞当選|還付金|不正ログイン|アカウント停止|パスワード再設定|"
    r"口座番号|暗証番号|事前振込|架空請求|ワンクリック|稼ぐ方法|稼げる|儲かる|"
    r"不労所得|必ず儲かる|確実に儲かる|スマホ副業|副業|せどり|バイナリーオプション|"
    r"FX自動売買|キャッシング|消費者金融|カードローン|即日融資|審査なし|"
    r"全額返金|返金保証|通販|サプリ|資料請求|"
    r"出会い系|パパ活|風俗|アダルト|エロ|セフレ|援助交際|"
    r"オンラインカジノ|オンカジ|パチンコ必勝|広告|PR記事)"
)
SYM_RE = re.compile(r"[!！★☆◆■●▲▼♪⇒→※【】]")


def read_row_group_with_retry(pf: pq.ParquetFile, rg: int, cols: list[str]):
    """通信断等に備えてリトライ付きで Parquet の 1 row group を読み取る。"""
    for attempt in range(4):
        try:
            return pf.read_row_group(rg, columns=cols)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def collect_benign_web(target_n: int, max_per_host: int) -> tuple[list[str], dict]:
    """FineWeb-2 jpn_Jpan からフィルタ・重複/ホスト制限を適用して target_n 件収集する。"""
    fs = HfFileSystem()
    kept: list[str] = []
    seen_md5: set[str] = set()
    host_counts: Counter = Counter()
    dropped: Counter = Counter()
    scanned = 0
    rgs_read = 0
    shards_used: list[str] = []
    cols = ["text", "url", "language", "language_score"]

    for shard in PARQUET_SHARDS:
        if len(kept) >= target_n:
            break
        shards_used.append(shard)
        with fs.open(shard, "rb") as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                if len(kept) >= target_n:
                    break
                tbl = read_row_group_with_retry(pf, rg, cols)
                rgs_read += 1
                texts = tbl["text"].to_pylist()
                urls = tbl["url"].to_pylist()
                langs = tbl["language"].to_pylist()
                scores = tbl["language_score"].to_pylist()
                for raw_t, raw_u, lang, score in zip(texts, urls, langs, scores):
                    if len(kept) >= target_n:
                        break
                    scanned += 1
                    if not isinstance(raw_t, str):
                        dropped["non_str"] += 1
                        continue
                    t = raw_t.strip()
                    if not (200 <= len(t) <= 2000):
                        dropped["length"] += 1
                        continue
                    if lang != "jpn" or (score is not None and score < 0.90):
                        dropped["lang_or_score"] += 1
                        continue
                    non_ws = re.sub(r"\s+", "", t)
                    if not non_ws or (len(JP_RE.findall(non_ws)) / len(non_ws)) < 0.65:
                        dropped["jp_ratio"] += 1
                        continue
                    if URL_RE.search(t):
                        dropped["url_in_text"] += 1
                        continue
                    if BAD_RE.search(t):
                        dropped["bad_keyword"] += 1
                        continue
                    if len(SYM_RE.findall(t)) >= 12:
                        dropped["symbol_excess"] += 1
                        continue
                    h = hashlib.md5(t.encode("utf-8")).hexdigest()
                    if h in seen_md5:
                        dropped["duplicate_md5"] += 1
                        continue
                    host = ""
                    if isinstance(raw_u, str) and raw_u:
                        host = urlparse(raw_u).netloc.lower()
                    if host and host_counts[host] >= max_per_host:
                        dropped["host_cap"] += 1
                        continue
                    seen_md5.add(h)
                    if host:
                        host_counts[host] += 1
                    kept.append(t)

    lengths = [len(x) for x in kept]
    stats = {
        "source": "HuggingFaceFW/fineweb-2 (jpn_Jpan, ODC-By 1.0)",
        "shards_used": shards_used,
        "row_groups_read": rgs_read,
        "rows_scanned": scanned,
        "collected": len(kept),
        "unique_hosts": len(host_counts),
        "max_per_host": max_per_host,
        "dropped_reasons": dict(dropped),
        "char_length": {
            "min": min(lengths) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
    }
    return kept, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benign_web.jsonl")
    ap.add_argument("--stats", default="data/prod/benign-web-stats.json")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--max-per-host", type=int, default=2)
    args = ap.parse_args()

    kept, stats = collect_benign_web(args.n, args.max_per_host)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in kept:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
