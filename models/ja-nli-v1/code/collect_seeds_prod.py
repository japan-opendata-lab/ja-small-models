#!/usr/bin/env python3
"""第11号 ja-nli-v1 本番工程向け種コーパス拡張収集・human-eval完全分離・3プール分割スクリプト。

PDL1.0 / 著作権法第13条の公的文書と GitHub API で SPDX (MIT / Apache-2.0) を一次確認した
OSS 日本語公式文書のみを使用し、第1工程の human-eval (600件+元プール650チャンク) と
文書・クラスタ・LSH レベルで完全に重複排除したうえで train / dev / sealed_test を構築する。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Callable, Dict, List, Set, Tuple
from urllib.parse import urljoin

import requests

ROOT = Path(".")
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import collect_seeds as cs  # noqa: E402
import nli_rules as nr  # noqa: E402

SEEDS_PROD_DIR = ROOT / "data" / "seeds_prod"
CACHE_DIR = cs.CACHE_DIR


def collect_html_index_series(
    session: requests.Session,
    specs: List[dict],
    max_workers: int = 8,
) -> List[dict]:
    """目次HTMLから規則一致リンクを抽出して並列取得する共通コレクタ。"""
    docs: List[dict] = []
    for sp in specs:
        src = sp["source"]
        idx_url = sp["index_url"]
        idx_html = cs.fetch_text(session, idx_url, sp["cache_index"])
        if not idx_html:
            print(f"  [WARN] index fetch failed: {src} -> {idx_url}")
            continue
        link_fn: Callable[[str], bool] = sp["link_filter"]
        links = sorted(
            {
                urljoin(idx_url, m)
                for m in re.findall(r'href="([^"]+\.html)"', idx_html)
                if link_fn(urljoin(idx_url, m))
            }
        )

        def _dl(u: str) -> Tuple[str, str]:
            h = hashlib.md5(u.encode("utf-8")).hexdigest()[:12]
            return u, cs.fetch_text(session, u, f"{sp['cache_prefix']}/{h}.html") or ""

        part_docs = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for u, html in ex.map(_dl, links):
                if not html or "お探しのページが見つかりません" in html:
                    continue
                paras = cs.extract_html_paragraphs(html)
                if not paras:
                    continue
                h = hashlib.md5(u.encode("utf-8")).hexdigest()[:12]
                fn = u.rsplit("/", 1)[-1]
                # 第1工程の既存 doc_id 命名と揃えて human-eval 重複判定と確実に整合させる
                doc_id = sp["doc_id_fn"](u, fn, h)
                docs.append({
                    "doc_id": doc_id,
                    "source": src,
                    "domain_group": sp["domain_group"],
                    "license": "PDL1.0",
                    "url": u,
                    "paras": paras,
                })
                part_docs += 1
        print(f"  {src}: {part_docs} docs ({len(links)} links)")
    return docs


def collect_gov_prod(session: requests.Session) -> List[dict]:
    """第1工程ソース＋新規確認済み PDL1.0 白書・デジタル庁・現代法令を収集する。"""
    docs: List[dict] = []

    # 1. 第1工程と共通の厚生労働白書(roudou)はそのまま再利用
    #    (doc_id 形式が第1工程と一致するため raw_cache も完全ヒットする)
    specs: List[dict] = []

    # 総務省 情報通信白書 R6..R1, H30
    for yr in ["r06", "r05", "r04", "r03", "r02", "r01", "h30"]:
        toc_url = f"https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/{yr}/html/nc000000.html"
        specs.append({
            "source": f"soumu_{yr}",
            "domain_group": "gov_whitepaper",
            "index_url": toc_url,
            "cache_index": f"soumu/{yr}_toc.html",
            "cache_prefix": f"soumu_prod/{yr}",
            "link_filter": lambda full, y=yr: f"/whitepaper/ja/{y}/html/" in full
            and not full.endswith(("nb000000.html", "nc000000.html")),
            "doc_id_fn": lambda u, fn, _h, y=yr: f"soumu_{y}_{fn}",
        })

    # 環境省 環境白書 R6..R1, H30..H27
    for yr in ["r06", "r05", "r04", "r03", "r02", "r01", "h30", "h29", "h28", "h27"]:
        specs.append({
            "source": f"env_{yr}",
            "domain_group": "gov_whitepaper",
            "index_url": f"https://www.env.go.jp/policy/hakusyo/{yr}/index.html",
            "cache_index": f"env/{yr}_idx.html",
            "cache_prefix": f"env_prod/{yr}",
            "link_filter": lambda full, y=yr: f"/hakusyo/{y}/html/" in full,
            "doc_id_fn": lambda u, fn, _h, y=yr: f"env_{y}_{fn}",
        })

    # 厚労省 労働経済白書 (24, 23, 21)
    for yr in ["24", "23", "21"]:
        specs.append({
            "source": f"mhlw_rodo_{yr}",
            "domain_group": "gov_whitepaper",
            "index_url": f"https://www.mhlw.go.jp/stf/wp/hakusyo/roudou/{yr}/index.html",
            "cache_index": f"mhlw/rodo_{yr}_idx.html",
            "cache_prefix": f"mhlw_prod/rodo_{yr}",
            "link_filter": lambda full, y=yr: f"/hakusyo/roudou/{y}/" in full
            and "backdata" not in full
            and not full.endswith("index.html"),
            "doc_id_fn": lambda u, fn, _h, y=yr: f"mhlw_rodo_{y}_{fn}",
        })

    # 国交省 国土交通白書 R6..R2
    mlit_years = [
        ("mlit/r05/hakusho/r06", "r06"),
        ("mlit/r04/hakusho/r05", "r05"),
        ("mlit/r03/hakusho/r04", "r04"),
        ("mlit/r02/hakusho/r03", "r03"),
        ("mlit/r01/hakusho/r02", "r02"),
    ]
    for yr_dir, yr_lbl in mlit_years:
        specs.append({
            "source": f"mlit_{yr_lbl}",
            "domain_group": "gov_whitepaper",
            "index_url": f"https://www.mlit.go.jp/hakusyo/{yr_dir}/index.html",
            "cache_index": f"mlit/{yr_lbl}_idx.html",
            "cache_prefix": f"mlit_prod/{yr_lbl}",
            "link_filter": lambda full, d=yr_dir: f"/hakusyo/{d}/html/" in full,
            "doc_id_fn": lambda u, fn, _h, y=yr_lbl: f"mlit_{y}_{fn}",
        })

    # 内閣府 防災白書 R6..R3 (PDL1.0)
    for yr in ["r06", "r05", "r04", "r03"]:
        specs.append({
            "source": f"cao_bousai_{yr}",
            "domain_group": "gov_whitepaper",
            "index_url": f"https://www.bousai.go.jp/kaigirep/hakusho/{yr}/honbun/index.html",
            "cache_index": f"cao/bousai_{yr}_idx.html",
            "cache_prefix": f"cao_prod/bousai_{yr}",
            "link_filter": lambda full, y=yr: f"/kaigirep/hakusho/{y}/honbun/" in full
            and not full.endswith("index.html"),
            "doc_id_fn": lambda _u, fn, _h, y=yr: f"cao_bousai_{y}_{fn}",
        })

    # 内閣府 高齢社会白書 R6(2024), R5(2023) (PDL1.0)
    for w_yr, lbl in [("w-2024", "r06"), ("w-2023", "r05")]:
        specs.append({
            "source": f"cao_kourei_{lbl}",
            "domain_group": "gov_whitepaper",
            "index_url": f"https://www8.cao.go.jp/kourei/whitepaper/{w_yr}/html/zenbun/index.html",
            "cache_index": f"cao/kourei_{lbl}_idx.html",
            "cache_prefix": f"cao_prod/kourei_{lbl}",
            "link_filter": lambda full, w=w_yr: f"/kourei/whitepaper/{w}/html/zenbun/" in full
            and not full.endswith("index.html"),
            "doc_id_fn": lambda _u, fn, _h, y=lbl: f"cao_kourei_{y}_{fn}",
        })

    # 内閣府 男女共同参画白書 R6, R5 (PDL1.0)
    for yr in ["r06", "r05"]:
        specs.append({
            "source": f"cao_danjo_{yr}",
            "domain_group": "gov_whitepaper",
            "index_url": f"https://www.gender.go.jp/about_danjo/whitepaper/{yr}/zentai/index.html",
            "cache_index": f"cao/danjo_{yr}_idx.html",
            "cache_prefix": f"cao_prod/danjo_{yr}",
            "link_filter": lambda full, y=yr: f"/about_danjo/whitepaper/{y}/zentai/html/" in full,
            "doc_id_fn": lambda _u, fn, h, y=yr: f"cao_danjo_{y}_{h}_{fn}",
        })

    # 内閣府 経済財政白書 R6(24), R5(23) (PDL1.0)
    for yr in ["24", "23"]:
        specs.append({
            "source": f"cao_je{yr}",
            "domain_group": "gov_whitepaper",
            "index_url": f"https://www5.cao.go.jp/j-j/wp/wp-je{yr}/index.html",
            "cache_index": f"cao/je{yr}_idx.html",
            "cache_prefix": f"cao_prod/je{yr}",
            "link_filter": lambda full, y=yr: f"/j-j/wp/wp-je{y}/h" in full,
            "doc_id_fn": lambda _u, fn, _h, y=yr: f"cao_je{y}_{fn}",
        })

    docs.extend(collect_html_index_series(session, specs, max_workers=10))

    # デジタル庁全サイトマップ (max_pages=4500)
    dg_docs = collect_digital_go_expanded(session, max_pages=4500)
    print(f"  digital_go: {len(dg_docs)} docs")
    docs.extend(dg_docs)

    # e-Gov 法令 API v2 (現代語平成・令和法律のみ拡張、後段で <=15% 上限管理)
    law_docs = collect_egov_laws_expanded(session, max_laws=700, max_articles=80)
    print(f"  egov_laws: {len(law_docs)} docs")
    docs.extend(law_docs)

    return docs


def collect_digital_go_expanded(session: requests.Session, max_pages: int = 4500) -> List[dict]:
    root_sm = cs.fetch_text(session, "https://www.digital.go.jp/sitemap.xml", "digital_go/root_sm.xml")
    sm_urls: List[str] = []
    if root_sm:
        sm_urls = re.findall(r"<loc>([^<]+)</loc>", root_sm)
    if not sm_urls:
        sm_urls = [f"https://www.digital.go.jp/sitemap.xml?page={p}" for p in range(1, 15)]

    urls: List[str] = []
    for idx, sm_u in enumerate(sm_urls, 1):
        sm = cs.fetch_text(session, sm_u, f"digital_go/sm_{idx}.xml")
        if not sm:
            continue
        for u in re.findall(r"<loc>([^<]+)</loc>", sm):
            if not u.startswith("https://www.digital.go.jp/en/") and any(
                seg in u for seg in ["/policies/", "/resources/", "/news/", "/councils/"]
            ):
                urls.append(u)
    urls = sorted(set(urls))
    rng = random.Random(42)
    rng.shuffle(urls)
    urls = urls[:max_pages]

    def _dl(u: str):
        h = hashlib.md5(u.encode()).hexdigest()[:16]
        return u, h, cs.fetch_text(session, u, f"digital_go/pages/{h}.html")

    docs: List[dict] = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for u, h, html in ex.map(_dl, urls):
            if not html:
                continue
            paras = cs.extract_html_paragraphs(html)
            if paras:
                docs.append({
                    "doc_id": f"digital_go_{h}",
                    "source": "digital_go",
                    "domain_group": "gov_policy",
                    "license": "PDL1.0",
                    "url": u,
                    "paras": paras,
                })
    return docs


def collect_egov_laws_expanded(
    session: requests.Session, max_laws: int = 700, max_articles: int = 80
) -> List[dict]:
    law_list: List[dict] = []
    seen_ids: Set[str] = set()
    for offset in range(0, 2400, 200):
        raw = cs.fetch_text(
            session,
            f"https://laws.e-gov.go.jp/api/2/laws?law_type=Act&limit=200&offset={offset}",
            f"egov/list_{offset}.json",
            is_json=True,
        )
        if not raw:
            continue
        data = json.loads(raw)
        for item in data.get("laws", []):
            li = item.get("law_info", {})
            ri = item.get("revision_info", {})
            lid = li.get("law_id")
            title = ri.get("law_title", "")
            era = li.get("law_num_era", "")
            if (
                lid
                and lid not in seen_ids
                and title
                and era in ("Heisei", "Reiwa")
                and "一部を改正する" not in title
            ):
                seen_ids.add(lid)
                law_list.append({"law_id": lid, "title": title})
    rng = random.Random(42)
    rng.shuffle(law_list)
    law_list = law_list[:max_laws]

    def _dl(item: dict):
        lid = item["law_id"]
        raw = cs.fetch_text(
            session,
            f"https://laws.e-gov.go.jp/api/2/law_data/{lid}",
            f"egov/laws/{lid}.json",
            is_json=True,
        )
        return item, raw

    docs: List[dict] = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        for item, raw in ex.map(_dl, law_list):
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except Exception:
                continue
            lf = d.get("law_full_text", {})
            articles: List[dict] = []
            cs._find_egov_articles(lf, articles)
            paras: List[str] = []
            title = item["title"]
            for art in articles[:max_articles]:
                art_title = ""
                sent_parts: List[str] = []
                for ch in art.get("children", []):
                    if not isinstance(ch, dict):
                        continue
                    tg = ch.get("tag")
                    if tg == "ArticleTitle":
                        art_title = cs._extract_egov_sentences(ch)
                    elif tg == "Paragraph":
                        for pch in ch.get("children", []):
                            if isinstance(pch, dict) and pch.get("tag") == "ParagraphSentence":
                                ps = cs._extract_egov_sentences(pch)
                                if ps:
                                    sent_parts.append(ps)
                body = "".join(sent_parts).strip()
                if len(body) < 45 or "。" not in body:
                    continue
                if len(re.findall(r"[ぁ-ん]", body)) < 10:
                    continue
                prefix = f"{title}{art_title}では、" if art_title else f"{title}では、"
                full = prefix + body
                if 55 <= len(full) <= 1000 and nr.brackets_balanced(full):
                    paras.append(full)
            if paras:
                docs.append({
                    "doc_id": f"egov_{item['law_id']}",
                    "source": "egov_laws",
                    "domain_group": "gov_law",
                    "license": "Art13_PDL1.0",
                    "url": f"https://laws.e-gov.go.jp/law/{item['law_id']}",
                    "paras": paras,
                })
    return docs


OSS_SOURCES_PROD = list(cs.OSS_SOURCES) + [
    {
        "source": "ms_mcp_ja",
        "repo": "microsoft/mcp-for-beginners",
        "branch": "main",
        "license": "MIT",
        "filter_fn": lambda p: (".ja.md" in p or "translations/ja/" in p) and p.endswith(".md"),
    },
    {
        "source": "ms_agents_ja",
        "repo": "microsoft/ai-agents-for-beginners",
        "branch": "main",
        "license": "MIT",
        "filter_fn": lambda p: (".ja.md" in p or "translations/ja/" in p) and p.endswith(".md"),
    },
    {
        "source": "astro_ja",
        "repo": "withastro/docs",
        "branch": "main",
        "license": "MIT",
        "filter_fn": lambda p: p.startswith("src/content/docs/ja/") and p.endswith((".md", ".mdx")),
    },
    {
        "source": "tauri_ja",
        "repo": "tauri-apps/tauri-docs",
        "branch": "v2",
        "license": "MIT",
        "filter_fn": lambda p: p.startswith("src/content/docs/ja/") and p.endswith((".md", ".mdx")),
    },
    {
        "source": "biome_ja",
        "repo": "biomejs/website",
        "branch": "main",
        "license": "MIT",
        "filter_fn": lambda p: p.startswith("src/content/docs/ja/") and p.endswith((".md", ".mdx")),
    },
    {
        "source": "rust_rbe_ja",
        "repo": "rust-lang-ja/rust-by-example-ja",
        "branch": "master",
        "license": "Apache-2.0",
        "filter_fn": lambda p: p.startswith("src/") and p.endswith(".md"),
    },
    {
        "source": "hf_diffusers_ja",
        "repo": "huggingface/diffusers",
        "branch": "main",
        "license": "Apache-2.0",
        "filter_fn": lambda p: "/ja/" in p and p.endswith((".md", ".mdx")),
    },
]


def verify_github_spdx(session: requests.Session, repo: str, expected_lic: str, src: str) -> str:
    """GitHub API メタデータで SPDX ライセンスが MIT または Apache-2.0 であることをプログラム検査する。"""
    meta_raw = cs.fetch_text(
        session, f"https://api.github.com/repos/{repo}", f"gh/{src}_meta.json", is_json=True
    )
    if not meta_raw:
        raise RuntimeError(f"SPDX metadata fetch failed for {repo}")
    meta = json.loads(meta_raw)
    spdx = (meta.get("license") or {}).get("spdx_id")
    if spdx not in ("MIT", "Apache-2.0") or spdx != expected_lic:
        raise RuntimeError(f"License gate failed for {repo}: got {spdx}, expected {expected_lic}")
    return spdx


def collect_oss_prod(session: requests.Session) -> List[dict]:
    docs: List[dict] = []
    for cfg in OSS_SOURCES_PROD:
        src = cfg["source"]
        repo = cfg["repo"]
        br = cfg["branch"]
        lic = verify_github_spdx(session, repo, cfg["license"], src)
        tree_raw = cs.fetch_text(
            session,
            f"https://api.github.com/repos/{repo}/git/trees/{br}?recursive=1",
            f"gh/{src}_tree.json",
            is_json=True,
        )
        if not tree_raw:
            print(f"  [WARN] failed tree for {repo}")
            continue
        tree = json.loads(tree_raw).get("tree", [])
        paths = [t["path"] for t in tree if t.get("type") == "blob" and cfg["filter_fn"](t["path"])]
        print(f"  GH {src} ({lic}): {len(paths)} files matched")

        def _dl(p: str):
            raw_url = f"https://raw.githubusercontent.com/{repo}/{br}/{p}"
            h = hashlib.md5(p.encode()).hexdigest()[:16]
            return p, raw_url, cs.fetch_text(session, raw_url, f"gh/{src}/{h}.md")

        with ThreadPoolExecutor(max_workers=16) as ex:
            for p, raw_url, md in ex.map(_dl, paths):
                if not md:
                    continue
                paras = cs.clean_markdown_to_paragraphs(md)
                if paras:
                    docs.append({
                        "doc_id": f"{src}::{p}",
                        "source": src,
                        "domain_group": "oss_docs",
                        "license": lic,
                        "url": f"https://github.com/{repo}/blob/{br}/{p}",
                        "paras": paras,
                    })
    return docs


def load_human_eval_exclusion() -> Tuple[Set[str], List[str], Set[str]]:
    """Stage 1 human_eval プールおよび確定 human-eval.jsonl の文書ID・前提を全読み込みする。"""
    he_doc_ids: Set[str] = set()
    he_premises: List[str] = []

    he_eval_path = ROOT / "data" / "eval" / "human-eval.jsonl"
    for ln in he_eval_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        he_doc_ids.add(r["doc_id"])
        he_premises.append(r["premise"])

    s1_chunks = ROOT / "data" / "seeds" / "chunks_all.jsonl"
    if s1_chunks.exists():
        for ln in s1_chunks.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("split_pool") == "human_eval":
                he_doc_ids.add(r["doc_id"])
                he_premises.append(r["premise"])

    unique_prems = list(dict.fromkeys(he_premises))
    norm_prems = {nr.normalize_for_dedup(p) for p in unique_prems if nr.normalize_for_dedup(p)}
    return he_doc_ids, unique_prems, norm_prems


def build_premise_lsh_index(
    texts: List[str], *, ngram_n: int = 5, num_perm: int = 64, bands: int = 16, seed: int = 42
):
    rows = num_perm // bands
    grams_list = [nr.char_ngrams(nr.normalize_for_dedup(t) or t, n=ngram_n) for t in texts]
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
    for idx, t in enumerate(texts):
        sig = nr.compute_minhash_sig(t, ngram_n=ngram_n, num_perm=num_perm, seed=seed)
        for b in range(bands):
            st = b * rows
            buckets[(b, sig[st : st + rows])].append(idx)
    return grams_list, buckets, rows, bands, ngram_n, num_perm, seed


def has_premise_lsh_hit(text: str, idx_tuple, jaccard_thresh: float = 0.75) -> bool:
    grams_list, buckets, rows, bands, ngram_n, num_perm, seed = idx_tuple
    sig = nr.compute_minhash_sig(text, ngram_n=ngram_n, num_perm=num_perm, seed=seed)
    st_grams = nr.char_ngrams(nr.normalize_for_dedup(text) or text, n=ngram_n)
    if not st_grams:
        return False
    checked: Set[int] = set()
    for b in range(bands):
        st = b * rows
        for ref_i in buckets.get((b, sig[st : st + rows]), ()):
            if ref_i in checked:
                continue
            checked.add(ref_i)
            sg = grams_list[ref_i]
            if not sg:
                continue
            inter = len(st_grams & sg)
            if inter / (len(st_grams) + len(sg) - inter) >= jaccard_thresh:
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chunks", type=int, default=42000)
    ap.add_argument("--max-law-ratio", type=float, default=0.15)
    ap.add_argument("--dev-cap", type=int, default=3500)
    ap.add_argument("--sealed-cap", type=int, default=3500)
    args = ap.parse_args()

    t0 = time.time()
    rng = random.Random(args.seed)
    session = requests.Session()
    SEEDS_PROD_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Collecting expanded PDL1.0 / Art13 public documents...")
    gov_docs = collect_gov_prod(session)
    print(f"[2/4] Collecting & SPDX-verifying MIT / Apache-2.0 OSS documents...")
    oss_docs = collect_oss_prod(session)
    all_docs = gov_docs + oss_docs
    print(f"Total docs extracted: {len(all_docs)} (gov={len(gov_docs)}, oss={len(oss_docs)})")

    print("[3/4] Chunking, LSH doc clustering, & strict human-eval cluster exclusion...")
    doc_records: List[dict] = []
    seen_chunk_norms: Set[str] = set()
    for doc in all_docs:
        chks = cs.chunk_paragraphs(doc["paras"], rng)
        dedup_chks = []
        for lb, txt in chks:
            nk = nr.normalize_for_dedup(txt)
            if not nk or nk in seen_chunk_norms:
                continue
            seen_chunk_norms.add(nk)
            dedup_chks.append((lb, txt))
        if not dedup_chks:
            continue
        rep_text = " ".join(c[1] for c in dedup_chks[:3])[:1500]
        doc_records.append({**doc, "chunks": dedup_chks, "rep_text": rep_text})

    # 文書クラスタリング (Jaccard >= 0.70)
    rep_texts = [d["rep_text"] for d in doc_records]
    dup_pairs = nr.lsh_find_near_duplicates(rep_texts, ngram_n=5, jaccard_thresh=0.70)
    uf = cs.UnionFind(len(doc_records))
    for u, v, _ in dup_pairs:
        uf.union(u, v)

    he_doc_ids, he_premises, he_norm_prems = load_human_eval_exclusion()
    he_lsh_idx = build_premise_lsh_index(he_premises, ngram_n=5)

    clusters: Dict[int, List[dict]] = defaultdict(list)
    for idx, d in enumerate(doc_records):
        cid = uf.find(idx)
        d["cluster_id"] = f"cl_{cid:06d}"
        clusters[cid].append(d)

    # human-eval に doc_id または premise LSH >= 0.75 で触れる全クラスタを永久除外
    tainted_cids: Set[int] = set()
    for cid, dlist in clusters.items():
        if any(d["doc_id"] in he_doc_ids for d in dlist):
            tainted_cids.add(cid)
            continue
        hit = False
        for d in dlist:
            for _lb, txt in d["chunks"]:
                nk = nr.normalize_for_dedup(txt)
                if nk in he_norm_prems or has_premise_lsh_hit(txt, he_lsh_idx, jaccard_thresh=0.75):
                    hit = True
                    break
            if hit:
                break
        if hit:
            tainted_cids.add(cid)

    clean_clusters = {cid: dl for cid, dl in clusters.items() if cid not in tainted_cids}
    print(
        f"Docs={len(doc_records)}, total_clusters={len(clusters)}, "
        f"tainted_he_clusters_removed={len(tainted_cids)}, clean_clusters={len(clean_clusters)}"
    )

    # 3プールへソース均等割当 (dev ~8%, sealed_test ~8%, train ~84%)
    pool_names = ["dev", "sealed_test", "train"]
    pool_ratios = {"dev": 0.085, "sealed_test": 0.085, "train": 0.830}

    by_src_clusters: Dict[str, List[List[dict]]] = defaultdict(list)
    for _cid, dlist in sorted(clean_clusters.items()):
        by_src_clusters[dlist[0]["source"]].append(dlist)

    pool_chunks: Dict[str, List[dict]] = {p: [] for p in pool_names}
    for src, clist in sorted(by_src_clusters.items()):
        rng.shuffle(clist)
        total_src_chunks = sum(len(d["chunks"]) for dl in clist for d in dl)
        quotas = {p: max(1, int(total_src_chunks * pool_ratios[p])) for p in pool_names}
        assigned = Counter()
        for dl in clist:
            best_p = min(
                pool_names,
                key=lambda p: (assigned[p] / max(1, quotas[p]), 0 if p != "train" else 1),
            )
            dl_chunks = []
            for d in dl:
                for idx, (lb, txt) in enumerate(d["chunks"]):
                    dl_chunks.append({
                        "chunk_id": f"{d['doc_id']}#c{idx:03d}",
                        "doc_id": d["doc_id"],
                        "cluster_id": d["cluster_id"],
                        "split_pool": best_p,
                        "source": d["source"],
                        "domain_group": d["domain_group"],
                        "license": d["license"],
                        "url": d["url"],
                        "len_bucket": lb,
                        "premise": txt,
                    })
            pool_chunks[best_p].extend(dl_chunks)
            assigned[best_p] += len(dl_chunks)

    print("Raw clean chunks per pool:", {p: len(v) for p, v in pool_chunks.items()})

    # 各プールからドメインバランス (law <= 15%, oss 最大 40%) でバランス選定
    def sample_balanced(items: List[dict], cap: int) -> List[dict]:
        law = [x for x in items if x["domain_group"] == "gov_law"]
        gov = [x for x in items if x["domain_group"] in ("gov_whitepaper", "gov_policy")]
        oss = [x for x in items if x["domain_group"] == "oss_docs"]
        rng.shuffle(law)
        rng.shuffle(gov)
        rng.shuffle(oss)
        # dev / sealed_test / train で law<=15% を厳守しつつ残りを gov と oss で最大化
        n_law = min(len(law), int(cap * args.max_law_ratio))
        n_oss = min(len(oss), int(cap * 0.40))
        n_gov = min(len(gov), cap - n_law - n_oss)
        picked = law[:n_law] + gov[:n_gov] + oss[:n_oss]
        if len(picked) < cap:
            rem = gov[n_gov:] + oss[n_oss:]
            rng.shuffle(rem)
            take = rem[: cap - len(picked)]
            picked.extend(take)
            # 念のため gov_law 割合を再確認し 15% 超なら調整不要（rem は law 以外のみ）
        rng.shuffle(picked)
        return picked

    dev_cap = min(len(pool_chunks["dev"]), args.dev_cap)
    sealed_cap = min(len(pool_chunks["sealed_test"]), args.sealed_cap)
    train_cap = min(len(pool_chunks["train"]), args.max_chunks - dev_cap - sealed_cap)

    final_chunks: List[dict] = []
    pool_stats = {}
    for p, cap in [("dev", dev_cap), ("sealed_test", sealed_cap), ("train", train_cap)]:
        picked = sample_balanced(pool_chunks[p], cap)
        final_chunks.extend(picked)
        pool_stats[p] = {
            "count": len(picked),
            "available": len(pool_chunks[p]),
            "by_domain": dict(Counter(x["domain_group"] for x in picked)),
            "by_source": dict(Counter(x["source"] for x in picked)),
            "by_len_bucket": dict(Counter(x["len_bucket"] for x in picked)),
        }

    # 最終安全確認：human-eval との doc_id / premise LSH 重複 0 件を assert
    sel_docs = {c["doc_id"] for c in final_chunks}
    assert len(sel_docs & he_doc_ids) == 0, f"human-eval doc leak: {len(sel_docs & he_doc_ids)}"
    he_lsh_hits = sum(
        1
        for c in final_chunks
        if has_premise_lsh_hit(c["premise"], he_lsh_idx, jaccard_thresh=0.75)
    )
    assert he_lsh_hits == 0, f"human-eval LSH leak: {he_lsh_hits}"

    chunks_path = SEEDS_PROD_DIR / "chunks_prod.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(final_chunks):
            row["seed_idx"] = i
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "seed": args.seed,
        "total_docs": len(doc_records),
        "total_clusters": len(clusters),
        "tainted_he_clusters_removed": len(tainted_cids),
        "clean_clusters": len(clean_clusters),
        "total_chunks_available_clean": sum(len(v) for v in pool_chunks.values()),
        "total_chunks_selected": len(final_chunks),
        "human_eval_doc_overlap": 0,
        "human_eval_premise_lsh075_overlap": 0,
        "chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "pools": pool_stats,
        "oss_spdx_verified": [
            {"source": c["source"], "repo": c["repo"], "license": c["license"]}
            for c in OSS_SOURCES_PROD
        ],
    }
    manifest_path = SEEDS_PROD_DIR / "seed-manifest-prod.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"[4/4] Done in {elapsed:.1f}s -> {chunks_path} ({len(final_chunks)} chunks)")
    for p, st in pool_stats.items():
        print(f"  {p}: {st['count']} (avail {st['available']}) | dom={st['by_domain']} | len={st['by_len_bucket']}")


if __name__ == "__main__":
    main()
