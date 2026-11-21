#!/usr/bin/env python3
"""第11号 ja-nli-v1 商用クリーン種コーパス収集・チャンク化・事前クラスタ分割スクリプト。

PDL1.0 等の公的文書（約60%）および MIT / Apache-2.0 の日本語 OSS 文書（約40%）のみを収集し、
既存 NLI データセット・CC BY-SA / NC ソースを一切排除した上で、
文書クラスタ単位で human_eval / dev / sealed_test / train の4プールへ事前分離する。
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nli_rules as nr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS_DIR = ROOT / "data" / "seeds"
CACHE_DIR = SEEDS_DIR / "raw_cache"

HEADERS = {"User-Agent": "ja-nli-v1-seed-collector/1.0 (+https://huggingface.co)"}

_BOILERPLATE_RE = re.compile(
    r"JavaScriptを使用|JavaScriptを有効|自動翻訳します|Adobe Reader|PDFファイルを見る|"
    r"サイトマップ|このページのトップへ|お問い合わせ先|Copyright|All Rights Reserved|"
    r"アクセシビリティ|プライバシーポリシー|Cookie|閲覧できない場合|"
    r"年（年度）の表記は|「年」とあるものは暦年|ページID|最終更新日|"
    r"ホーム\s*>\s*|目次へ戻る|前のページへ|次のページへ"
)
_RE_FOOTNOTE_NUM = re.compile(r"(?<=[ぁ-んァ-ヶ一-龥])[0-9]{1,2}(?=。)")


def fetch_text(
    session: requests.Session,
    url: str,
    cache_key: str,
    *,
    timeout: int = 20,
    is_json: bool = False,
) -> Optional[str]:
    """キャッシュ付き HTTP GET（文字コード自動判別）。"""
    cpath = CACHE_DIR / cache_key
    if cpath.exists():
        return cpath.read_text(encoding="utf-8", errors="ignore")
    try:
        r = session.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            return None
        if is_json:
            txt = r.text
        else:
            r.encoding = r.apparent_encoding or "utf-8"
            txt = r.text
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(txt, encoding="utf-8")
        return txt
    except Exception:
        return None


def split_sentences_jp(text: str) -> List[str]:
    """括弧深さを考慮した日本語文分割。"""
    sents: List[str] = []
    buf: List[str] = []
    d1 = d2 = d3 = 0
    for ch in text:
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
    tail = "".join(buf).strip()
    if tail:
        sents.append(tail)
    return sents


def is_valid_paragraph(text: str) -> bool:
    if len(text) < 35:
        return False
    if _BOILERPLATE_RE.search(text):
        return False
    if "。" not in text and "？" not in text:
        return False
    if nr.japanese_char_ratio(text) < 0.52:
        return False
    if not nr.brackets_balanced(text):
        return False
    if re.search(r"https?://|www\.", text):
        return False
    return True


def extract_html_paragraphs(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "table",
            "rt",
            "rp",
            "sup",
            "figure",
            "figcaption",
            "noscript",
        ]
    ):
        tag.decompose()
    for sel in [".breadcrumb", ".sidebar", ".toc", "#menu", "#header", "#footer", ".nav"]:
        for el in soup.select(sel):
            el.decompose()

    paras: List[str] = []
    seen: Set[str] = set()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    for el in main.find_all(["p", "dd", "li"]):
        if el.find(["p", "table", "ul", "ol"]):
            continue
        t = nr.clean_surface(el.get_text("", strip=False))
        t = _RE_FOOTNOTE_NUM.sub("", t)
        if not is_valid_paragraph(t):
            continue
        # 箇条書きの短い見出し項目を除外（文末句点必須）
        if t[-1] not in ("。", "？", "！"):
            continue
        k = nr.normalize_for_dedup(t)
        if k not in seen:
            seen.add(k)
            paras.append(t)
    return paras


def clean_markdown_to_paragraphs(md: str) -> List[str]:
    """Markdown からコード・表・リンク記法を除去し自然な日本語段落リストを返す。"""
    # YAML frontmatter
    md = re.sub(r"^---\s*\n.*?\n---\s*\n", "", md, count=1, flags=re.DOTALL)
    # HTML comments
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)
    # Fenced code blocks
    md = re.sub(r"```.*?```", "\n\n", md, flags=re.DOTALL)
    md = re.sub(r"~~~.*?~~~", "\n\n", md, flags=re.DOTALL)

    lines_out: List[str] = []
    for ln in md.splitlines():
        s = ln.strip()
        if not s:
            lines_out.append("")
            continue
        # 見出し・表・水平線・ディレクティブ・画像・HTMLタグのみの行を段落区切りに
        if (
            s.startswith("#")
            or s.startswith("|")
            or s.startswith("---")
            or s.startswith("===")
            or s.startswith(":::")
            or s.startswith("///")
            or s.startswith("![")
            or re.match(r"^<[A-Za-z/!][^>]*>$", s)
        ):
            lines_out.append("")
            continue
        # Admonition / blockquote prefix
        s = re.sub(r"^(?:>+\s*|!!!\s*\w+.*|[-*+]\s+|\d+\.\s+)", "", s)
        # Markdown links & images
        s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
        s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
        s = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", s)
        # Inline tags (e.g. <br>, <a ...>)
        s = re.sub(r"<[^>]+>", "", s)
        # Bold/italic/inline code backticks (コード記法に見える長いもの以外)
        s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
        s = re.sub(r"__([^_]+)__", r"\1", s)
        s = re.sub(r"`([^`]{1,40})`", r"\1", s)
        # 残ったバッククォートやURLがある行はスキップ
        if "`" in s or "http://" in s or "https://" in s:
            lines_out.append("")
            continue
        lines_out.append(s)

    blocks = re.split(r"\n\s*\n", "\n".join(lines_out))
    paras: List[str] = []
    seen: Set[str] = set()
    for b in blocks:
        t = nr.clean_surface(b.replace("\n", ""))
        if not is_valid_paragraph(t):
            continue
        if t[-1] not in ("。", "？", "！"):
            continue
        # 英語率や記号率が高すぎる技術断片を除外
        if len(re.findall(r"[A-Za-z_/{}()=;\[\]]", t)) / max(1, len(t)) > 0.35:
            continue
        k = nr.normalize_for_dedup(t)
        if k not in seen:
            seen.add(k)
            paras.append(t)
    return paras


def chunk_paragraphs(paras: List[str], rng: random.Random) -> List[Tuple[str, str]]:
    """段落群から重複なしで short(80-180), medium(180-450), long(450-1200) チャンクを切り出す。"""
    # 全文リストへ（各段落内の文を単位にして境界が切れないようにする）
    chunks: List[Tuple[str, str]] = []
    bucket_cycle = ["medium", "short", "medium", "long", "short", "medium", "medium", "long", "short", "medium"]
    rng.shuffle(bucket_cycle)
    b_idx = 0

    def target_range(b: str) -> Tuple[int, int]:
        if b == "short":
            return (80, 180)
        if b == "medium":
            return (180, 450)
        return (450, 1150)

    cur_sents: List[str] = []
    cur_len = 0

    for p in paras:
        sents = [s for s in split_sentences_jp(p) if len(s) >= 12]
        for s in sents:
            b = bucket_cycle[b_idx % len(bucket_cycle)]
            lo, hi = target_range(b)
            if cur_len > 0 and cur_len + len(s) > hi:
                txt = "".join(cur_sents).strip()
                if 75 <= len(txt) <= 1250 and nr.brackets_balanced(txt):
                    actual_b = "short" if len(txt) < 180 else ("medium" if len(txt) < 450 else "long")
                    chunks.append((actual_b, txt))
                    b_idx += 1
                cur_sents = [s]
                cur_len = len(s)
            else:
                cur_sents.append(s)
                cur_len += len(s)
                b = bucket_cycle[b_idx % len(bucket_cycle)]
                lo, hi = target_range(b)
                if cur_len >= lo and (b == "short" or cur_len >= (lo + hi) // 2):
                    txt = "".join(cur_sents).strip()
                    if 75 <= len(txt) <= 1250 and nr.brackets_balanced(txt):
                        actual_b = "short" if len(txt) < 180 else ("medium" if len(txt) < 450 else "long")
                        chunks.append((actual_b, txt))
                        b_idx += 1
                    cur_sents = []
                    cur_len = 0

    if cur_sents:
        txt = "".join(cur_sents).strip()
        if 80 <= len(txt) <= 1250 and nr.brackets_balanced(txt):
            actual_b = "short" if len(txt) < 180 else ("medium" if len(txt) < 450 else "long")
            chunks.append((actual_b, txt))
    return chunks


# --- 1. 公的文書コレクタ ---

def collect_soumu(session: requests.Session) -> List[dict]:
    docs = []
    for yr in ["r06", "r05", "r04", "r03", "r02", "r01"]:
        toc_url = f"https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/{yr}/html/nc000000.html"
        toc_html = fetch_text(session, toc_url, f"soumu/{yr}_toc.html")
        if not toc_html:
            continue
        links = sorted(
            set(
                urljoin(toc_url, m)
                for m in re.findall(r'href="([^"]+\.html)"', toc_html)
                if f"/whitepaper/ja/{yr}/html/" in urljoin(toc_url, m)
                and not m.endswith(("nb000000.html", "nc000000.html"))
            )
        )
        def _dl(u):
            fn = u.rsplit("/", 1)[-1]
            html = fetch_text(session, u, f"soumu/{yr}/{fn}")
            return u, html

        with ThreadPoolExecutor(max_workers=12) as ex:
            for u, html in ex.map(_dl, links):
                if not html or "お探しのページが見つかりません" in html:
                    continue
                paras = extract_html_paragraphs(html)
                if paras:
                    docs.append({
                        "doc_id": f"soumu_{yr}_{u.rsplit('/', 1)[-1]}",
                        "source": f"soumu_{yr}",
                        "domain_group": "gov_whitepaper",
                        "license": "PDL1.0",
                        "url": u,
                        "paras": paras,
                    })
    return docs


def collect_env(session: requests.Session) -> List[dict]:
    docs = []
    for yr in ["r06", "r05", "r04", "r03", "r02", "r01"]:
        idx_url = f"https://www.env.go.jp/policy/hakusyo/{yr}/index.html"
        idx_html = fetch_text(session, idx_url, f"env/{yr}_idx.html")
        if not idx_html:
            continue
        links = sorted(
            set(
                urljoin(idx_url, m)
                for m in re.findall(r'href="([^"]+\.html)"', idx_html)
                if f"/hakusyo/{yr}/html/" in urljoin(idx_url, m)
            )
        )
        def _dl(u):
            fn = u.rsplit("/", 1)[-1]
            return u, fetch_text(session, u, f"env/{yr}/{fn}")

        with ThreadPoolExecutor(max_workers=8) as ex:
            for u, html in ex.map(_dl, links):
                if not html:
                    continue
                paras = extract_html_paragraphs(html)
                if paras:
                    docs.append({
                        "doc_id": f"env_{yr}_{u.rsplit('/', 1)[-1]}",
                        "source": f"env_{yr}",
                        "domain_group": "gov_whitepaper",
                        "license": "PDL1.0",
                        "url": u,
                        "paras": paras,
                    })
    return docs


def collect_mhlw_and_mlit(session: requests.Session) -> List[dict]:
    docs = []
    # 厚生労働省 労働経済白書 (24=R6, 23=R5, 21=R3)
    for yr in ["24", "23", "21"]:
        idx_url = f"https://www.mhlw.go.jp/stf/wp/hakusyo/roudou/{yr}/index.html"
        idx_html = fetch_text(session, idx_url, f"mhlw/rodo_{yr}_idx.html")
        if idx_html:
            links = sorted(
                set(
                    urljoin(idx_url, m)
                    for m in re.findall(r'href="([^"]+\.html)"', idx_html)
                    if f"/hakusyo/roudou/{yr}/" in urljoin(idx_url, m)
                    and "backdata" not in m
                    and not m.endswith("index.html")
                )
            )
            with ThreadPoolExecutor(max_workers=8) as ex:
                for u, html in ex.map(
                    lambda u: (u, fetch_text(session, u, f"mhlw/rodo_{yr}/{u.rsplit('/', 1)[-1]}")),
                    links,
                ):
                    if not html:
                        continue
                    paras = extract_html_paragraphs(html)
                    if paras:
                        docs.append({
                            "doc_id": f"mhlw_rodo_{yr}_{u.rsplit('/', 1)[-1]}",
                            "source": f"mhlw_rodo_{yr}",
                            "domain_group": "gov_whitepaper",
                            "license": "PDL1.0",
                            "url": u,
                            "paras": paras,
                        })

    # 国土交通省 国土交通白書 (R6, R5)
    for yr_dir, yr_lbl in [("mlit/r05/hakusho/r06", "r06"), ("mlit/r04/hakusho/r05", "r05")]:
        mlit_idx = f"https://www.mlit.go.jp/hakusyo/{yr_dir}/index.html"
        mlit_html = fetch_text(session, mlit_idx, f"mlit/{yr_lbl}_idx.html")
        if mlit_html:
            links = sorted(
                set(
                    urljoin(mlit_idx, m)
                    for m in re.findall(r'href="([^"]+\.html)"', mlit_html)
                    if f"/hakusyo/{yr_dir}/html/" in urljoin(mlit_idx, m)
                )
            )
            with ThreadPoolExecutor(max_workers=8) as ex:
                for u, html in ex.map(
                    lambda u: (u, fetch_text(session, u, f"mlit/{yr_lbl}/{u.rsplit('/', 1)[-1]}")),
                    links,
                ):
                    if not html:
                        continue
                    paras = extract_html_paragraphs(html)
                    if paras:
                        docs.append({
                            "doc_id": f"mlit_{yr_lbl}_{u.rsplit('/', 1)[-1]}",
                            "source": f"mlit_{yr_lbl}",
                            "domain_group": "gov_whitepaper",
                            "license": "PDL1.0",
                            "url": u,
                            "paras": paras,
                        })
    return docs


def collect_digital_go(session: requests.Session, max_pages: int = 2000) -> List[dict]:
    urls: List[str] = []
    for p in range(1, 6):
        sm = fetch_text(session, f"https://www.digital.go.jp/sitemap.xml?page={p}", f"digital_go/sm_{p}.xml")
        if not sm:
            break
        locs = re.findall(r"<loc>([^<]+)</loc>", sm)
        if not locs:
            break
        for u in locs:
            if not u.startswith("https://www.digital.go.jp/en/") and any(
                seg in u for seg in ["/policies/", "/resources/", "/news/"]
            ):
                urls.append(u)
    urls = sorted(set(urls))
    rng = random.Random(42)
    rng.shuffle(urls)
    urls = urls[:max_pages]

    docs = []
    def _dl(u):
        h = hashlib.md5(u.encode()).hexdigest()[:16]
        return u, h, fetch_text(session, u, f"digital_go/pages/{h}.html")

    with ThreadPoolExecutor(max_workers=12) as ex:
        for u, h, html in ex.map(_dl, urls):
            if not html:
                continue
            paras = extract_html_paragraphs(html)
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


def _extract_egov_sentences(node) -> str:
    parts: List[str] = []

    def walk(x):
        if isinstance(x, str):
            parts.append(x)
        elif isinstance(x, list):
            for i in x:
                walk(i)
        elif isinstance(x, dict):
            # Ruby タグ中の読み <Rt> を除外
            if x.get("tag") == "Rt":
                return
            for k, v in x.items():
                if k == "children":
                    walk(v)

    walk(node)
    return nr.clean_surface("".join(parts))


def _find_egov_articles(node, out: List[dict]) -> None:
    if isinstance(node, dict):
        if node.get("tag") == "Article":
            out.append(node)
            return
        for c in node.get("children", []) if isinstance(node.get("children"), list) else []:
            _find_egov_articles(c, out)
    elif isinstance(node, list):
        for item in node:
            _find_egov_articles(item, out)


def collect_egov_laws(session: requests.Session, max_laws: int = 200) -> List[dict]:
    law_list: List[dict] = []
    for offset in [1200, 1400, 1600, 1800, 2000]:
        raw = fetch_text(
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
            # 明治・大正などのカタカナ文語法・改正法だけ名称のものを避け現代ひらがな法令を優先
            era = li.get("law_num_era", "")
            if lid and title and era in ("Heisei", "Reiwa") and "一部を改正する" not in title:
                law_list.append({"law_id": lid, "title": title})
    rng = random.Random(42)
    rng.shuffle(law_list)
    law_list = law_list[:max_laws]

    docs = []
    def _dl(item):
        lid = item["law_id"]
        raw = fetch_text(
            session,
            f"https://laws.e-gov.go.jp/api/2/law_data/{lid}",
            f"egov/laws/{lid}.json",
            is_json=True,
        )
        return item, raw

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
            _find_egov_articles(lf, articles)
            paras: List[str] = []
            title = item["title"]
            for art in articles[:40]:
                # 条文単位で ParagraphSentence を連結
                art_title = ""
                sent_parts: List[str] = []
                for ch in art.get("children", []):
                    if not isinstance(ch, dict):
                        continue
                    tg = ch.get("tag")
                    if tg == "ArticleTitle":
                        art_title = _extract_egov_sentences(ch)
                    elif tg == "Paragraph":
                        for pch in ch.get("children", []):
                            if isinstance(pch, dict) and pch.get("tag") == "ParagraphSentence":
                                ps = _extract_egov_sentences(pch)
                                if ps:
                                    sent_parts.append(ps)
                body = "".join(sent_parts).strip()
                # カタカナ文語法や極短条文を除外
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


# --- 2. OSS 日本語ドキュメントコレクタ (MIT / Apache-2.0) ---

OSS_SOURCES = [
    {
        "source": "fastapi_ja",
        "repo": "fastapi/fastapi",
        "branch": "master",
        "license": "MIT",
        "filter_fn": lambda p: p.startswith("docs/ja/docs/") and p.endswith(".md"),
    },
    {
        "source": "vite_ja",
        "repo": "vitejs/docs-ja",
        "branch": "main",
        "license": "MIT",
        "filter_fn": lambda p: p.endswith(".md") and not p.startswith(".github/") and p not in ("CONTRIBUTING.md", "README.md"),
    },
    {
        "source": "hf_transformers_ja",
        "repo": "huggingface/transformers",
        "branch": "main",
        "license": "Apache-2.0",
        "filter_fn": lambda p: p.startswith("docs/source/ja/") and p.endswith((".md", ".mdx")),
    },
    {
        "source": "hf_course_ja",
        "repo": "huggingface/course",
        "branch": "main",
        "license": "Apache-2.0",
        "filter_fn": lambda p: p.startswith("chapters/ja/") and p.endswith(".mdx"),
    },
    {
        "source": "tf_docs_ja",
        "repo": "tensorflow/docs-l10n",
        "branch": "master",
        "license": "Apache-2.0",
        "filter_fn": lambda p: p.startswith("site/ja/") and p.endswith(".md"),
    },
    {
        "source": "ms_ml_ja",
        "repo": "microsoft/ML-For-Beginners",
        "branch": "main",
        "license": "MIT",
        "filter_fn": lambda p: (".ja.md" in p or "translations/ja/" in p) and p.endswith(".md"),
    },
    {
        "source": "ms_ai_ja",
        "repo": "microsoft/AI-For-Beginners",
        "branch": "main",
        "license": "MIT",
        "filter_fn": lambda p: (".ja.md" in p or "translations/ja/" in p) and p.endswith(".md"),
    },
    {
        "source": "ms_web_ja",
        "repo": "microsoft/Web-Dev-For-Beginners",
        "branch": "main",
        "license": "MIT",
        "filter_fn": lambda p: (".ja.md" in p or "translations/ja/" in p) and p.endswith(".md"),
    },
    {
        "source": "ms_genai_ja",
        "repo": "microsoft/generative-ai-for-beginners",
        "branch": "main",
        "license": "MIT",
        "filter_fn": lambda p: (".ja.md" in p or "translations/ja/" in p) and p.endswith(".md"),
    },
    {
        "source": "ms_ds_ja",
        "repo": "microsoft/Data-Science-For-Beginners",
        "branch": "main",
        "license": "MIT",
        "filter_fn": lambda p: (".ja.md" in p or "translations/ja/" in p) and p.endswith(".md"),
    },
    {
        "source": "ms_iot_ja",
        "repo": "microsoft/IoT-For-Beginners",
        "branch": "main",
        "license": "MIT",
        "filter_fn": lambda p: (".ja.md" in p or "translations/ja/" in p) and p.endswith(".md"),
    },
]


def collect_oss_docs(session: requests.Session) -> List[dict]:
    docs = []
    for cfg in OSS_SOURCES:
        src = cfg["source"]
        repo = cfg["repo"]
        br = cfg["branch"]
        lic = cfg["license"]
        tree_raw = fetch_text(
            session,
            f"https://api.github.com/repos/{repo}/git/trees/{br}?recursive=1",
            f"gh/{src}_tree.json",
            is_json=True,
        )
        if not tree_raw:
            print(f"[WARN] failed tree for {repo}")
            continue
        tree = json.loads(tree_raw).get("tree", [])
        paths = [t["path"] for t in tree if t.get("type") == "blob" and cfg["filter_fn"](t["path"])]
        print(f"  GH {src}: {len(paths)} files matched")

        def _dl(p):
            raw_url = f"https://raw.githubusercontent.com/{repo}/{br}/{p}"
            h = hashlib.md5(p.encode()).hexdigest()[:16]
            return p, raw_url, fetch_text(session, raw_url, f"gh/{src}/{h}.md")

        with ThreadPoolExecutor(max_workers=16) as ex:
            for p, raw_url, md in ex.map(_dl, paths):
                if not md:
                    continue
                paras = clean_markdown_to_paragraphs(md)
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


# --- 3. クラスタリング & 層化プール分割 ---

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-chunks", type=int, default=18000)
    ap.add_argument("--max-law-ratio", type=float, default=0.15)
    args = ap.parse_args()

    t0 = time.time()
    rng = random.Random(args.seed)
    session = requests.Session()

    print("[1/4] Collecting PDL1.0 / Art13 public documents...")
    gov_docs: List[dict] = []
    for name, fn in [
        ("soumu", collect_soumu),
        ("env", collect_env),
        ("mhlw_mlit", collect_mhlw_and_mlit),
        ("digital_go", collect_digital_go),
        ("egov_laws", collect_egov_laws),
    ]:
        part = fn(session)
        print(f"  {name}: {len(part)} docs")
        gov_docs.extend(part)

    print("[2/4] Collecting MIT / Apache-2.0 OSS documents...")
    oss_docs = collect_oss_docs(session)
    all_docs = gov_docs + oss_docs
    print(f"Total docs extracted: {len(all_docs)} (gov={len(gov_docs)}, oss={len(oss_docs)})")

    print("[3/4] Chunking & document-level LSH clustering...")
    doc_records: List[dict] = []
    seen_chunk_norms: Set[str] = set()
    for doc in all_docs:
        chks = chunk_paragraphs(doc["paras"], rng)
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
    uf = UnionFind(len(doc_records))
    for u, v, _ in dup_pairs:
        uf.union(u, v)

    clusters: Dict[int, List[dict]] = defaultdict(list)
    for idx, d in enumerate(doc_records):
        cid = uf.find(idx)
        d["cluster_id"] = f"cl_{cid:06d}"
        clusters[cid].append(d)

    print(f"Unique docs={len(doc_records)}, clusters={len(clusters)}, near_dup_doc_pairs={len(dup_pairs)}")

    # ドメイン・ソースバランス＆法令 15% 上限管理
    # まず全クラスタ単位のチャンク数を集計し、目標数に向けてバランスサンプリング
    raw_by_domain = Counter()
    for d in doc_records:
        raw_by_domain[d["domain_group"]] += len(d["chunks"])
    print("Raw chunks available by domain_group:", dict(raw_by_domain))

    # チャンク数キャップの調整（law <= 15%, gov ~ 60%, oss ~ 40%）
    target = args.target_chunks
    max_law_chunks = int(target * args.max_law_ratio)
    max_gov_nonlaw = int(target * 0.45)
    max_oss_chunks = int(target * 0.40)

    # クラスタをシャッフルしてドメインバケットごとに採用上限まで確保
    cluster_items = list(clusters.items())
    rng.shuffle(cluster_items)

    # ソース均一性を保つため、ソースごとにクラスタをラウンドロビンまたは層化順にする
    selected_clusters: List[Tuple[int, List[dict]]] = []
    domain_counts = Counter()

    # 単一巨大ドキュメントが偏らないよう、1クラスタからの抽出上限を 80 チャンクに設定
    for cid, dlist in cluster_items:
        dom = dlist[0]["domain_group"]
        tot_c = sum(min(len(d["chunks"]), 60) for d in dlist)
        if dom == "gov_law" and domain_counts["gov_law"] >= max_law_chunks:
            continue
        if dom in ("gov_whitepaper", "gov_policy") and (
            domain_counts["gov_whitepaper"] + domain_counts["gov_policy"] >= max_gov_nonlaw + 1500
        ):
            # 白書＋政策が溢れすぎないよう調整（ただし全体が目標に満たない時は後で緩める）
            pass
        selected_clusters.append((cid, dlist))
        domain_counts[dom] += tot_c

    # 実際は入手全量を保持しつつ、プール分割後に目標チャンク数へ層化サンプリングすると柔軟
    # 先に全クラスタを 4 プールへ層化割当 (human_eval: ~4%, dev: ~9%, sealed_test: ~9%, train: ~78%)
    pool_names = ["human_eval", "dev", "sealed_test", "train"]
    pool_ratios = {"human_eval": 0.045, "dev": 0.095, "sealed_test": 0.095, "train": 0.765}

    # source別にクラスタをグループ化して各ソースが全4プールに入るように割当
    by_src_clusters: Dict[str, List[List[dict]]] = defaultdict(list)
    for cid, dlist in clusters.items():
        by_src_clusters[dlist[0]["source"]].append(dlist)

    pool_chunks: Dict[str, List[dict]] = {p: [] for p in pool_names}
    for src, clist in sorted(by_src_clusters.items()):
        rng.shuffle(clist)
        total_src_chunks = sum(len(d["chunks"]) for dl in clist for d in dl)
        quotas = {p: max(1, int(total_src_chunks * pool_ratios[p])) for p in pool_names}
        assigned = Counter()
        # 順番に最も充足率の低いプールへ割り当て（クラスタ単位で分割）
        for dl in clist:
            # まだ少ないプール優先（まず human_eval/dev/sealed_test に最低件数を配る）
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

    # 各プールを目標チャンク数・ドメイン比率・長さ比率に整える
    # 目標チャンク数: human_eval=600, dev=1550, sealed_test=1550, train=14300 (計 18,000)
    pool_targets = {
        "human_eval": min(len(pool_chunks["human_eval"]), 650),
        "dev": min(len(pool_chunks["dev"]), 1600),
        "sealed_test": min(len(pool_chunks["sealed_test"]), 1600),
        "train": min(len(pool_chunks["train"]), 14500),
    }

    def sample_balanced(items: List[dict], cap: int) -> List[dict]:
        # 法令上限 15%、OSS 目標 40%、公的非法令目標 45%
        law = [x for x in items if x["domain_group"] == "gov_law"]
        gov = [x for x in items if x["domain_group"] in ("gov_whitepaper", "gov_policy")]
        oss = [x for x in items if x["domain_group"] == "oss_docs"]
        rng.shuffle(law)
        rng.shuffle(gov)
        rng.shuffle(oss)
        n_law = min(len(law), int(cap * args.max_law_ratio))
        n_oss = min(len(oss), int(cap * 0.40))
        n_gov = min(len(gov), cap - n_law - n_oss)
        picked = law[:n_law] + gov[:n_gov] + oss[:n_oss]
        if len(picked) < cap:
            rem = [x for x in gov[n_gov:] + oss[n_oss:]]
            rng.shuffle(rem)
            picked.extend(rem[: cap - len(picked)])
        rng.shuffle(picked)
        return picked

    final_chunks: List[dict] = []
    pool_stats = {}
    for p in pool_names:
        picked = sample_balanced(pool_chunks[p], pool_targets[p])
        final_chunks.extend(picked)
        pool_stats[p] = {
            "count": len(picked),
            "available": len(pool_chunks[p]),
            "by_domain": dict(Counter(x["domain_group"] for x in picked)),
            "by_source": dict(Counter(x["source"] for x in picked)),
            "by_len_bucket": dict(Counter(x["len_bucket"] for x in picked)),
        }

    # chunk_id の振り直し・JSONL 保存
    chunks_path = SEEDS_DIR / "chunks_all.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(final_chunks):
            row["seed_idx"] = i
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "seed": args.seed,
        "total_docs": len(doc_records),
        "total_clusters": len(clusters),
        "total_chunks_available": sum(len(v) for v in pool_chunks.values()),
        "total_chunks_selected": len(final_chunks),
        "chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "pools": pool_stats,
        "sources_license_audit": {
            "soumu_whitepapers": {"license": "PDL1.0", "terms_url": "https://www.soumu.go.jp/menu_kyotsuu/policy/tyosaku.html"},
            "env_whitepapers": {"license": "PDL1.0", "terms_url": "https://www.env.go.jp/mail.html"},
            "mhlw_whitepapers": {"license": "PDL1.0", "terms_url": "https://www.mhlw.go.jp/chosakuken/index.html"},
            "mlit_whitepapers": {"license": "PDL1.0", "terms_url": "https://www.mlit.go.jp/link.html"},
            "digital_go": {"license": "PDL1.0", "terms_url": "https://www.digital.go.jp/copyright-policy"},
            "egov_laws": {"license": "Art13_Copyright_Act / PDL1.0", "terms_url": "https://laws.e-gov.go.jp/"},
            "fastapi_ja": {"license": "MIT", "repo": "fastapi/fastapi"},
            "vite_ja": {"license": "MIT", "repo": "vitejs/docs-ja"},
            "hf_transformers_ja": {"license": "Apache-2.0", "repo": "huggingface/transformers"},
            "hf_course_ja": {"license": "Apache-2.0", "repo": "huggingface/course"},
            "tf_docs_ja": {"license": "Apache-2.0", "repo": "tensorflow/docs-l10n"},
            "ms_beginners_ja": {"license": "MIT", "repos": ["microsoft/ML-For-Beginners", "microsoft/AI-For-Beginners", "microsoft/Web-Dev-For-Beginners", "microsoft/generative-ai-for-beginners", "microsoft/Data-Science-For-Beginners", "microsoft/IoT-For-Beginners"]},
        },
    }
    manifest_path = SEEDS_DIR / "seed-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"[4/4] Done in {elapsed:.1f}s -> {chunks_path} ({len(final_chunks)} chunks)")
    for p, st in pool_stats.items():
        print(f"  {p}: {st['count']} (avail {st['available']}) | dom={st['by_domain']} | len={st['by_len_bucket']}")


if __name__ == "__main__":
    main()
