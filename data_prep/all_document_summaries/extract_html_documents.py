import re
import json
# import textwrap # Not used in this modification, but was in original context
import html
from pathlib import Path
from urllib.parse import urlparse
from collections import defaultdict
from datetime import datetime # Ensure datetime is imported
import os
import sys
from bs4 import BeautifulSoup, Comment, XMLParsedAsHTMLWarning
import warnings
import unicodedata, nltk, html
import re, html, unicodedata, string
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]  # project root
HTML_CLEANED_TRIPLES = ROOT / "gatech_html" / "triples_gatech_cleaned.jsonl"
FULL_HTML_DATA_ROOT = ROOT.parent / "scraper" / "gatech_crawl" / "html"
DOCUMENTS_FILTERED = 'documents_needed/html_documents_filtered.json'

#nltk.download('punkt_tab')

def find_number_files_in_directory(directory: Path):
    return len([f for f in directory.glob('**/*') if f.is_file()])

def load_gatech_triples(path: Path):
    gatech_triples = []
    with open(path, 'r') as f:
        for line in f:
            gatech_triples.append(json.loads(line))
    return gatech_triples

def find_documents_needed(gatech_triples: list):
    documents_needed = []
    for triple in gatech_triples:
        sha = triple['sha']
        f_name = f"{sha}.html"
        f_path = FULL_HTML_DATA_ROOT / f_name
        if f_path.exists():
            html_text = f_path.read_text()
            doc = {
                'sha': sha,
                'url': triple['url'],
                'html': f_path.read_text()
            }
            documents_needed.append(doc)
            
    return documents_needed

DEFAULT_MAX_TOKENS = 400                    # ≈ 512‑token BPE safety

# “hard” boilerplate words – curated + top‑50 from earlier DF counts
_HARD_BOILER = [
    "navigation", "give", "visit", "search", "menu", "skip", "main",
    "accessibility", "legal", "privacy", "copyright",
    "footer", "header"
]

# HTML selectors that almost always carry chrome
_EXTRA_STRIP_SELECTORS = [
    "[role='banner']",
    "[role='navigation']",
    "[role='search']",
    ".breadcrumb",
    ".breadcrumbs",
    ".cookie-banner",
    ".cookie-consent",
    ".offcanvas",
    ".modal",
]

# ---------------------------------------------------------------------------
# 2. LOW‑LEVEL HELPERS --------------------------------------------------------

_BOILER_RE = re.compile(
    r"(THEME OUTPUT|DEBUG|HOOK:|BEGIN|END|SUGGESTIONS:|FILE NAME)",
    re.I,
)

_STRIP_PUNCT = str.maketrans("", "", string.punctuation)


def _strip_markup(html_raw: str) -> BeautifulSoup:
    """
    Remove obviously irrelevant DOM parts and return a cleaned BeautifulSoup.
    """
    soup = BeautifulSoup(html_raw, "lxml")

    # a) user‑supplied selectors
    for sel in [
        "header", "footer", "nav", "aside",
        ".gt-menu", ".gt-footer", ".site-header", ".site-footer",
        * _EXTRA_STRIP_SELECTORS
    ]:
        for node in soup.select(sel):
            node.decompose()

    # b) invisible / scriptish elements
    for tag in soup(["script", "style", "noscript", "meta", "link"]):
        tag.decompose()

    # c) comments
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    # d) <figure><img …/></figure>  → delete if contains no <p>/<span> text
    for fig in soup.find_all("figure"):
        if not fig.find(text=True, recursive=False):   # no visible children
            fig.decompose()

    return soup


def _normalise(text: str) -> str:
    """
    HTML‑unescape, unicode‑normalise, strip punctuation & collapse whitespace.
    """
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _BOILER_RE.sub(" ", text)
    text = re.sub(r"<[^>]+>", " ", text)        # stray tags
    text = text.translate(_STRIP_PUNCT)
    text = re.sub(r"\s+", " ", text).strip().lower()
    for word in _HARD_BOILER:
        text = text.replace(word, "")
    text = text.replace("  ", " ")
    re.sub(r"\s{2,}", "", text)
    return text


def _dedupe_sentences(sent_list: List[str]) -> List[str]:
    """
    Drop consecutive duplicate sentences.
    """
    out = []
    prev = None
    for s in sent_list:
        if s != prev:
            out.append(s)
        prev = s
    return out


def _is_boiler(sentence: str) -> bool:
    """
    True if >50 % of tokens are in the HARD_BOILER stop‑set.
    """
    toks = sentence.split()
    if not toks:
        return True
    bad = sum(1 for t in toks if t in _HARD_BOILER)
    return bad / len(toks) > 0.5


def _sentence_chunks(text: str, max_tokens: int = DEFAULT_MAX_TOKENS):
    """
    Yield ≤max_tokens word chunks, skipping boilerplate‑heavy passages.
    """
    sents = _dedupe_sentences(nltk.sent_tokenize(text))
    chunk, size = [], 0
    for s in sents:
        if _is_boiler(s):
            continue
        tok_len = len(s.split())
        if size + tok_len > max_tokens and chunk:
            yield " ".join(chunk)
            chunk, size = [], 0
        chunk.append(s)
        size += tok_len
    if chunk:
        yield " ".join(chunk)

# ---------------------------------------------------------------------------
# 3. PUBLIC API ---------------------------------------------------------------

def clean_html(html_raw: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> Dict:
    """
    Convert raw HTML to {title, full_text, passages[]} after aggressive cleaning.
    """
    soup = _strip_markup(html_raw)

    # --- title --------------------------------------------------------------
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string
    else:                                      # fall back to first <h1>
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)
    title = _normalise(title)

    # --- body ---------------------------------------------------------------
    body_raw = soup.get_text(" ", strip=True)
    body = _normalise(body_raw)

    passages = list(_sentence_chunks(body, max_tokens))

    return {
        "title": title,
        "full_text": body,
        "passages": passages,
    }

def create_html_documents(documents_needed: list):
    for i, doc in enumerate(documents_needed):
        if i % 100 == 0:
            print(f"Processed {i} documents")
        doc['html_cleaned'] = clean_html(doc['html'])

    with open(DOCUMENTS_FILTERED, 'w') as f:
        for doc in documents_needed:
            f.write(json.dumps(doc) + '\n')

    return documents_needed

gatech_triples = load_gatech_triples(HTML_CLEANED_TRIPLES)
documents_needed = find_documents_needed(gatech_triples)
documents_needed = create_html_documents(documents_needed)
