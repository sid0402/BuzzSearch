from __future__ import annotations
import json, re, html, unicodedata, string
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

import markdown_it
import nltk

# ─────────────────────────── paths ────────────────────────────
ROOT                    = Path(__file__).resolve().parents[1]
REDDIT_TRIPLES_FILE     = ROOT / "reddit_posts" / "triples_reddit_self_ruthless_filtered.jsonl"
FULL_REDDIT_FILE        = ROOT.parent / "scraper" / "reddit" / "posts_gatech_new.jsonl"
DOCS_OUT_PATH           = ROOT / "all_document_summaries" / "documents_needed" / "reddit_documents_filtered.json"

# ─────────────────── constants / helpers ──────────────────────
URL_RE     = re.compile(r"https?://\S+")
PUNCT_TR   = str.maketrans("", "", string.punctuation)
STOP_SET   = {"edit", "tl;dr", "thanks", "link", "jpg", "png", "gif", "http", "https"}
TAG_TOKENS = re.compile(r"\b(?:p|br|hr|ul|ol|li|strong|em|a|peg)\b", re.I)

SPECIAL_SUBS = {                 # run *before* punctuation strip
    r"\br&d\b":   "research and development",
    r"\be\.g\.\b": "for example",
    r"\bi\.e\.\b": "that is",
}

MAX_TOK   = 400
MAX_SENT  = 6

md = markdown_it.MarkdownIt()     # fast no‑dependency Markdown → HTML

# ───────────────────────── IO helpers ─────────────────────────
def load_jsonl(path: Path) -> list:
    with path.open() as f:
        return [json.loads(line) for line in f]

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, ensure_ascii=False)

# ───────────────────── text normalisation ─────────────────────
def _plain(text: str) -> str:
    """Markdown → plain lower‑case text with URLs & punctuation removed."""
    text = md.render(text)
    for pat, repl in SPECIAL_SUBS.items():
        text = re.sub(pat, repl, text, flags=re.I)

    text = TAG_TOKENS.sub(" ", text)      # strip tiny tag artefacts
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" ", text)
    text = text.translate(PUNCT_TR)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def _dedupe(sentences: List[str]) -> List[str]:
    out, prev = [], None
    for s in sentences:
        if s != prev:
            out.append(s)
        prev = s
    return out

def _is_boiler(sentence: str) -> bool:
    toks = sentence.split()
    return toks and sum(w in STOP_SET for w in toks) / len(toks) > 0.5

def to_passages(raw_md: str,
                max_tokens: int = MAX_TOK,
                max_sent: int = MAX_SENT) -> List[str]:
    norm = _plain(raw_md)
    sents = [s for s in nltk.sent_tokenize(norm) if not _is_boiler(s)]
    sents = _dedupe(sents)

    chunk, words, out = [], 0, []
    for s in sents:
        tok_len = len(s.split())
        if (words + tok_len > max_tokens or len(chunk) >= max_sent) and chunk:
            out.append(" ".join(chunk))
            chunk, words = [], 0
        chunk.append(s)
        words += tok_len
    if chunk:
        out.append(" ".join(chunk))
    return out

# ───────────────────────── main pipeline ──────────────────────
def build_documents() -> list:
    print("Loading data …")
    triples         = load_jsonl(REDDIT_TRIPLES_FILE)
    all_posts       = {p["id"]: p for p in load_jsonl(FULL_REDDIT_FILE)}

    docs = []
    for i, tpl in enumerate(triples, 1):
        post = all_posts.get(tpl["id"])
        if not post:                       # should rarely happen
            continue
        if i % 500 == 0:
            print(f"processed {i:,}/{len(triples):,}")

        passages = to_passages(post["selftext"])
        if not passages:
            continue

        docs.append({
            **post,
            "passages": passages
        })
    return docs

nltk.download("punkt", quiet=True)
documents = build_documents()
save_json(DOCS_OUT_PATH, documents)