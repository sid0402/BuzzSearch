import json, re, pathlib
from urllib.parse import urlparse
from collections import defaultdict

RAW   = pathlib.Path("triples_gatech.jsonl")
CLEAN = pathlib.Path("triples_gatech_cleaned.jsonl")

PHONE_RE       = re.compile(r"\(?\b[2-9]\d{2}[)\s.-]?\d{3}[-.\s]?\d{4}\b")
URL_TEXT_RE    = re.compile(r"https?://", re.I)
PUNCT_ANCHORRE = re.compile(r"^[\W\u203a\u00bb]+$")
DOMAIN_ALLOW   = (".gatech.edu", ".gatech.com")

NUM_ONLY_RE   = re.compile(r"^\d+$")
PUNCT_ONLY_RE = re.compile(r"^[\W\u203a\u00bb]+$")

def keep_anchor(anchor: str) -> bool:
    a = anchor.strip()
    if not a:
        return False
    if PHONE_RE.search(a):
        return False
    if URL_TEXT_RE.search(a):
        return False
    if NUM_ONLY_RE.fullmatch(a):
        return False
    if PUNCT_ONLY_RE.fullmatch(a):
        return False
    if len(a.split()) > 8:
        return False
    return True

def keep_url(u: str) -> bool:
    parsed = urlparse(u)
    # keep only gatech hosts
    if not parsed.hostname or not parsed.hostname.endswith(".gatech.edu"):
        return False
    # fragment that starts with "bottom_tabs_anchor"  → drop
    if parsed.fragment.startswith("bottom_tabs_anchor"):
        return False
    exts = ('pdf',"png","jpg","jpeg","gif","svg","mp4","mp3","avi","zip","tar","exe")
    if any(ext in parsed.path.lower() for ext in exts):
        return False
    return True

def normalize(text: str) -> str:
    """Lower‑case and collapse whitespace for fuzzy de‑dupe."""
    return re.sub(r"\s+", " ", text.strip().lower())

# ------------------------------------------------------------
def load_raw(path=RAW):
    with path.open() as f:
        for line in f:
            yield json.loads(line)

def clean_triples():
    seen_exact = set()                   # (anchor,context,url)  exact
    seen_anchor_ctx = defaultdict(int)   # how many kept for each URL
    
    # Load all triples first
    all_triples = list(load_raw())
    
    # Filter out vieques URLs
    all_triples = filter_vieques_urls(all_triples)

    with CLEAN.open("w") as out:
        kept, dropped = 0, 0

        for trip in all_triples:
            a, c, u = trip["anchor"], trip["context"], trip["url"]

            # 1. basic filters
            if not (keep_anchor(a) and keep_url(u)):
                dropped += 1
                continue

            # 2. exact duplicate filter
            sig = (normalize(a), normalize(c), u)
            if sig in seen_exact:
                dropped += 1
                continue
            seen_exact.add(sig)

            # 3. limit very similar anchors per URL  (optional)
            if seen_anchor_ctx[u] >= 5:          # keep at most 5 per target
                dropped += 1
                continue
            seen_anchor_ctx[u] += 1

            out.write(json.dumps(trip) + "\n")
            kept += 1

    print(f"✓  kept {kept:,} triples   (dropped {dropped:,}) → {CLEAN}")

def filter_vieques_urls(triples):
    """
    Filter out any triples that have 'vieques' in the URL.
    
    Args:
        triples: List of triple dictionaries with 'url' keys
        
    Returns:
        List of filtered triples
    """
    filtered_triples = []
    dropped = 0
    
    for triple in triples:
        url = triple.get('url', '')
        # Case-insensitive check for 'vieques' in the URL
        if 'vieques' in url.lower():
            dropped += 1
            continue
        filtered_triples.append(triple)
    
    print(f"Filtered out {dropped} triples containing 'vieques' in URL")
    return filtered_triples

# ------------------------------------------------------------
if __name__ == "__main__":
    if not RAW.exists():
        raise SystemExit(f"{RAW} not found")
    clean_triples()