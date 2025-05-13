import re
import json
# import textwrap # Not used in this modification, but was in original context
import html
from pathlib import Path
from urllib.parse import urlparse
from collections import defaultdict
from datetime import datetime # Ensure datetime is imported

# ---------- paths -------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # project root
RAW_REDDIT_JL = ROOT / "scraper" / "reddit" / "posts_gatech_new.jsonl"
# Output files will be named with "_ruthless_filtered" suffix
SELF_OUT_FILTERED = Path(__file__).parent / "triples_reddit_self_ruthless_filtered.jsonl"
LINK_OUT = Path(__file__).parent / "triples_reddit_links_original.jsonl" # Link triples are not changed by this request

# ---------- link / anchor filters (from original context, for LINK triples) ----
IMG_EXT = {"pdf", "png", "jpg", "jpeg", "gif", "svg", "webp", "mp4", "mov"}
BAD_HOST = lambda h: not h.endswith(".gatech.edu") if h else True # handle None hostname
UTM_RE = re.compile(r"&?(utm_[^=&]+|mc_[^=&]+|fbclid)=[^&#]+")

ANCHOR_URL_RE = re.compile(r"^https?://", re.I)
ANCHOR_DIGIT_RE = re.compile(r"^\d+$")
ANCHOR_PUNCT_RE = re.compile(r"^[\W]+$")

# ---------- Ruthless filter constants for SELF triples ------------
# These values are based on the latest script provided by the user
RUTHLESS_MIN_SCORE = 2
RUTHLESS_ANCHOR_MIN_WORDS = 3
RUTHLESS_CONTEXT_MIN_WORDS = 51 # Meaning context must have at least 151 words (i.e., >150)
RUTHLESS_CONTEXT_EXCERPT_TOKENS = 10000 # Grab 200 tokens initially for context
RUTHLESS_REDUNDANCY_THRESHOLD = 1000 # Jaccard similarity threshold

MIN_POST_YEAR = 2000 # New constant for date filtering

COMMON_GREETINGS_PATTERNS = [
    r"hey all", r"hey everyone", r"hey guys", r"hi all", r"hi everyone", r"hi guys", r"hi there",
    r"hello all", r"hello everyone", r"hello there", r"hello", r"hi", r"hey", # Shorter greetings too
    r"good morning", r"good afternoon", r"good evening",
    r"hope you're doing well", r"hope you are doing well", r"hope you're well", r"hope you are well",
    r"hope this is the right place", r"first time poster", r"long time lurker",
    r"as the title says", r"title says it all", r"basically title", r"title pretty much says it all",
    r"question in title", r"see title" # "see title" can also be reference only
]
GREETING_STRIP_RE = re.compile(
    r"^\s*(" + "|".join(COMMON_GREETINGS_PATTERNS) + r")[\s.,!?;:-]*",
    re.IGNORECASE
)

REFERENCE_ONLY_TEXTS = [
    "title", "see title", "title says it all", "/thread", "[deleted]", "[removed]"
]
REFERENCE_ONLY_RE = re.compile(
    r"^\s*(" + "|".join(re.escape(t) for t in REFERENCE_ONLY_TEXTS) + r")\s*$",
    re.IGNORECASE
)

# --- Helper functions for LINK triples (from original context, slightly hardened) ---
def is_gatech_textual(url: str) -> bool:
    if not isinstance(url, str): return False
    try:
        p = urlparse(url)
        if BAD_HOST(p.hostname or ""): # Ensure hostname is not None before .endswith
            return False
        path_parts = p.path.rsplit(".", 1)
        if len(path_parts) > 1:
            ext = path_parts[-1].lower()
            if ext in IMG_EXT or ext == "pdf":
                return False
        return True
    except Exception:
        return False

def clean_url(url: str) -> str:
    if not isinstance(url, str): return ""
    url = html.unescape(url)
    url = UTM_RE.sub("", url)
    return url.rstrip("?#&")

def good_anchor(a: str) -> bool:
    if not isinstance(a, str): return False
    a = a.strip()
    if not a: return False
    if ANCHOR_URL_RE.match(a): return False
    if ANCHOR_DIGIT_RE.match(a) or ANCHOR_PUNCT_RE.match(a): return False
    if len(a.split()) > 8: return False
    return True

# ---------- Text processing helper (MODIFIED as per previous iteration) ----------
LINK_MD_RE = re.compile(r"\[([^\]]{1,80})\]\((https?://[^\s)<>\"']+)\)")
GENERAL_LINK_LIKE_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")

def md_to_text(md: str) -> str:
    if not isinstance(md, str): return ""
    text = html.unescape(md)
    text = re.sub(r"(?m)^\s*>+ ?", " ", text)
    text = re.sub(r"(?m)^\s*#+\s*", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n+", " ", text)
    text = re.sub(LINK_MD_RE, r"\1", text)
    text = re.sub(GENERAL_LINK_LIKE_RE, r"\1", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"\*\*|__|\*|_", " ", text)
    text = re.sub(r"https?://[^\s<>\"']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def excerpt(text: str, tokens=40) -> str:
    if not isinstance(text, str): return ""
    return " ".join(text.split()[:tokens])

# ---------- New helper functions for RUTHLESS SELF triple filtering ----------
def strip_leading_greetings(text: str) -> str:
    if not isinstance(text, str): return ""
    return GREETING_STRIP_RE.sub("", text, 1).strip()

def is_reference_only(text: str) -> bool:
    if not isinstance(text, str): return False
    return bool(REFERENCE_ONLY_RE.fullmatch(text))

def jaccard_similarity(set1: set, set2: set) -> float:
    if not isinstance(set1, set) or not isinstance(set2, set): return 0.0
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0: return 1.0
    return intersection / union

# ---------- main --------------------------------------------------
def build_triples_ruthlessly():
    SELF_OUT_FILTERED.parent.mkdir(parents=True, exist_ok=True)
    LINK_OUT.parent.mkdir(parents=True, exist_ok=True)

    if SELF_OUT_FILTERED.exists(): SELF_OUT_FILTERED.unlink()
    if LINK_OUT.exists(): LINK_OUT.unlink()

    self_f = SELF_OUT_FILTERED.open("w", encoding='utf-8') # Changed to 'w' to overwrite
    link_f = LINK_OUT.open("w", encoding='utf-8') # Changed to 'w' to overwrite
    
    per_url_cap_links = defaultdict(int)
    kept_self_triples_count = 0
    processed_posts_count = 0
    total_link_triples_count = 0

    print(f"Starting ruthless self-triple generation. Output to: {SELF_OUT_FILTERED}")
    print(f"Link triple generation (original logic). Output to: {LINK_OUT}")

    with RAW_REDDIT_JL.open('r', encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            processed_posts_count = i + 1
            if processed_posts_count % 20000 == 0 and processed_posts_count > 0:
                print(f"Processed {processed_posts_count:,} posts. Kept {kept_self_triples_count:,} self-triples.")

            try:
                post = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line {processed_posts_count}")
                continue

            # --- DATE FILTER (Applied to all posts) ---
            created_utc_timestamp = post.get("created_utc")
            if created_utc_timestamp is None:
                # print(f"Warning: Skipping post ID {post.get('id', 'N/A')} due to missing 'created_utc' at line {processed_posts_count}")
                continue
            try:
                post_date = datetime.utcfromtimestamp(float(created_utc_timestamp))
                if post_date.year < MIN_POST_YEAR:
                    continue
            except (ValueError, TypeError) as e:
                # print(f"Warning: Skipping post ID {post.get('id', 'N/A')} due to invalid timestamp '{created_utc_timestamp}' (Error: {e}) at line {processed_posts_count}")
                continue
            # --- END DATE FILTER ---

            raw_title = post.get("title", "").strip()
            raw_body = post.get("selftext", "").strip()
            score = post.get("score", 0)
            permalink = post.get("permalink", "")

            if permalink.startswith("/r/"):
                self_permalink = "https://www.reddit.com" + permalink
            elif permalink.startswith("r/"):
                 self_permalink = "https://www.reddit.com/" + permalink
            elif not permalink.startswith("http"):
                 self_permalink = "" # Mark as invalid
            else:
                self_permalink = permalink
            
            if not self_permalink: continue

            # --- SELF triple generation with RUTHLESS filtering ---
            if score < RUTHLESS_MIN_SCORE:
                continue

            title_words = raw_title.split()
            if len(title_words) < RUTHLESS_ANCHOR_MIN_WORDS:
                continue
            
            if not raw_body:
                continue

            body_for_context_md_cleaned = md_to_text(raw_body)
            body_after_greeting_strip = strip_leading_greetings(body_for_context_md_cleaned)

            if not body_after_greeting_strip or is_reference_only(body_after_greeting_strip):
                continue

            context_candidate = excerpt(body_after_greeting_strip, tokens=RUTHLESS_CONTEXT_EXCERPT_TOKENS)
            context_candidate_words = context_candidate.split()
            
            if len(context_candidate_words) < RUTHLESS_CONTEXT_MIN_WORDS:
                continue
            
            anchor_tokens_for_jaccard = set(word.lower() for word in title_words if word)
            context_tokens_for_jaccard = set(word.lower() for word in context_candidate_words if word)
            
            # No need for special handling of empty sets here, jaccard_similarity handles it.
            if jaccard_similarity(anchor_tokens_for_jaccard, context_tokens_for_jaccard) > RUTHLESS_REDUNDANCY_THRESHOLD:
                continue
            
            self_f.write(json.dumps({
                "id":      post.get("id", ""),
                "anchor":  raw_title,
                "context": context_candidate,
                "url":     self_permalink
            }) + "\n")
            kept_self_triples_count += 1

            # --- LINK triples logic (original, from user's context) ---
            # This logic will only run for posts that passed the date filter.
            if raw_body:
                original_selftext_for_links = post.get("selftext", "") # Use original for regex matching
                for anchor_text, raw_url_from_md in LINK_MD_RE.findall(original_selftext_for_links):
                    cleaned_link_url = clean_url(raw_url_from_md)
                    if not is_gatech_textual(cleaned_link_url):
                        continue
                    
                    stripped_anchor = anchor_text.strip()
                    if not good_anchor(stripped_anchor):
                        continue
                    
                    if per_url_cap_links[cleaned_link_url] >= 5:
                        continue
                    per_url_cap_links[cleaned_link_url] += 1

                    full_body_plain_for_links = md_to_text(raw_body)
                    link_context_window_text = ""
                    final_link_context = ""
                    try:
                        search_body_lower = full_body_plain_for_links.lower()
                        search_anchor_lower = stripped_anchor.lower()
                        if not search_anchor_lower: continue

                        idx = search_body_lower.index(search_anchor_lower)
                        
                        anchor_original_len = len(stripped_anchor)
                        start_char_idx = max(0, idx - 200)
                        end_char_idx = min(len(full_body_plain_for_links), idx + anchor_original_len + 200)
                        link_context_window_text = full_body_plain_for_links[start_char_idx : end_char_idx]
                        final_link_context = excerpt(link_context_window_text, tokens=80)

                    except ValueError:
                        pass 
                    
                    if final_link_context:
                        link_f.write(json.dumps({
                            "id":      post.get("id", ""),
                            "anchor":  stripped_anchor,
                            "context": final_link_context,
                            "url":     cleaned_link_url
                        }) + "\n")
                        total_link_triples_count += 1

    self_f.close()
    link_f.close()
    print(f"\nProcessing finished for {processed_posts_count:,} posts.")
    print(f"✓ Wrote {kept_self_triples_count:,} RUTHLESSLY FILTERED self‑triples → {SELF_OUT_FILTERED.name}")
    print(f"✓ Wrote {total_link_triples_count:,} link‑triples → {LINK_OUT.name} (original logic, date filtered)")

# ---------- run ---------------------------------------------------
if __name__ == "__main__":
    print(f"Project ROOT set to: {ROOT}")
    print(f"Expecting raw Reddit data at: {RAW_REDDIT_JL}")
    if not RAW_REDDIT_JL.exists():
        print(f"ERROR: Raw Reddit data file not found at {RAW_REDDIT_JL}")
        print("Please ensure the file exists or adjust the ROOT path. You might need to create a dummy file for testing.")
    else:
        build_triples_ruthlessly()