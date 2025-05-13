import re
import json
# import textwrap # Not used in this modification, but was in original context
import html
from pathlib import Path
from urllib.parse import urlparse
from collections import defaultdict
from datetime import datetime # Ensure datetime is imported

ROOT = Path(__file__).resolve().parents[1]  # project root
REDDIT_FILTERED_TRIPLES = ROOT / "reddit_posts" / "triples_reddit_self_ruthless_filtered.jsonl"
GATECH_TRIPLES = ROOT / "gatech_html" / "triples_gatech_cleaned.jsonl"

def load_gatech_triples(path: Path):
    gatech_triples = []
    with open(path, 'r') as f:
        for line in f:
            triple = json.loads(line)
            gatech_triples.append(triple)
    return gatech_triples

def load_reddit_triples(path: Path):
    reddit_triples = []
    with open(path, 'r') as f:
        for line in f:
            triple = json.loads(line)
            reddit_triples.append(triple)
    return reddit_triples

def find_urls_gatech(gatech_triples: list):
    gatech_urls = []
    for triple in gatech_triples:
        gatech_urls.append(triple['url'])
    return gatech_urls

def find_urls_reddit(reddit_triples: list):
    reddit_urls = []
    for triple in reddit_triples:
        reddit_urls.append(triple['url'])
    return reddit_urls

gatech_urls = find_urls_gatech(load_gatech_triples(GATECH_TRIPLES))
reddit_urls = find_urls_reddit(load_reddit_triples(REDDIT_FILTERED_TRIPLES))

print(f"gatech_urls: {len(gatech_urls)}")
print(f"reddit_urls: {len(reddit_urls)}")

with open("gatech_html_urls.json", "w") as f:
    json.dump(gatech_urls, f)

with open("reddit_urls.json", "w") as f:
    json.dump(reddit_urls, f)






