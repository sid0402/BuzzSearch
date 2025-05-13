import os, json, time, pathlib
from dotenv import load_dotenv
import praw
from tqdm import tqdm

load_dotenv()

reddit = praw.Reddit(
    client_id     = os.getenv("REDDIT_CLIENT_ID"),
    client_secret = os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent    = os.getenv("REDDIT_USER_AGENT"),
)

SUB      = "gatech"
OUT_PATH = pathlib.Path("posts_gatech_new.jsonl")
BATCH    = 100                 # 1–100; Reddit’s hard limit per request

after    = None                # the pagination cursor
seen     = set()               # post IDs we’ve already saved
total    = 0

with open("posts_gatech_new.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        seen.add(data["id"])
    
print(f"seen: {len(seen)}")


query_letters = "%&*()_+-=[]{}|;:?cdfghijklmnopqrstuvwxyz"
for letter in query_letters:
    after = None
    QUERY = letter
    print(f"QUERY: {QUERY}")
    with OUT_PATH.open("a") as out:
        pbar = tqdm(desc="fetched", unit="posts")
        while True:
                # ---- build one request -------------------------------------------
            params = {"limit": BATCH}
            if after:
                params["after"] = after

            batch = list(
                reddit.subreddit(SUB).search(
                        QUERY,      
                        syntax="lucene",
                        sort="new",
                        params=params,
                    )
                )
            if not batch:                 # no more results
                break

            for post in reversed(batch):  # reverse ⇒ oldest first
                if post.id in seen:
                    print(f"already seen: {post.id}")
                    continue
                seen.add(post.id)

                out.write(json.dumps({
                    "id":          post.id,
                    "created_utc": post.created_utc,
                    "author":      str(post.author),
                    "title":       post.title,
                    "selftext":    post.selftext,
                    "url":         post.url,
                    "permalink":   "https://reddit.com" + post.permalink,
                    "score":       post.score,
                }) + "\n")

                total += 1
                pbar.update(1)

        # ------------------------------------------------------------------
        # the fullname of the newest item in *this* batch becomes `after`
        # for the *next* request
            after = batch[0].fullname        # batch[0] is newest because sort="new"
            pbar.close()

            print(f"after: {after}")

print(f"✔ saved {total:,} unique posts → {OUT_PATH}")