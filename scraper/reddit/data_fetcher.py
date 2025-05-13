import os, json, time, pathlib, math, datetime as dt
from dotenv import load_dotenv
import praw
from tqdm import tqdm

load_dotenv()

reddit = praw.Reddit(
    client_id     = os.getenv("REDDIT_CLIENT_ID"),
    client_secret = os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent    = os.getenv("REDDIT_USER_AGENT"),
)
SUB = "gatech"
WINDOW = dt.timedelta(days=3)
START  = dt.datetime(2015,1,1)
END    = dt.datetime.utcnow()

total_windows = math.ceil((END - START) / WINDOW)
pbar = tqdm(total=total_windows, desc="Time slices")
cur = START
count = 0

# Open in append mode; will create the file if it doesn't exist
with open("posts_gatech.jsonl", "a") as out:
    while cur < END:
        nxt = min(cur + WINDOW, END)
        post_count_in_window = 0
        #query = f"timestamp:{int(cur.timestamp())}..{int(nxt.timestamp())}"
        query = "e"
        for post in reddit.subreddit(SUB).search(
                query,
                sort="new",
                syntax="lucene",
                time_filter="all",
                limit=None
            ):
            out.write(json.dumps({
                "id": post.id,
                "created_utc": post.created_utc,
                "author": str(post.author),
                "title": post.title,
                "selftext": post.selftext,
                "url": post.url,
                "permalink": "https://reddit.com"+post.permalink,
                "score": post.score
            }) + "\n")
            count += 1
            post_count_in_window += 1

        out.flush()  # ensure every window is on disk
        print(f"Found {post_count_in_window} posts in window {cur} to {nxt}")

        cur = nxt
        pbar.update(1)
        time.sleep(1)   # stay under 60 req/min

pbar.close()
print(f"✔ Total posts fetched: {count:,}")  