# reddit_india_comments.py

import praw
import json

# ---- Step 1: Reddit API credentials ----
reddit = praw.Reddit(
          client_id='Kx8XlkSqC1FHsxxP_LfVzQ',
    client_secret='3TzGZESZkI0k87DQe20gvre6lIItjw',
    user_agent='Meras/0.1 by u/Mindless-Eye979'
  
)

# ---- Step 2: Select subreddit ----
subreddit = reddit.subreddit("India")  # India subreddit

# ---- Step 3: Collect comments ----
all_comments = []

# Fetch top 100 posts (you can change limit as needed)
for post in subreddit.hot(limit=100):
    post.comments.replace_more(limit=0)  # Expand "MoreComments"
    for comment in post.comments.list():
        all_comments.append({
            "post_id": post.id,
            "post_title": post.title,
            "comment_id": comment.id,
            "comment_body": comment.body,
            "comment_score": comment.score
        })

# ---- Step 4: Save comments to JSON ----
with open("india_comments.json", "w", encoding="utf-8") as f:
    json.dump(all_comments, f, ensure_ascii=False, indent=4)

print(f"Saved {len(all_comments)} comments to india_comments.json")
# ---- Step 5: Load JSON into DataFrame ----
import pandas as pd
import json

with open("india_comments.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(df.head())
