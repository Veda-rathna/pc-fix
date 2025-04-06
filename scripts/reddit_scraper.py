# scripts/reddit_scraper.py

import os
import praw
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import time
from tqdm import tqdm
import re
from collections import defaultdict

# Load environment variables
load_dotenv()

class RedditScraper:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        self.subreddits = ["techsupport", "buildapc", "pcmasterrace", "windows10", "hardware", "computers", "linux4noobs"]
        self.keywords = [
            "help", "issue", "problem", "troubleshoot", "fix", 
            "error", "broken", "repair", "not working", "crash", 
            "freeze", "bsod", "slow", "boot", "shutdown", "restart", 
            "fan", "overheat", "lag", "glitch", "hang"
        ]
        self.keyword_pattern = re.compile(r"\b(" + "|".join(map(re.escape, self.keywords)) + r")\b", re.IGNORECASE)

    def post_matches_keywords(self, title, selftext):
        return bool(self.keyword_pattern.search(title)) or bool(self.keyword_pattern.search(selftext))

    def scrape_posts(self, subreddit_name, limit=3000, timeframe_list=None, sleep_time=0.3):
        if timeframe_list is None:
            timeframe_list = ["week", "month", "year", "all"]

        subreddit = self.reddit.subreddit(subreddit_name)
        print(f"Fetching up to {limit} PC issue posts from r/{subreddit_name}...")

        seen_ids = set()
        post_data = []

        for timeframe in timeframe_list:
            raw_limit = limit * 3  # Over-fetching to improve keyword filtering
            try:
                posts = subreddit.top(timeframe, limit=raw_limit)
            except Exception as e:
                print(f"Failed to get top posts from r/{subreddit_name} for {timeframe}: {e}")
                continue

            for post in tqdm(posts, desc=f"{subreddit_name} ({timeframe})"):
                if post.id in seen_ids:
                    continue
                if not self.post_matches_keywords(post.title, post.selftext):
                    continue

                try:
                    post.comments.replace_more(limit=10)
                except Exception as e:
                    print(f"Error loading comments: {e}")
                    continue

                comments = []
                for comment in post.comments.list():
                    if comment.score > 2:
                        comments.append({
                            "body": comment.body,
                            "score": comment.score,
                            "is_solution": any(word in comment.body.lower() for word in ["solution", "fixed", "solved", "working"])
                        })

                post_data.append({
                    "id": post.id,
                    "title": post.title,
                    "selftext": post.selftext,
                    "score": post.score,
                    "created_utc": datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d"),
                    "url": post.url,
                    "num_comments": post.num_comments,
                    "comments": comments,
                    "subreddit": subreddit_name
                })

                seen_ids.add(post.id)

                if len(post_data) >= limit:
                    break

                time.sleep(sleep_time)

            if len(post_data) >= limit:
                break

        print(f"Collected {len(post_data)} valid PC issue posts from r/{subreddit_name}")
        return post_data

    def scrape_all_subreddits(self, limit=3000):
        all_posts = []
        for subreddit in self.subreddits:
            posts = self.scrape_posts(subreddit, limit=limit)
            all_posts.extend(posts)
        return all_posts

    def save_to_csv(self, posts, filename="reddit_pc_issues.csv"):
        os.makedirs("data", exist_ok=True)

        df_posts = pd.DataFrame([{k: v for k, v in post.items() if k != 'comments'} for post in posts])
        post_path = os.path.join("data", filename)
        df_posts.to_csv(post_path, index=False)
        print(f"Saved {len(df_posts)} posts to {post_path}")

        comments_data = []
        for post in posts:
            post_id = post['id']
            for comment in post['comments']:
                comment['post_id'] = post_id
                comments_data.append(comment)

        if comments_data:
            df_comments = pd.DataFrame(comments_data)
            comments_path = os.path.join("data", "reddit_comments.csv")
            df_comments.to_csv(comments_path, index=False)
            print(f"Saved {len(df_comments)} comments to {comments_path}")


if __name__ == "__main__":
    scraper = RedditScraper()
    posts = scraper.scrape_all_subreddits(limit=3000)  # Increase to 3000 per subreddit
    scraper.save_to_csv(posts)
