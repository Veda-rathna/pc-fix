# scripts/reddit_scraper.py
import os
import praw
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import time
from tqdm import tqdm

# Load environment variables
load_dotenv()

class RedditScraper:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        self.subreddits = ["techsupport", "buildapc", "pcmasterrace"]
        
    def scrape_posts(self, subreddit_name, limit=100, timeframe="month"):
        """
        Scrape posts from a specific subreddit
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        
        if timeframe == "day":
            posts = subreddit.top("day", limit=limit)
        elif timeframe == "week":
            posts = subreddit.top("week", limit=limit)
        elif timeframe == "month":
            posts = subreddit.top("month", limit=limit)
        else:
            posts = subreddit.top("all", limit=limit)
            
        post_data = []
        
        print(f"Scraping {limit} posts from r/{subreddit_name}...")
        for post in tqdm(posts, total=limit):
            # Skip non-troubleshooting posts
            if not any(keyword in post.title.lower() for keyword in 
                      ["help", "issue", "problem", "troubleshoot", "fix", 
                       "error", "broken", "repair", "not working"]):
                continue
                
            # Get comments
            post.comments.replace_more(limit=15)  # Load more comments, limit to avoid API throttling
            comments = []
            
            # Find comments that appear to contain solutions
            for comment in post.comments.list():
                if comment.score > 2:  # Only consider upvoted comments
                    comments.append({
                        "body": comment.body,
                        "score": comment.score,
                        "is_solution": "solution" in comment.body.lower() or 
                                      "fixed" in comment.body.lower() or 
                                      "solved" in comment.body.lower() or
                                      "working" in comment.body.lower()
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
            
            # Sleep briefly to avoid hitting rate limits
            time.sleep(0.5)
            
        return post_data
    
    def scrape_all_subreddits(self, limit=100):
        """
        Scrape posts from all configured subreddits
        """
        all_posts = []
        
        for subreddit in self.subreddits:
            posts = self.scrape_posts(subreddit, limit=limit)
            all_posts.extend(posts)
            
        return all_posts
    
    def save_to_csv(self, posts, filename="reddit_pc_issues.csv"):
        """
        Save scraped posts to CSV file
        """
        # Create a flattened dataframe for the main post data
        df_posts = pd.DataFrame([{k: v for k, v in post.items() if k != 'comments'} 
                               for post in posts])
        
        # Save to CSV
        output_path = os.path.join("data", filename)
        df_posts.to_csv(output_path, index=False)
        print(f"Saved {len(df_posts)} posts to {output_path}")
        
        # Also save comments to a separate CSV
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
    posts = scraper.scrape_all_subreddits(limit=100)  # Start with a small number for testing
    scraper.save_to_csv(posts)