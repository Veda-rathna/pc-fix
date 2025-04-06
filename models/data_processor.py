# models/data_processor.py
import pandas as pd
import numpy as np
import spacy
import re
import spacy
import sqlite3
import os
from tqdm import tqdm

class PCFixDataProcessor:
    def __init__(self):
        # Load NLP models
        print("Loading NLP models...")
        self.nlp = spacy.load("en_core_web_md")
        
        # Define issue categories
        self.categories = {
            "hardware": ["gpu", "cpu", "ram", "memory", "motherboard", "psu", "power supply", 
                         "fan", "cooling", "ssd", "hdd", "hard drive", "graphics card"],
            "software": ["driver", "program", "application", "software", "update", "windows", 
                         "macos", "linux", "ubuntu", "game", "gaming"],
            "networking": ["internet", "wifi", "ethernet", "connection", "router", "network", 
                           "connectivity", "ping", "latency"],
            "peripheral": ["monitor", "keyboard", "mouse", "headset", "speaker", "display", 
                           "screen", "resolution"]
        }
        
        # Create database connection
        self.db_path = os.path.join("data", "pcfix.db")
        self.setup_database()
        
    def setup_database(self):
        """Create SQLite database and tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            score INTEGER,
            created_date TEXT,
            url TEXT,
            num_comments INTEGER,
            subreddit TEXT,
            category TEXT,
            embedding_file TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS solutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT,
            content TEXT,
            score INTEGER,
            is_solution INTEGER,
            confidence REAL,
            FOREIGN KEY (post_id) REFERENCES posts (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters and excessive whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def categorize_issue(self, text):
        """Categorize PC issue based on keywords"""
        text = text.lower()
        category_scores = {}
        
        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in text)
            category_scores[category] = score
        
        # Get the category with highest score
        if sum(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        else:
            return "other"
    
    def extract_keywords(self, text):
        """Extract key technical terms from text"""
        doc = self.nlp(text)
        keywords = []
        
        # Extract nouns and technical terms
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
                keywords.append(token.text.lower())
                
        # Add named entities
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG"]:
                keywords.append(ent.text.lower())
                
        return list(set(keywords))
    
    def process_reddit_data(self, posts_csv="reddit_pc_issues.csv", comments_csv="reddit_comments.csv"):
        """Process scraped Reddit data and store in database"""
        print("Processing Reddit data...")
        
        # Load CSV files
        posts_path = os.path.join("data", posts_csv)
        comments_path = os.path.join("data", comments_csv)
        
        if not os.path.exists(posts_path):
            print(f"Error: {posts_path} not found. Please run the scraper first.")
            return
            
        posts_df = pd.read_csv(posts_path)
        comments_df = pd.read_csv(comments_path) if os.path.exists(comments_path) else pd.DataFrame()
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Process each post
        embeddings_dir = os.path.join("data", "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        
        for idx, row in tqdm(posts_df.iterrows(), total=len(posts_df)):
            # Clean and combine title and content
            title = self.clean_text(row['title'])
            content = self.clean_text(row['selftext']) if 'selftext' in row else ""
            full_text = f"{title} {content}"
            
            # Categorize issue
            category = self.categorize_issue(full_text)
            
            # Generate embedding for the post
            doc = self.nlp(full_text)
            embedding = doc.vector
            embedding_file = f"post_{row['id']}.npy"
            np.save(os.path.join(embeddings_dir, embedding_file), embedding)
            
            # Store post in database
            cursor.execute('''
            INSERT OR REPLACE INTO posts 
            (id, title, content, score, created_date, url, num_comments, subreddit, category, embedding_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['id'], 
                title,
                content,
                row['score'],
                row['created_utc'],
                row['url'],
                row['num_comments'],
                row['subreddit'],
                category,
                embedding_file
            ))
            
            # Process associated comments if available
            if not comments_df.empty:
                post_comments = comments_df[comments_df['post_id'] == row['id']]
                
                for _, comment_row in post_comments.iterrows():
                    # Clean comment text
                    comment_text = self.clean_text(comment_row['body'])
                    
                    # Calculate solution confidence
                    is_solution = 1 if comment_row['is_solution'] else 0
                    confidence = min(comment_row['score'] / 10, 1.0) + (0.5 if is_solution else 0)
                    
                    # Store comment solution in database
                    cursor.execute('''
                    INSERT INTO solutions
                    (post_id, content, score, is_solution, confidence)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (
                        row['id'],
                        comment_text,
                        comment_row['score'],
                        is_solution,
                        confidence
                    ))
        
        conn.commit()
        conn.close()
        print(f"Processed {len(posts_df)} posts and stored in the database")
    
    def search_similar_issues(self, query, top_n=5):
        """
        Search for similar issues using semantic similarity
        """
        # Clean and encode the query
        clean_query = self.clean_text(query)
        query_doc = self.nlp(clean_query)
        query_embedding = query_doc.vector
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all posts
        cursor.execute("SELECT id, title, category, embedding_file FROM posts")
        posts = cursor.fetchall()
        
        # Calculate similarity scores
        results = []
        embeddings_dir = os.path.join("data", "embeddings")
        
        for post_id, title, category, embedding_file in posts:
            # Load the embedding
            embedding_path = os.path.join(embeddings_dir, embedding_file)
            if os.path.exists(embedding_path):
                post_embedding = np.load(embedding_path)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, post_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(post_embedding)
                )
                
                results.append({
                    'id': post_id,
                    'title': title,
                    'category': category,
                    'similarity': similarity
                })
        
        # Sort by similarity and get top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = results[:top_n]
        
        # Get solutions for top results
        final_results = []
        for result in top_results:
            cursor.execute('''
            SELECT content, confidence
            FROM solutions
            WHERE post_id = ?
            ORDER BY confidence DESC
            ''', (result['id'],))
            
            solutions = cursor.fetchall()
            
            final_results.append({
                'id': result['id'],
                'title': result['title'],
                'category': result['category'],
                'similarity': result['similarity'],
                'solutions': [{'content': s[0], 'confidence': s[1]} for s in solutions]
            })
        
        conn.close()
        return final_results

if __name__ == "__main__":
    processor = PCFixDataProcessor()
    # Test processing
    processor.process_reddit_data()
    
    # Test search
    test_query = "PC freezes when playing games after 30 minutes"
    results = processor.search_similar_issues(test_query)
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Similarity: {result['similarity']:.2f}):")
        print(f"Title: {result['title']}")
        print(f"Category: {result['category']}")
        print("Top Solutions:")
        for j, solution in enumerate(result['solutions'][:2]):
            print(f"  {j+1}. [{solution['confidence']:.2f}] {solution['content'][:100]}...")