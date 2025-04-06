# models/pc_fix_model.py
import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sqlite3
from tqdm import tqdm
import joblib
from typing import List, Dict, Tuple, Any

class PCFixModel:
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        """
        Initialize the PC troubleshooting model with a transformer model.
        
        Args:
            model_name: The sentence transformer model to use
            device: The device to run the model on (None for auto-detection)
        """
        self.db_path = os.path.join("data", "pcfix.db")
        self.model_dir = os.path.join("data", "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Set device (CPU/GPU)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pre-trained transformer model for text embeddings
        print(f"Loading sentence transformer model: {model_name}")
        self.transformer = SentenceTransformer(model_name, device=self.device)
        
        # Load or initialize issue classifier model
        self.category_classifier = self._load_or_init_classifier()
        
        # Load cached embeddings if they exist
        self.issue_embeddings = None
        self.issue_data = None
        self._load_cached_data()
    
    def _load_or_init_classifier(self):
        """Load pre-trained classifier or initialize a new one"""
        classifier_path = os.path.join(self.model_dir, "category_classifier.joblib")
        
        if os.path.exists(classifier_path):
            print("Loading pre-trained category classifier...")
            return joblib.load(classifier_path)
        else:
            print("Initializing new category classifier...")
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _load_cached_data(self):
        """Load cached embeddings and issue data if available"""
        embeddings_path = os.path.join(self.model_dir, "issue_embeddings.npy")
        issue_data_path = os.path.join(self.model_dir, "issue_data.json")
        
        if os.path.exists(embeddings_path) and os.path.exists(issue_data_path):
            print("Loading cached embeddings and issue data...")
            self.issue_embeddings = np.load(embeddings_path)
            with open(issue_data_path, 'r') as f:
                self.issue_data = json.load(f)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector using the transformer model"""
        return self.transformer.encode(text, convert_to_numpy=True, show_progress_bar=False)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts to embedding vectors"""
        return self.transformer.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    def train(self, force_retrain=False):
        """
        Train or update the model with the latest data from the database.
        
        Args:
            force_retrain: Whether to force retraining even if cached data exists
        """
        if self.issue_embeddings is not None and self.issue_data is not None and not force_retrain:
            print("Using cached embeddings. Use force_retrain=True to retrain.")
            return
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all posts with their solutions
        print("Fetching posts and solutions from database...")
        cursor.execute("""
        SELECT p.id, p.title, p.content, p.category, 
               GROUP_CONCAT(s.content, '|||') as solutions,
               GROUP_CONCAT(s.confidence, '|||') as confidences
        FROM posts p
        LEFT JOIN solutions s ON p.id = s.post_id
        GROUP BY p.id
        """)
        rows = cursor.fetchall()
        
        if not rows:
            print("No data found in database. Please run the scraper first.")
            conn.close()
            return
        
        # Prepare data for training
        print("Preparing training data...")
        issue_texts = []
        categories = []
        issue_data = []
        
        for row in rows:
            # Combine title and content for the issue text
            issue_text = f"{row['title']} {row['content']}".strip()
            if not issue_text:
                continue  # Skip empty texts
                
            # Get solutions if available
            solutions = []
            if row['solutions']:
                solution_texts = row['solutions'].split('|||')
                solution_confidences = [float(c) for c in row['confidences'].split('|||')]
                
                for sol_text, conf in zip(solution_texts, solution_confidences):
                    if sol_text.strip():
                        solutions.append({"content": sol_text.strip(), "confidence": conf})
            
            # Add to training data
            issue_texts.append(issue_text)
            categories.append(row['category'])
            
            issue_data.append({
                "id": row['id'],
                "title": row['title'],
                "category": row['category'],
                "solutions": solutions
            })
        
        conn.close()
        
        # Train category classifier if we have enough data
        if len(categories) > 10:
            print("Generating embeddings for category classification...")
            X_embeddings = self.encode_batch(issue_texts)
            
            print("Training category classifier...")
            self.category_classifier.fit(X_embeddings, categories)
            
            # Save the trained classifier
            classifier_path = os.path.join(self.model_dir, "category_classifier.joblib")
            joblib.dump(self.category_classifier, classifier_path)
            print(f"Category classifier saved to {classifier_path}")
        
        # Generate and cache embeddings for all issues
        print("Generating and caching embeddings for all issues...")
        self.issue_embeddings = self.encode_batch(issue_texts)
        self.issue_data = issue_data
        
        # Save cached data
        np.save(os.path.join(self.model_dir, "issue_embeddings.npy"), self.issue_embeddings)
        with open(os.path.join(self.model_dir, "issue_data.json"), 'w') as f:
            json.dump(issue_data, f)
        
        print(f"Model trained on {len(issue_texts)} issues.")
    
    def predict_category(self, query: str) -> str:
        """Predict category for a new query"""
        query_embedding = self.encode_text(query)
        predicted_category = self.category_classifier.predict([query_embedding])[0]
        return predicted_category
    
    def search_similar_issues(self, query: str, top_n: int = 5) -> List[Dict]:
        """
        Search for issues similar to the query and return top matches with solutions.
        
        Args:
            query: The user's troubleshooting query
            top_n: Number of top results to return
            
        Returns:
            List of dictionaries containing issue matches with solutions
        """
        if self.issue_embeddings is None or self.issue_data is None:
            print("No cached embeddings found. Training model...")
            self.train()
            
            if self.issue_embeddings is None:
                return []  # Still no data after training attempt
        
        # Encode the query
        query_embedding = self.encode_text(query)
        
        # Calculate similarity to all issues
        similarities = cosine_similarity([query_embedding], self.issue_embeddings)[0]
        
        # Get indices of top matches
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        # Format results
        results = []
        for idx in top_indices:
            issue = self.issue_data[idx]
            results.append({
                "id": issue["id"],
                "title": issue["title"],
                "category": issue["category"],
                "similarity": float(similarities[idx]),
                "solutions": issue["solutions"]
            })
        
        return results
    
    def fine_tune(self, feedback_data: List[Dict[str, Any]]):
        """
        Fine-tune the model based on user feedback.
        
        Args:
            feedback_data: List of dictionaries with feedback information
                Each dict contains: {"query": str, "selected_solution_id": str, "helpful": bool}
        """
        # This would be implemented in a production system
        # For now, we'll just log the feedback
        feedback_file = os.path.join(self.model_dir, "user_feedback.jsonl")
        
        with open(feedback_file, 'a') as f:
            for item in feedback_data:
                f.write(json.dumps(item) + "\n")
        
        print(f"Feedback saved to {feedback_file}")
        # In a real implementation, you would periodically retrain with this feedback

# Utility function to demonstrate the model
def demo_model():
    model = PCFixModel()
    model.train()
    
    test_queries = [
        "My computer freezes during games after about 30 minutes",
        "Blue screen error when booting Windows",
        "Computer won't connect to WiFi but other devices work fine"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print(f"Predicted category: {model.predict_category(query)}")
        
        results = model.search_similar_issues(query, top_n=2)
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (Similarity: {result['similarity']:.2f}):")
            print(f"Title: {result['title']}")
            print(f"Category: {result['category']}")
            
            for j, solution in enumerate(result['solutions'][:2]):
                print(f"  Solution {j+1} [{solution['confidence']:.2f}]: {solution['content'][:100]}...")

if __name__ == "__main__":
    demo_model()