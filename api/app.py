# api/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pc_fix_model import PCFixModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the model
model = PCFixModel()
model.train()  # This will load cached model if available or train a new one

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "message": "PC Fix API is running"}), 200

@app.route('/troubleshoot', methods=['POST'])
def troubleshoot():
    """Main endpoint for troubleshooting PC issues using the ML model"""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({"error": "Missing query parameter"}), 400
    
    query = data['query']
    num_results = data.get('num_results', 3)  # Default to 3 results
    
    try:
        # Get category prediction
        category = model.predict_category(query)
        
        # Search for similar issues using the ML model
        results = model.search_similar_issues(query, top_n=num_results)
        
        # Format the response
        response = {
            "query": query,
            "predicted_category": category,
            "results": []
        }
        
        for result in results:
            # Get top 3 solutions per result
            top_solutions = result.get('solutions', [])[:3]
            
            response["results"].append({
                "issue_title": result['title'],
                "category": result['category'],
                "relevance_score": float(result['similarity']),
                "solutions": [
                    {
                        "content": solution['content'],
                        "confidence": float(solution['confidence'])
                    } for solution in top_solutions
                ]
            })
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": "Failed to process your request", "details": str(e)}), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get all issue categories"""
    categories = ["hardware", "software", "networking", "peripheral", "other"]
    return jsonify(categories), 200

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get basic stats about the database and model"""
    import sqlite3
    
    conn = sqlite3.connect(model.db_path)
    cursor = conn.cursor()
    
    # Get post counts
    cursor.execute("SELECT COUNT(*) FROM posts")
    post_count = cursor.fetchone()[0]
    
    # Get solution counts
    cursor.execute("SELECT COUNT(*) FROM solutions")
    solution_count = cursor.fetchone()[0]
    
    # Get category distribution
    cursor.execute("""
    SELECT category, COUNT(*) as count 
    FROM posts 
    GROUP BY category
    ORDER BY count DESC
    """)
    categories = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    # Add model info
    model_info = {
        "model_type": "sentence-transformer",
        "embedding_size": 384 if model.issue_embeddings is not None else 0,
        "issues_indexed": len(model.issue_data) if model.issue_data is not None else 0,
    }
    
    return jsonify({
        "total_posts": post_count,
        "total_solutions": solution_count,
        "categories": categories,
        "model_info": model_info
    }), 200

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for model improvement"""
    data = request.json
    
    if not data or 'query' not in data or 'selected_solution_id' not in data:
        return jsonify({"error": "Missing required feedback parameters"}), 400
    
    # Save feedback for future fine-tuning
    feedback = {
        "query": data['query'],
        "selected_solution_id": data['selected_solution_id'],
        "helpful": data.get('helpful', True),
        "timestamp": data.get('timestamp')
    }
    
    # Submit to model for fine-tuning
    model.fine_tune([feedback])
    
    return jsonify({"status": "Feedback recorded successfully"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)