import os
import sys

import traceback
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_fetcher import NewsFetcher
from rag.vector_store import VectorStore
from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
import json
from datetime import datetime

# Load environment variables


app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static',
            static_url_path='/static')
CORS(app)

# Initialize components
try:
    news_fetcher = NewsFetcher()
    vector_store = VectorStore()
    print("Successfully initialized NewsFetcher and VectorStore")
except Exception as e:
    print(f"Error initializing components: {str(e)}")
    print(traceback.format_exc())
    raise

@app.route('/')
def home():
    """Serve the web interface."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_news():
    """Analyze news based on user query."""
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
            
        # Search for relevant articles
        results = vector_store.search(query)
        
        if not results:
            return jsonify({'articles': []})
            
        # Format the results for the frontend
        articles = []
        for article, similarity in results:
            # Safely extract source name
            source_name = 'Unknown'
            if article.get('source'):
                if isinstance(article['source'], dict):
                    source_name = article['source'].get('name', 'Unknown')
                else:
                    source_name = str(article['source'])
            
            articles.append({
                'title': article.get('title', 'No Title'),
                'description': article.get('description', 'No description available'),
                'url': article.get('url', '#'),
                'source': source_name,
                'publishedAt': article.get('publishedAt', article.get('published_at', datetime.now().isoformat())),
                'similarity': float(similarity)  # Convert numpy float to Python float
            })
            
        return jsonify({'articles': articles})
        
    except Exception as e:
        print(f"Error in analyze_news: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/refresh', methods=['POST'])
def refresh_news():
    """Refresh the news database."""
    try:
        # Fetch new articles
        articles = news_fetcher.fetch_news()
        
        # Update vector store
        vector_store.update(articles)
        
        return jsonify({
            'message': f'Successfully refreshed {len(articles)} articles',
            'count': len(articles)
        })
        
    except Exception as e:
        print(f"Error in refresh_news: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure the templates directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), '../templates'), exist_ok=True)
    
    # Run the Flask app
    print("Starting News Analysis API...")
    app.run(host='0.0.0.0', port=8000, debug=True) 