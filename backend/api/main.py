import os
import sys
from dotenv import load_dotenv
import traceback
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_fetcher import NewsFetcher
from rag.vector_store import NewsVectorStore
from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS

# Load environment variables
load_dotenv()

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static',
            static_url_path='/static')
CORS(app)

# Initialize components
try:
    news_fetcher = NewsFetcher()
    vector_store = NewsVectorStore()
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
            return jsonify({'error': 'No query provided'}), 400
            
        print(f"Analyzing query: {query}")
        
        # Get relevant articles from vector store
        relevant_articles = vector_store.search(query)
        print(f"Found {len(relevant_articles)} relevant articles")
        
        # Generate answer using the articles
        answer = f"Based on the latest news, here's what I found about '{query}':\n\n"
        
        if not relevant_articles:
            answer = "I couldn't find any recent news articles related to your query. Try refreshing the news database or using different search terms."
        else:
            # Group articles by source
            articles_by_source = {}
            for article in relevant_articles:
                source = article['source']
                if source not in articles_by_source:
                    articles_by_source[source] = []
                articles_by_source[source].append(article)
            
            # Add summary for each source
            for source, articles in articles_by_source.items():
                answer += f"From {source}:\n"
                for article in articles[:3]:  # Limit to top 3 articles per source
                    answer += f"- {article['title']}\n"
                answer += "\n"
            
            answer += "These are the most relevant recent developments. You can click on any article to read more details."
        
        return jsonify({
            'answer': answer,
            'relevant_articles': relevant_articles
        })
        
    except Exception as e:
        print(f"Error in analyze_news: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/refresh', methods=['GET', 'POST'])
def refresh_news():
    """Refresh the news database."""
    try:
        print("Refreshing news database...")
        articles = news_fetcher.fetch_news()
        print(f"Fetched {len(articles)} articles")
        
        vector_store.update_articles(articles)
        print("Vector store updated successfully")
        
        return jsonify({
            'message': f'Successfully refreshed news database with {len(articles)} articles'
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