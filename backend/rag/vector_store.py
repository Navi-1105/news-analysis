from collections import Counter
import re
from typing import List, Dict, Any, Tuple
import json
import os
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import traceback

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class VectorStore:
    def __init__(self):
        """Initialize the vector store."""
        print("Initializing VectorStore...")
        
        # Initialize NLTK
        nltk.download('punkt')
        nltk.download('stopwords')
        
        # Load articles from file
        self.articles = []
        self.vectors = None
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        # Try to load articles from file
        articles_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'articles.json')
        print(f"Looking for articles file at: {articles_file}")
        
        try:
            if os.path.exists(articles_file):
                print("Found articles file, loading...")
                with open(articles_file, 'r', encoding='utf-8') as f:
                    self.articles = json.load(f)
                print(f"Loaded {len(self.articles)} articles from file")
                
                if self.articles:
                    # Create vectors for loaded articles
                   
                    texts = [f"{article['title']} {article.get('description', article.get('content', ''))}" for article in self.articles]

                    self.vectors = self.vectorizer.fit_transform(texts)
                    print(f"Created vectors for {len(self.articles)} articles")
                else:
                    print("No articles to vectorize")
            else:
                print("No articles file found")
        except Exception as e:
            print(f"Error loading articles: {str(e)}")
            print(f"Error details: {traceback.format_exc()}")
            self.articles = []
            self.vectors = None
        
    def update(self, new_articles):
        """Update the vector store with new articles."""
        print(f"Updating vector store with {len(new_articles)} new articles")
        self.articles = new_articles

        # Create vectors for the new articles
        if self.articles:
            texts = [f"{article['title']} {article['description']}" for article in self.articles]
            self.vectors = self.vectorizer.fit_transform(texts)
            print(f"Created vectors for {len(self.articles)} articles")
        else:
            self.vectors = None
            print("No articles to vectorize")

        # Save articles to file
        try:
            articles_file = os.path.join(os.path.dirname(__file__), '..', 'articles.json')
            with open(articles_file, 'w', encoding='utf-8') as f:
                json.dump(self.articles, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(self.articles)} articles to file")
        except Exception as e:
            print(f"Error saving articles: {str(e)}")
            print(f"Full error details: {e.__class__.__name__}")
            
    def search(self, query, top_k=10):
        """Search for relevant articles using TF-IDF and cosine similarity."""
        print(f"Searching for: {query}")
        if not self.articles or self.vectors is None:
            print("No articles available for search")
            return []
            
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        print("Transformed query to vector")
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        print(f"Calculated similarities, max similarity: {similarities.max()}")
        
        # Get top k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return articles with their similarity scores
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include articles with some relevance
                results.append((self.articles[idx], float(similarities[idx])))
                
        print(f"Found {len(results)} relevant articles")
        return results

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better matching."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization of text"""
        # Convert to lowercase and split on non-alphanumeric characters
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def _update_word_frequencies(self):
        """Update word frequencies from all articles"""
        self.word_frequencies = Counter()
        for article in self.articles:
            # Use both title and content for better matching
            text = f"{article.get('title', '')} {article.get('content', '')}"
            words = self._tokenize(text)
            self.word_frequencies.update(words)

    def _calculate_tfidf(self, text: str) -> Dict[str, float]:
        """Calculate TF-IDF scores for a text"""
        words = self._tokenize(text)
        word_count = Counter(words)
        total_words = len(words)
        
        # Calculate TF-IDF scores
        scores = {}
        for word, count in word_count.items():
            # Term Frequency
            tf = count / total_words if total_words > 0 else 0
            # Inverse Document Frequency
            idf = math.log(len(self.articles) / (self.word_frequencies[word] + 1))
            scores[word] = tf * idf
        
        return scores

    def _calculate_similarity(self, query_scores: Dict[str, float], article_scores: Dict[str, float]) -> float:
        """Calculate cosine similarity between query and article scores"""
        # Get all unique words
        words = set(query_scores.keys()) | set(article_scores.keys())
        
        # Calculate dot product and magnitudes
        dot_product = sum(query_scores.get(word, 0) * article_scores.get(word, 0) for word in words)
        query_magnitude = math.sqrt(sum(score * score for score in query_scores.values()))
        article_magnitude = math.sqrt(sum(score * score for score in article_scores.values()))
        
        # Avoid division by zero
        if query_magnitude == 0 or article_magnitude == 0:
            return 0.0
            
        return dot_product / (query_magnitude * article_magnitude)

    def add_articles(self, new_articles: List[Dict[str, Any]]) -> None:
        """Add new articles to the store."""
        try:
            # Load existing articles
            existing_articles = []
            if os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'articles.json')):
                with open(os.path.join(os.path.dirname(__file__), '..', 'articles.json'), 'r', encoding='utf-8') as f:
                    existing_articles = json.load(f)
            
            # Add new articles
            all_articles = existing_articles + new_articles
            
            # Save updated articles
            with open(os.path.join(os.path.dirname(__file__), '..', 'articles.json'), 'w', encoding='utf-8') as f:
                json.dump(all_articles, f, indent=2)
            
            # Reload articles to update the vector store
            self.load_articles()
            print(f"Added {len(new_articles)} new articles. Total articles: {len(self.articles)}")
        except Exception as e:
            print(f"Error adding articles: {str(e)}")

    def clear(self):
        """Clear all articles"""
        self.articles = []
        self.word_frequencies = Counter()
        print("Cleared all articles from vector store")

    def _update_vectors(self):
        """Update vectors for all articles"""
        if self.articles:
            texts = [f"{article['title']} {article['description']}" for article in self.articles]
            self.vectors = self.vectorizer.fit_transform(texts)
            print(f"Updated vectors for {len(self.articles)} articles")
        else:
            print("No articles to update vectors") 