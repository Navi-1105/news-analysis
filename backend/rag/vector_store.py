from collections import Counter
import re
from typing import List, Dict, Any, Tuple
import json
import os
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NewsVectorStore:
    def __init__(self, articles_file: str = "articles.json"):
        self.articles_file = articles_file
        self.articles = []
        self.word_frequencies = {}
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            max_features=5000
        )
        self.load_articles()

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better matching."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_articles(self) -> None:
        """Load articles from JSON file and prepare for search."""
        try:
            if os.path.exists(self.articles_file):
                with open(self.articles_file, 'r', encoding='utf-8') as f:
                    self.articles = json.load(f)
                print(f"Loaded {len(self.articles)} articles from {self.articles_file}")
                
                # Preprocess all article texts
                processed_texts = []
                for article in self.articles:
                    # Combine title and description for better matching
                    text = f"{article['title']} {article.get('description', '')}"
                    processed_text = self.preprocess_text(text)
                    processed_texts.append(processed_text)
                
                # Fit vectorizer on all processed texts
                if processed_texts:
                    self.vectorizer.fit(processed_texts)
            else:
                print(f"No articles file found at {self.articles_file}")
                self.articles = []
        except Exception as e:
            print(f"Error loading articles: {str(e)}")
            self.articles = []

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

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant articles using TF-IDF and cosine similarity."""
        if not self.articles:
            return []

        try:
            # Preprocess query
            processed_query = self.preprocess_text(query)
            
            # Transform query and articles
            query_vector = self.vectorizer.transform([processed_query])
            article_vectors = self.vectorizer.transform([
                self.preprocess_text(f"{article['title']} {article.get('description', '')}")
                for article in self.articles
            ])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, article_vectors).flatten()
            
            # Get top k results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # Filter out very low similarity scores
            results = []
            for idx in top_indices:
                similarity = float(similarities[idx])  # Convert to float for JSON serialization
                if similarity > 0.1:  # Only include results with similarity > 0.1
                    article = self.articles[idx].copy()
                    article['similarity'] = similarity
                    results.append(article)
            
            return results
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []

    def add_articles(self, new_articles: List[Dict[str, Any]]) -> None:
        """Add new articles to the store."""
        try:
            # Load existing articles
            existing_articles = []
            if os.path.exists(self.articles_file):
                with open(self.articles_file, 'r', encoding='utf-8') as f:
                    existing_articles = json.load(f)
            
            # Add new articles
            all_articles = existing_articles + new_articles
            
            # Save updated articles
            with open(self.articles_file, 'w', encoding='utf-8') as f:
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