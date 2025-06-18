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
        nltk.download('punkt')
        nltk.download('stopwords')
        self.articles = []
        self.vectors = None
        self.word_frequencies = Counter()
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        # Load articles from file
        self._load_articles()

    def _extract_article_text(self, article: Dict[str, Any]) -> str:
        """Safely extract text from an article, handling missing or None fields."""
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        content = article.get('content', '') or ''
        
        # Combine title with either description or content, preferring description
        text_parts = [title]
        if description:
            text_parts.append(description)
        elif content:
            text_parts.append(content)
        
        return ' '.join(text_parts)

    def _load_articles(self):
        """Load articles from file and create vectors."""
        articles_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'articles.json')
        print(f"Looking for articles file at: {articles_file}")
        
        try:
            if os.path.exists(articles_file):
                print("Found articles file, loading...")
                with open(articles_file, 'r', encoding='utf-8') as f:
                    self.articles = json.load(f)
                print(f"Loaded {len(self.articles)} articles from file")
                
                if self.articles:
                    # Use consistent text extraction
                    texts = [self._extract_article_text(article) for article in self.articles]
                    self.vectors = self.vectorizer.fit_transform(texts)
                    self._update_word_frequencies()
                    print(f"Created vectors for {len(self.articles)} articles")
                else:
                    print("No articles to vectorize")
            else:
                print("No articles file found")
                self.articles = []
                self.vectors = None
        except Exception as e:
            print(f"Error loading articles: {str(e)}")
            print(f"Error details: {traceback.format_exc()}")
            self.articles = []
            self.vectors = None

    def update(self, new_articles):
        """Update the vector store with new articles."""
        print(f"Updating vector store with {len(new_articles)} new articles")
        self.articles = new_articles
        
        if self.articles:
            # Use consistent text extraction
            texts = [self._extract_article_text(article) for article in self.articles]
            self.vectors = self.vectorizer.fit_transform(texts)
            self._update_word_frequencies()
            print(f"Created vectors for {len(self.articles)} articles")
        else:
            self.vectors = None
            print("No articles to vectorize")
        
        # Save updated articles to file
        self._save_articles()

    def _save_articles(self):
        """Save articles to file."""
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
        
        # Transform query to vector and calculate similarities
        query_vector = self.vectorizer.transform([query])
        print("Transformed query to vector")
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        print(f"Calculated similarities, max similarity: {similarities.max()}")
        
        # Get top k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.articles[idx], float(similarities[idx])))
        
        print(f"Found {len(results)} relevant articles")
        return results

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better matching."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization of text"""
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def _update_word_frequencies(self):
        """Update word frequencies from all articles"""
        self.word_frequencies = Counter()
        for article in self.articles:
            # Use consistent text extraction
            text = self._extract_article_text(article)
            words = self._tokenize(text)
            self.word_frequencies.update(words)

    def _calculate_tfidf(self, text: str) -> Dict[str, float]:
        """Calculate TF-IDF scores for a text using sklearn vectorizer."""
        # Preprocess and tokenize the input text
        tokens = self._tokenize(text)
        if not tokens or self.vectors is None:
            return {}

        try:
            # Get feature names from the vectorizer
            feature_names = self.vectorizer.get_feature_names_out()
            # Transform the text to a TF-IDF vector
            tfidf_vector = self.vectorizer.transform([text])
            # Convert the sparse vector to a dense array
            tfidf_scores = tfidf_vector.toarray()[0]

            # Map feature names to their TF-IDF scores
            tfidf_dict = {feature: tfidf_scores[idx] for idx, feature in enumerate(feature_names) if tfidf_scores[idx] > 0}
            return tfidf_dict
        except Exception as e:
            print(f"Error calculating TF-IDF: {str(e)}")
            return {}

    def _calculate_tfidf_manual(self, text: str) -> Dict[str, float]:
        """Manual TF-IDF calculation implementation."""
        if not self.word_frequencies:
            self._update_word_frequencies()
        
        words = self._tokenize(text)
        word_count = Counter(words)
        total_words = len(words)
        
        # Calculate TF-IDF scores manually
        scores = {}
        for word, count in word_count.items():
            # Term Frequency: count of word in document / total words in document
            tf = count / total_words if total_words > 0 else 0
            
            # Inverse Document Frequency: log(total documents / documents containing word)
            docs_with_word = sum(1 for article in self.articles 
                               if word in self._tokenize(self._extract_article_text(article)))
            idf = math.log(len(self.articles) / (docs_with_word + 1)) if docs_with_word > 0 else 0
            
            scores[word] = tf * idf
        
        return scores

    def _calculate_similarity(self, query_scores: Dict[str, float], article_scores: Dict[str, float]) -> float:
        """Calculate cosine similarity between query and article TF-IDF scores."""
        # Get all unique words from both dictionaries
        words = set(query_scores.keys()) | set(article_scores.keys())
        
        # Calculate dot product and magnitudes for cosine similarity
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
            articles_path = os.path.join(os.path.dirname(__file__), '..', 'articles.json')
            if os.path.exists(articles_path):
                with open(articles_path, 'r', encoding='utf-8') as f:
                    existing_articles = json.load(f)
            
            # Combine existing and new articles
            all_articles = existing_articles + new_articles
            
            # Update the vector store
            self.articles = all_articles
            self._update_vectors()
            self._save_articles()
            
            print(f"Added {len(new_articles)} new articles. Total articles: {len(self.articles)}")
        except Exception as e:
            print(f"Error adding articles: {str(e)}")

    def _update_vectors(self):
        """Update vectors for all articles."""
        if self.articles:
            texts = [self._extract_article_text(article) for article in self.articles]
            self.vectors = self.vectorizer.fit_transform(texts)
            self._update_word_frequencies()
            print(f"Updated vectors for {len(self.articles)} articles")
        else:
            self.vectors = None
            print("No articles to update vectors")

    def delete_article(self, article_id: str) -> bool:
        """Delete an article by ID."""
        try:
            # Find and remove article by ID
            original_count = len(self.articles)
            self.articles = [article for article in self.articles if article.get('id') != article_id]
            
            if len(self.articles) < original_count:
                # Update vectors after deletion
                self._update_vectors()
                self._save_articles()
                print(f"Deleted article with ID: {article_id}")
                return True
            else:
                print(f"Article with ID {article_id} not found")
                return False
        except Exception as e:
            print(f"Error deleting article: {str(e)}")
            return False

    def clear(self):
        """Clear all articles from the vector store."""
        self.articles = []
        self.vectors = None
        self.word_frequencies = Counter()
        print("Cleared all articles from vector store")

    def get_article_count(self) -> int:
        """Get the total number of articles in the store."""
        return len(self.articles)

    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.word_frequencies) if self.word_frequencies else 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_articles': len(self.articles),
            'vocabulary_size': len(self.word_frequencies) if self.word_frequencies else 0,
            'vector_dimensions': self.vectors.shape[1] if self.vectors is not None else 0,
            'has_vectors': self.vectors is not None
        }
