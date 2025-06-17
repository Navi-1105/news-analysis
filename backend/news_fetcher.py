import os
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

class NewsFetcher:
    def __init__(self):
        # Try to load environment variables
        load_dotenv()
        
        # Get API key from environment variable or use default
        self.api_key = os.getenv('NEWS_API_KEY', 'eb12454f8c2f4c20ab38261e6cb66b9a')
        print(f"Using API key: {self.api_key[:8]}...{self.api_key[-4:]}")
        
        self.api = NewsApiClient(api_key=self.api_key)
        self.articles_file = "articles.json"
        
        # Trusted news sources for better quality
        self.trusted_sources = [
            'reuters.com', 'apnews.com', 'bloomberg.com', 'bbc.com',
            'cnn.com', 'nbcnews.com', 'cbsnews.com', 'abcnews.go.com',
            'foxnews.com', 'theguardian.com', 'washingtonpost.com',
            'nytimes.com', 'wsj.com', 'time.com', 'newsweek.com',
            'usatoday.com', 'npr.org', 'aljazeera.com', 'dw.com'
        ]

        # Search categories for broader coverage
        self.categories = [
            'general', 'business', 'technology', 'science',
            'health', 'sports', 'entertainment'
        ]

    def fetch_news(self) -> List[Dict[str, Any]]:
        """Fetch news articles from various sources."""
        try:
            # Calculate date range (last 7 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            all_articles = []
            
            # Fetch from each category
            for category in self.categories:
                print(f"\nFetching {category} news...")
                
                # Fetch top headlines for this category
                headlines = self.api.get_top_headlines(
                    language='en',
                    category=category,
                    page_size=100,
                    from_param=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d')
                )
                
                if headlines.get('status') == 'ok':
                    print(f"Found {len(headlines.get('articles', []))} {category} headlines")
                    all_articles.extend(self._process_articles(headlines.get('articles', [])))
            
            # Fetch everything with broader search
            print("\nFetching everything with broader search...")
            search_queries = [
                "technology OR AI OR artificial intelligence",
                "business OR economy OR market",
                "science OR research OR discovery",
                "health OR medical OR healthcare",
                "world OR international OR global",
                "politics OR government OR policy",
                "environment OR climate OR weather"
            ]
            
            for query in search_queries:
                print(f"\nSearching for: {query}")
                everything = self.api.get_everything(
                    q=query,
                    language='en',
                    page_size=100,
                    from_param=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    sort_by='relevancy'
                )
                
                if everything.get('status') == 'ok':
                    print(f"Found {len(everything.get('articles', []))} articles for query: {query}")
                    all_articles.extend(self._process_articles(everything.get('articles', [])))
            
            # Remove duplicates based on URL
            unique_articles = {article['url']: article for article in all_articles}.values()
            unique_articles = list(unique_articles)
            
            # Sort by published date (newest first)
            unique_articles.sort(key=lambda x: x['published_at'], reverse=True)
            
            # Save to file
            self._save_articles(unique_articles)
            
            print(f"\nTotal unique articles fetched: {len(unique_articles)}")
            return unique_articles
            
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return []

    def _process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean articles."""
        processed = []
        for article in articles:
            # Skip articles without required fields
            if not all(k in article for k in ['title', 'url', 'source']):
                continue
                
            # Skip articles from untrusted sources
            source_domain = article['source'].get('id', '').lower()
            if not any(trusted in source_domain for trusted in self.trusted_sources):
                continue
            
            # Skip articles with very short content
            if len(article.get('description', '')) < 50:
                continue
            
            processed_article = {
                'title': article['title'],
                'description': article.get('description', ''),
                'url': article['url'],
                'source': article['source'].get('name', 'Unknown'),
                'published_at': article.get('publishedAt', datetime.now().isoformat()),
                'content': article.get('content', ''),
                'author': article.get('author', 'Unknown')
            }
            processed.append(processed_article)
        
        return processed

    def _save_articles(self, articles: List[Dict[str, Any]]) -> None:
        """Save articles to JSON file."""
        try:
            with open(self.articles_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2)
            print(f"Saved {len(articles)} articles to {self.articles_file}")
        except Exception as e:
            print(f"Error saving articles: {str(e)}")

    def get_article_by_url(self, url: str) -> Dict[str, Any]:
        """Get a specific article by URL."""
        try:
            if os.path.exists(self.articles_file):
                with open(self.articles_file, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                for article in articles:
                    if article['url'] == url:
                        return article
        except Exception as e:
            print(f"Error getting article: {str(e)}")
        return None 