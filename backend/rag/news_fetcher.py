from newsapi import NewsApiClient
import os
from typing import List, Dict
from datetime import datetime, timedelta
import json

class NewsFetcher:
    def __init__(self):
        self.api_key = "eb12454f8c2f4c20ab38261e6cb66b9a"  # Your News API key
        self.newsapi = NewsApiClient(api_key=self.api_key)

    def fetch_recent_news(self, category: str = None, language: str = 'en', 
                         days_back: int = 7) -> List[Dict[str, str]]:
        """
        Fetch recent news articles
        """
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)

        # Fetch news
        try:
            response = self.newsapi.get_everything(
                q='technology OR AI OR artificial intelligence',  # Default query
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language=language,
                sort_by='relevancy',
                page_size=100
            )

            # Process articles
            articles = []
            for article in response['articles']:
                if article['content'] and article['title']:  # Only include articles with content
                    articles.append({
                        'title': article['title'],
                        'content': article['content'],
                        'source': article['source']['name'],
                        'date': article['publishedAt'],
                        'url': article['url']
                    })

            return articles

        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return []

    def save_articles(self, articles: List[Dict[str, str]], filename: str = 'news_articles.json'):
        """
        Save articles to a JSON file
        """
        os.makedirs('data', exist_ok=True)
        with open(f'data/{filename}', 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2)

    def load_articles(self, filename: str = 'news_articles.json') -> List[Dict[str, str]]:
        """
        Load articles from a JSON file
        """
        try:
            with open(f'data/{filename}', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def prepare_training_data(self, articles: List[Dict[str, str]]) -> List[str]:
        """
        Prepare articles for model training
        """
        training_texts = []
        for article in articles:
            # Create a formatted text for training
            text = f"Title: {article['title']}\nContent: {article['content']}\nSource: {article['source']}\nDate: {article['date']}\n\n"
            training_texts.append(text)
        return training_texts 