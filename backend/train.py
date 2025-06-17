import os
from rag.news_fetcher import NewsFetcher
from rag.vector_store import NewsVectorStore
from fine_tuning.model import NewsModel

def main():
    # Create necessary directories
    os.makedirs('models/fine_tuned', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Initialize components
    news_fetcher = NewsFetcher()
    vector_store = NewsVectorStore()
    news_model = NewsModel()

    print("Fetching recent news articles...")
    articles = news_fetcher.fetch_recent_news(days_back=7)
    
    if not articles:
        print("No articles fetched. Please check your API key and internet connection.")
        return

    print(f"Fetched {len(articles)} articles")

    # Save articles
    news_fetcher.save_articles(articles)
    print("Articles saved to data/news_articles.json")

    # Add articles to vector store
    print("Adding articles to vector store...")
    vector_store.add_articles(articles)
    print("Articles added to vector store")

    # Prepare training data
    print("Preparing training data...")
    training_texts = news_fetcher.prepare_training_data(articles)
    
    # Prepare dataset for fine-tuning
    print("Preparing dataset for fine-tuning...")
    train_dataset, data_collator = news_model.prepare_dataset(
        texts=training_texts,
        output_dir='data'
    )

    # Fine-tune the model
    print("Starting model fine-tuning...")
    news_model.fine_tune(
        train_dataset=train_dataset,
        data_collator=data_collator,
        output_dir='models/fine_tuned'
    )
    print("Model fine-tuning completed!")

if __name__ == "__main__":
    main() 