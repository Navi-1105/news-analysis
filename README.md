# News Analysis System

A real-time news analysis system that fetches, processes, and analyzes news articles using natural language processing and vector search capabilities.

## Features

- Real-time news fetching from multiple trusted sources
- Advanced text analysis using TF-IDF and cosine similarity
- Vector-based semantic search
- Modern web interface with real-time updates
- Automatic news categorization and relevance scoring

## Prerequisites

- Python 3.8 or higher
- News API key (get one from [newsapi.org](https://newsapi.org))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Navi-1105/news-analysis.git
cd news-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your News API key:
```
NEWS_API_KEY=your_api_key_here
```

## Usage

1. Start the server:
```bash
python backend/api/main.py
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Use the interface to:
   - Search for news articles
   - Analyze news content
   - Get real-time updates

## Project Structure

```
news-analysis-system/
├── backend/
│   ├── api/
│   │   └── main.py
│   ├── rag/
│   │   └── vector_store.py
│   ├── news_fetcher.py
│   └── templates/
│       └── index.html
├── requirements.txt
├── .env
└── README.md
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [News API](https://newsapi.org) for providing the news data
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Tailwind CSS](https://tailwindcss.com) for the UI components 