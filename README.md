# StockRadar

StockRadar is a stock market tracking application that provides portfolio management, stock recommendations, and news impact analysis.

## Project Structure

```
/
├── app.py               # Main FastAPI backend application
├── crawl.py             # RSS feed crawler for financial news
├── index.py             # Script to build search indexes for news articles
├── index.html           # Static HTML frontend (alternative to React frontend)
├── stockradar.db        # SQLite database for portfolio storage
├── .env                 # Environment variables
├── package.json         # Node.js dependencies
├── stockradar-ui/       # React frontend application
│   ├── app.js           # Main React application
│   ├── src/             # React source code
│   └── package.json     # React dependencies
└── data/                # Data storage
    ├── raw_articles/    # Crawled news articles
    └── index/           # Search indexes for news articles
```

## Features

- Portfolio management (add, edit, delete holdings)
- Stock recommendations based on portfolio similarity
- News impact analysis with sentiment scoring
- Live price tracking with yfinance

## Setup

### Prerequisites

- Python 3.8+
- Node.js 16+
- pip
- npm

### Backend Setup

1. Install Python dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run the API server:

   ```
   uvicorn app:app --reload
   ```

3. Build the search indexes (optional):

   ```
   python index.py
   ```

4. Crawl for news (optional):
   ```
   python crawl.py
   ```

### Frontend Setup

#### Option 1: Static HTML

The static HTML frontend is served directly by FastAPI at the root URL.

#### Option 2: React Frontend

1. Navigate to the React app directory:

   ```
   cd stockradar-ui
   ```

2. Install dependencies:

   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

## Usage

1. Open your browser to http://localhost:8000 for the static HTML interface or http://localhost:3000 for the React interface.
2. Add stocks to your portfolio using the "Add Holding" button.
3. View recommendations based on your portfolio.
4. Check news impact analysis to see how recent news affects your holdings.

## Environment Variables

Create a `.env` file with the following:

```
NEWSAPI_KEY=your_newsapi_key
DATABASE_URL=sqlite:///./stockradar.db
```

## License

MIT
