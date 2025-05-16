# StockRadar: Intelligent Stock Market Analysis Platform

StockRadar is an intelligent stock market analysis platform that combines portfolio management, machine learning-based stock recommendations, and real-time news impact analysis.


## Features

- Portfolio management (add, edit, delete holdings)
- Stock recommendations based on portfolio similarity
- News impact analysis with sentiment scoring
- Live price tracking with yfinance
- Real-time RSS feed processing
- Interactive data visualization

## Tech Stack

### Backend

- FastAPI (Python)
- SQLite with SQLAlchemy
- yfinance for stock data
- TextBlob for sentiment analysis
- FAISS for similarity search

### Frontend

- React
- Chart.js for visualizations
- Material-UI components

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- pip
- npm

### Backend Setup

1. Clone the repository:

```bash
git clone https://github.com/chirumamilla1522/666.git
cd 666
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `.env` file in the root directory with:

```
DATABASE_URL=sqlite:///./stockradar.db
```

5. Start the FastAPI server:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. Navigate to the React app directory:

```bash
cd stockradar-ui
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm start
```

## Usage

1. Open your browser to:

   - Static HTML interface: http://localhost:8000
   - React interface: http://localhost:3000

2. Add stocks to your portfolio using the "Add Holding" button
3. View recommendations based on your portfolio
4. Check news impact analysis to see how recent news affects your holdings

## API Documentation

Once the server is running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
/
├── app.py               # Main FastAPI backend application
├── crawl.py             # RSS feed crawler for financial news
├── index.py             # Script to build search indexes
├── index.html           # Static HTML frontend
├── stockradar.db        # SQLite database
├── requirements.txt     # Python dependencies
├── stockradar-ui/       # React frontend
│   ├── src/            # React source code
│   └── package.json    # React dependencies
└── data/               # Data storage
    ├── raw_articles/   # Crawled news articles
    └── index/          # Search indexes
```

## Development

### Running Tests

```bash
pytest
```

### Code Style

```bash
black .
flake8
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Course: 601.666 Information Retrieval and Web Agents
- Institution: Johns Hopkins University
- Instructor: [Instructor Name]
