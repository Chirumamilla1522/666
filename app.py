import os
import ssl
import uuid
import datetime
import glob
import json
import logging
import warnings
import time
import random
from pathlib import Path
from typing import List
from urllib.parse import urlparse
from functools import lru_cache

import yfinance as yf
import pandas as pd
from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Query,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, Column, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import re

import crawl  # your crawler

# ─── Suppress warnings & SSL ────────────────────────────────────────────────────
from bs4 import BeautifulSoup, GuessedAtParserWarning
import urllib3
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ─── Load environment (optional Gemini config) ─────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
# import google.generativeai as genai
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# ─── FastAPI setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="StockRadar Core API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse("index.html")


# ─── Database (SQLite + SQLAlchemy) ─────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stockradar.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class HoldingModel(Base):
    __tablename__ = "portfolio"
    id        = Column(String(36), primary_key=True, index=True)
    ticker    = Column(String(10), nullable=False, unique=True, index=True)
    quantity  = Column(Float, nullable=False)
    avg_price = Column(Float, nullable=False)

class ArticleModel(Base):
    __tablename__ = "articles"
    id        = Column(String(36), primary_key=True, index=True)
    ticker    = Column(String(10), index=True, nullable=False)
    title     = Column(Text, nullable=False)
    link      = Column(Text, nullable=False, unique=True)
    summary   = Column(Text)
    published = Column(DateTime, nullable=False)

Base.metadata.create_all(bind=engine)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Pydantic Schemas ──────────────────────────────────────────────────────────
class Holding(BaseModel):
    id: str
    ticker: str
    quantity: float
    avg_price: float

    # enable .from_orm() on Pydantic v2
    model_config = ConfigDict(from_attributes=True)

class HoldingCreate(BaseModel):
    ticker: str
    quantity: float

class Recommendation(BaseModel):
    ticker: str
    name: str
    sector: str
    similarity_score: float

class Quote(BaseModel):
    ticker: str
    name: str
    price: float
    change: float
    spark: List[float]

class Performer(BaseModel):
    ticker: str
    name: str
    sector: str
    change_percent: float

class NewsImpactItem(BaseModel):
    ticker: str
    headline: str
    link: str
    source: str
    delta: float
    impact: float
    sentiment: float
    spark: List[float]
    summary: str
    published: datetime.datetime

class StockItem(BaseModel):
    ticker: str
    name: str

class PortfolioNewsFlag(BaseModel):
    ticker: str
    has_news: bool


# ─── Load S&P 500 Metadata ──────────────────────────────────────────────────────
wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
df = pd.read_html(wiki_url, flavor="lxml")[0]
stock_keys = df["Symbol"].tolist()
stock_meta = {
    row.Symbol: {"name": row.Security, "sector": row["GICS Sector"]}
    for _, row in df.iterrows()
}


# ─── TF-IDF Setup for recommendations ──────────────────────────────────────────
tfidf_corpus     = [f"{stock_meta[t]['name']} {stock_meta[t]['sector']}" for t in stock_keys]
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix     = tfidf_vectorizer.fit_transform(tfidf_corpus)


# ─── Portfolio CRUD ────────────────────────────────────────────────────────────
@app.get("/portfolio", response_model=List[Holding])
def get_portfolio(db: Session = Depends(get_db)):
    holdings = db.query(HoldingModel).all()
    
    # Make sure all values are properly set to avoid null values
    validated_holdings = []
    for h in holdings:
        # Ensure ticker and other fields are properly set
        if not h.ticker or not isinstance(h.quantity, (int, float)) or not isinstance(h.avg_price, (int, float)):
            logging.warning(f"Invalid holding data detected: {h.id}")
            continue
            
        validated_holdings.append(Holding.from_orm(h))
    
    return validated_holdings

@app.post("/portfolio", response_model=Holding, status_code=201)
def add_holding(h: HoldingCreate, db: Session = Depends(get_db)):
    t = h.ticker.upper()
    if db.query(HoldingModel).filter_by(ticker=t).first():
        raise HTTPException(400, "Ticker already in portfolio")
    
    # Check if ticker exists in our stock metadata
    if t not in stock_keys and not t.startswith("^"):  # Allow indices that start with ^
        # Try to verify ticker exists by fetching data
        try:
            # Use our cached ticker data function
            yf_ticker = get_ticker_data(t, force_refresh=True)
            if yf_ticker is None or not yf_ticker.info or "regularMarketPrice" not in yf_ticker.info:
                raise HTTPException(400, f"Invalid ticker: {t}")
        except Exception as e:
            logging.warning(f"Error validating ticker {t}: {str(e)}")
            raise HTTPException(400, f"Could not validate ticker: {t}")
    
    # Get current price
    try:
        # Use our cached ticker data function
        yf_ticker = get_ticker_data(t)
        
        if yf_ticker is None:
            avg_price = 0.0
        else:
            # Try to get the latest price from info
            info = yf_ticker.info
            if info and "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
                avg_price = float(info["regularMarketPrice"])
            else:
                # Fall back to historical data
                hist = yf_ticker.history(period="1d")
                if hist is not None and not hist.empty:
                    avg_price = float(hist["Close"].iloc[-1])
                else:
                    avg_price = 0.0
    except Exception as e:
        logging.warning(f"Error fetching price for {t}: {str(e)}")
        avg_price = 0.0
    
    rec = HoldingModel(id=uuid.uuid4().hex, ticker=t, quantity=h.quantity, avg_price=avg_price)
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return Holding.from_orm(rec)

@app.put("/portfolio/{holding_id}", response_model=Holding)
def update_holding(holding_id: str, h: HoldingCreate, db: Session = Depends(get_db)):
    rec = db.query(HoldingModel).get(holding_id)
    if not rec:
        raise HTTPException(404, "Holding not found")
    
    t = h.ticker.upper()
    
    # Update price only if ticker changed
    if t != rec.ticker:
        try:
            # Use our cached ticker data function
            yf_ticker = get_ticker_data(t, force_refresh=True)
            
            if yf_ticker is None:
                avg_price = rec.avg_price
            else:
                # Try to get the latest price from info
                info = yf_ticker.info
                if info and "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
                    avg_price = float(info["regularMarketPrice"])
                else:
                    # Fall back to historical data
                    hist = yf_ticker.history(period="1d")
                    if hist is not None and not hist.empty:
                        avg_price = float(hist["Close"].iloc[-1])
                    else:
                        avg_price = rec.avg_price
        except Exception as e:
            logging.warning(f"Error fetching price for {t}: {str(e)}")
            avg_price = rec.avg_price
    else:
        avg_price = rec.avg_price
    
    rec.ticker = t
    rec.quantity = h.quantity
    rec.avg_price = avg_price
    
    db.commit()
    db.refresh(rec)
    return Holding.from_orm(rec)

@app.delete("/portfolio/{holding_id}", status_code=204)
def delete_holding(holding_id: str, db: Session = Depends(get_db)):
    db.query(HoldingModel).filter_by(id=holding_id).delete()
    db.commit()


# ─── Stocks list for autocomplete ──────────────────────────────────────────────
@app.get("/stocks", response_model=List[StockItem])
def list_stocks():
    return [{"ticker": tk, "name": stock_meta[tk]["name"]} for tk in stock_keys]


# ─── Caching and Rate Limiting ────────────────────────────────────────────────
# Cache for ticker data to reduce API calls
ticker_cache = {}
last_request_time = time.time() - 2  # Initialize with offset to allow immediate first request

def get_ticker_data(ticker_symbol, force_refresh=False):
    """Get ticker data with caching and rate limiting"""
    global last_request_time
    
    yf_sym = ticker_symbol.replace(".", "-")
    current_time = time.time()
    
    # Check cache first if not forcing refresh
    if not force_refresh and yf_sym in ticker_cache:
        cache_entry = ticker_cache[yf_sym]
        # Use cache if it's less than 15 minutes old
        if current_time - cache_entry["timestamp"] < 900:  # 15 minutes in seconds
            return cache_entry["data"]
    
    # Add delay between requests to avoid rate limiting
    time_since_last_request = current_time - last_request_time
    if time_since_last_request < 2:  # Ensure at least 2 seconds between requests
        delay = 2 - time_since_last_request + random.uniform(0.1, 0.5)  # Add some randomness
        time.sleep(delay)
    
    # Update last request time
    last_request_time = time.time()
    
    try:
        # Get ticker data
        ticker = yf.Ticker(yf_sym)
        
        # Store in cache
        ticker_cache[yf_sym] = {
            "data": ticker,
            "timestamp": time.time()
        }
        
        return ticker
    except Exception as e:
        logging.warning(f"Error fetching ticker data for {ticker_symbol}: {str(e)}")
        # Return cached data if available, even if expired
        if yf_sym in ticker_cache:
            logging.info(f"Using expired cache for {ticker_symbol}")
            return ticker_cache[yf_sym]["data"]
        return None

# ─── Quotes Endpoint ────────────────────────────────────────────────────────────
@app.get("/quotes", response_model=List[Quote])
def get_quotes(db: Session = Depends(get_db), force_refresh: bool = Query(False)):
    out = []
    holdings = db.query(HoldingModel).all()
    
    if not holdings:
        return []
    
    for h in holdings:
        try:
            # Get ticker with caching and rate limiting
            ticker = get_ticker_data(h.ticker, force_refresh)
            
            if ticker is None:
                # Use avg_price from database as fallback if no ticker data
                out.append(Quote(
                    ticker=h.ticker,
                    name=stock_meta.get(h.ticker, {}).get("name", h.ticker),
                    price=h.avg_price,
                    change=0.0,
                    spark=[h.avg_price] * 10  # Flat line as fallback
                ))
                continue
                
            # Try to get the latest price from info
            try:
                info = ticker.info
                if info and "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
                    price = float(info["regularMarketPrice"])
                    prev_close = info.get("regularMarketPreviousClose", h.avg_price)
                    change = round(price - prev_close, 2) if prev_close else 0.0
                    
                    # Get historical data for spark line
                    try:
                        hist = ticker.history(period="1d", interval="30m")
                        spark = hist["Close"].tolist() if not hist.empty else [price] * 10
                    except:
                        spark = [price] * 10
                    
                    out.append(Quote(
                        ticker=h.ticker,
                        name=stock_meta.get(h.ticker, {}).get("name", h.ticker),
                        price=round(price, 2),
                        change=change,
                        spark=spark
                    ))
                    continue
            except Exception as e:
                logging.warning(f"Error getting info for {h.ticker}: {str(e)}")
            
            # Fall back to historical data if info approach failed
            try:
                hist = ticker.history(period="2d")
                if hist is not None and not hist.empty and len(hist) >= 2:
                    cl = hist["Close"].iloc[-1]
                    op = hist["Open"].iloc[0] if len(hist) > 1 else h.avg_price
                    
                    # For spark line, try to get intraday data
                    try:
                        intraday = ticker.history(period="1d", interval="30m")
                        spark = intraday["Close"].tolist() if not intraday.empty else hist["Close"].tolist()
                    except:
                        spark = hist["Close"].tolist()
                    
                    out.append(Quote(
                        ticker=h.ticker,
                        name=stock_meta.get(h.ticker, {}).get("name", h.ticker),
                        price=round(float(cl), 2),
                        change=round(float(cl - op), 2),
                        spark=spark
                    ))
                    continue
            except Exception as e:
                logging.warning(f"Error getting history for {h.ticker}: {str(e)}")
            
            # If we got here, use the fallback
            out.append(Quote(
                ticker=h.ticker,
                name=stock_meta.get(h.ticker, {}).get("name", h.ticker),
                price=h.avg_price,
                change=0.0,
                spark=[h.avg_price] * 10  # Flat line as fallback
            ))
            
        except Exception as e:
            logging.warning(f"Unexpected error for {h.ticker}: {str(e)}")
            # Final fallback
            out.append(Quote(
                ticker=h.ticker,
                name=stock_meta.get(h.ticker, {}).get("name", h.ticker),
                price=h.avg_price,
                change=0.0,
                spark=[h.avg_price] * 10  # Flat line as fallback
            ))
    
    return out


# ─── Top Performers ────────────────────────────────────────────────────────────
@app.get("/top-performers", response_model=List[Performer])
def top_performers(force_refresh: bool = Query(False)):
    perf = []
    num_tickers_to_check = min(50, len(stock_keys))  # Reduced from 100 to limit API calls
    
    for tk in stock_keys[:num_tickers_to_check]:
        try:
            # Get ticker with caching and rate limiting
            ticker = get_ticker_data(tk, force_refresh)
            
            if ticker is None:
                continue
            
            # Try to get info first for the latest data
            try:
                info = ticker.info
                if info and "regularMarketPrice" in info and "regularMarketPreviousClose" in info:
                    current = info["regularMarketPrice"]
                    prev = info["regularMarketPreviousClose"]
                    
                    if current and prev and prev > 0:
                        pct = round((current - prev) / prev * 100, 2)
                        
                        perf.append({
                            "ticker": tk,
                            "name": stock_meta[tk]["name"],
                            "sector": stock_meta[tk]["sector"],
                            "change_percent": pct
                        })
                        continue
            except Exception as e:
                logging.warning(f"Error getting info for top performer {tk}: {str(e)}")
            
            # Fall back to history if info approach didn't work
            try:
                hist = ticker.history(period="2d")
                if hist is not None and not hist.empty and len(hist) >= 2:
                    today_close = hist["Close"].iloc[-1]
                    prev_close = hist["Close"].iloc[0]
                    
                    if prev_close > 0:
                        pct = round((today_close - prev_close) / prev_close * 100, 2)
                        
                        perf.append({
                            "ticker": tk,
                            "name": stock_meta[tk]["name"],
                            "sector": stock_meta[tk]["sector"],
                            "change_percent": pct
                        })
            except Exception as e:
                logging.warning(f"Error getting history for top performer {tk}: {str(e)}")
        except Exception as e:
            logging.warning(f"Error processing top performer {tk}: {str(e)}")
            continue
    
    if perf:
        # Sort by change_percent in descending order and take top 5
        return sorted(perf, key=lambda x: x["change_percent"], reverse=True)[:5]
    
    return []


# ─── TF-IDF Recommendations ────────────────────────────────────────────────────
@app.get("/recommendations", response_model=List[Recommendation])
def recommend(db: Session = Depends(get_db)):
    holdings = [h.ticker for h in db.query(HoldingModel).all()]
    if not holdings:
        # Return empty array instead of raising error
        return []
    
    # Filter holdings to only include those in stock_keys
    valid_holdings = [t for t in holdings if t in stock_keys]
    if not valid_holdings:
        return []
    
    idxs = [stock_keys.index(t) for t in valid_holdings]
    centroid = tfidf_matrix[idxs].mean(axis=0)
    centroid = centroid.A if hasattr(centroid, "A") else centroid.toarray()
    sims = cosine_similarity(centroid, tfidf_matrix).flatten()

    recs = []
    for i in sims.argsort()[::-1]:
        tk = stock_keys[i]
        if tk in holdings: continue
        m = stock_meta[tk]
        recs.append(Recommendation(
            ticker=tk, name=m["name"], sector=m["sector"],
            similarity_score=round(float(sims[i]),3)
        ))
        if len(recs) >= 5: break
    return recs


# ─── Advanced News-Impact Endpoint ─────────────────────────────────────────────
@app.get("/news-impact", response_model=List[NewsImpactItem])
def news_impact(
    background_tasks: BackgroundTasks,
    db: Session            = Depends(get_db),
    window: int            = Query(60, ge=1),
    min_sentiment: float   = Query(0.0, ge=0.0),
    recent_hours: int      = Query(24, ge=1),
    limit: int             = Query(20, ge=1, le=100),
):
    # launch crawler in background
    background_tasks.add_task(crawl.main)

    try:
        now    = datetime.datetime.utcnow()
        cutoff = now - datetime.timedelta(hours=recent_hours)

        # portfolio + recommendations
        holdings = db.query(HoldingModel).all()
        
        # Try to get recommendations, but handle empty portfolio gracefully
        try:
            recs = recommend(db)
            tickers = {h.ticker for h in holdings} | {r.ticker for r in recs}
        except HTTPException:
            # If portfolio is empty, just use holdings (which might also be empty)
            tickers = {h.ticker for h in holdings}
        
        # If still no tickers, use S&P 500 list
        if not tickers:
            tickers = set(stock_keys[:20])  # Use first 20 S&P stocks as fallback

        # compile regex once
        # Use a more robust pattern that looks for stock symbols
        if tickers:
            ticker_re = re.compile(r"\b(" + "|".join(re.escape(t) for t in tickers) + r")\b", re.IGNORECASE)
        else:
            # If no tickers in portfolio or recommendations, use S&P 500 stock list
            ticker_re = re.compile(r"\b(" + "|".join(re.escape(t) for t in stock_keys) + r")\b", re.IGNORECASE)

        seen   = set()
        scored = []

        for fn in glob.glob("data/raw_articles/rss_articles_*.json"):
            for art in json.loads(Path(fn).read_text()):
                link = art.get("link","")
                if not link or link in seen:
                    continue
                seen.add(link)

                # parse published timestamp
                pub = art.get("published","")
                try:
                    dt = datetime.datetime.fromisoformat(pub.replace("Z","+00:00"))
                    pub_dt = dt.replace(tzinfo=None)
                except:
                    continue
                if pub_dt < cutoff:
                    continue

                text = (art.get("title","")+" "+art.get("summary","")).upper()
                m = ticker_re.search(text)
                if not m:
                    continue
                tick = m.group(1).upper()

                # sentiment
                pol = TextBlob(text).sentiment.polarity
                if abs(pol) < min_sentiment:
                    continue

                # price window
                try:
                    df = yf.Ticker(tick.replace(".", "-")).history(
                        start=pub_dt - datetime.timedelta(minutes=30),
                        end=  pub_dt + datetime.timedelta(minutes=window+30),
                        interval="1m"
                    )
                    if df is None or df.empty:
                        continue
                    closes = df["Close"].tolist()
                except:
                    continue

                delta  = closes[min(window, len(closes)-1)] - closes[0]
                impact = round(abs(delta) * abs(pol), 4)

                # clean summary
                raw   = art.get("summary","") or ""
                clean = BeautifulSoup(raw, "lxml").get_text().strip()
                if clean == art.get("title","").strip():
                    clean = ""

                scored.append({
                    "ticker":    tick,
                    "headline":  art.get("title",""),
                    "link":      link,
                    "source":    art.get("source",""),
                    "delta":     delta,
                    "sentiment": round(pol,3),
                    "impact":    impact,
                    "spark":     closes,
                    "summary":   clean,
                    "published": pub_dt
                })

        top = sorted(scored, key=lambda x: x["impact"], reverse=True)[:limit]
        return top

    except Exception:
        logging.exception("Error in /news-impact")
        raise HTTPException(500, "News impact processing failed")
