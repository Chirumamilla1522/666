# # # #!/usr/bin/env python3
# # # import os, ssl, uuid, datetime, glob, logging, feedparser
# # # from pathlib import Path
# # # from typing    import List
# # # import yfinance as yf
# # # import pandas   as pd
# # # from fastapi    import FastAPI, HTTPException, Depends, Query
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from fastapi.responses      import FileResponse
# # # from pydantic               import BaseModel, ConfigDict
# # # from sqlalchemy             import create_engine, Column, String, Float, DateTime, Text
# # # from sqlalchemy.ext.declarative import declarative_base
# # # from sqlalchemy.orm         import sessionmaker, Session
# # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # from sklearn.metrics.pairwise        import cosine_similarity
# # # from textblob import TextBlob

# # # # ─── SSL & Logging ─────────────────────────────────────────────────────────────
# # # ssl._create_default_https_context = ssl._create_unverified_context
# # # logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# # # # ─── FastAPI setup ─────────────────────────────────────────────────────────────
# # # app = FastAPI(title="StockRadar Core API")
# # # app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# # # @app.get("/", include_in_schema=False)
# # # def serve_ui():
# # #     return FileResponse("index.html")

# # # # ─── Database (SQLite + SQLAlchemy) ─────────────────────────────────────────────
# # # DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stockradar.db")
# # # engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# # # SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
# # # Base         = declarative_base()

# # # class HoldingModel(Base):
# # #     __tablename__ = "portfolio"
# # #     id        = Column(String(36), primary_key=True, index=True)
# # #     ticker    = Column(String(10), nullable=False, unique=True, index=True)
# # #     quantity  = Column(Float, nullable=False)
# # #     avg_price = Column(Float, nullable=False)

# # # class ArticleModel(Base):
# # #     __tablename__ = "articles"
# # #     id        = Column(String(36), primary_key=True, index=True)
# # #     ticker    = Column(String(10), index=True, nullable=False)
# # #     title     = Column(Text, nullable=False)
# # #     link      = Column(Text, nullable=False, unique=True)
# # #     summary   = Column(Text)
# # #     published = Column(DateTime, nullable=False)

# # # Base.metadata.create_all(bind=engine)

# # # def get_db():
# # #     db = SessionLocal()
# # #     try:
# # #         yield db
# # #     finally:
# # #         db.close()

# # # # ─── Pydantic Schemas ──────────────────────────────────────────────────────────
# # # class Holding(BaseModel):
# # #     id: str; ticker: str; quantity: float; avg_price: float
# # #     model_config = ConfigDict(from_attributes=True)

# # # class HoldingCreate(BaseModel):
# # #     ticker: str; quantity: float

# # # class Recommendation(BaseModel):
# # #     ticker: str
# # #     name: str
# # #     sector: str
# # #     description: str
# # #     similarity_score: float
# # #     spark: List[float]
# # #     spark_times: List[datetime.datetime]
# # #     model_config = ConfigDict(from_attributes=True)


# # # class Quote(BaseModel):
# # #     ticker: str
# # #     name: str
# # #     price: float
# # #     change: float
# # #     spark: List[float]

# # # class Performer(BaseModel):
# # #     ticker: str
# # #     name: str
# # #     sector: str
# # #     change_percent: float
# # #     spark: List[float]

# # # class NewsImpactItem(BaseModel):
# # #     ticker: str
# # #     headline: str
# # #     link: str
# # #     source: str
# # #     delta: float
# # #     impact: float
# # #     spark: List[float]
# # #     summary: str
# # #     published: datetime.datetime

# # # class StockItem(BaseModel):
# # #     ticker: str
# # #     name: str

# # # class PortfolioNewsFlag(BaseModel):
# # #     ticker: str
# # #     has_news: bool

# # # SUMMARIES_DIR = Path("data/summaries")
# # # # ─── Load S&P 500 Metadata ──────────────────────────────────────────────────────
# # # wiki_url   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# # # df         = pd.read_html(wiki_url, flavor="lxml")[0]
# # # stock_keys = df["Symbol"].tolist()

# # # # ─── Load cached Wikipedia summaries ────────────────────────────────────────────
# # # SUMMARY_DIR     = "data/summaries"
# # # wiki_summaries  = {}
# # # for path in glob.glob(f"{SUMMARY_DIR}/*.txt"):
# # #     tk = Path(path).stem
# # #     with open(path, encoding="utf-8") as f:
# # #         wiki_summaries[tk] = f.read().strip()
# # #         # print(wiki_summaries[tk])

# # # # ─── Build stock_meta from df + cache ──────────────────────────────────────────
# # # stock_meta = {}
# # # for _, row in df.iterrows():
# # #     tk = row.Symbol
# # #     stock_meta[tk] = {
# # #         "name":        row.Security,
# # #         "sector":      row["GICS Sector"],
# # #         "description": wiki_summaries.get(tk, "")
# # #     }

# # # # ─── TF-IDF for recommendations ────────────────────────────────────────────────
# # # tfidf_corpus     = [
# # #     f"{stock_meta[t]['name']} {stock_meta[t]['sector']} {stock_meta[t]['description'][:500]}"
# # #     for t in stock_keys
# # # ]
# # # tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
# # # tfidf_matrix     = tfidf_vectorizer.fit_transform(tfidf_corpus)

# # # # import re
# # # # import yfinance as yf

# # # def fetch_and_store_ticker_news(ticker: str, db: Session):
# # #     """
# # #     Uses yfinance’s built-in Google Finance news fetcher.
# # #     """
# # #     yf_sym = ticker.replace(".", "-")
# # #     for art in yf.Ticker(yf_sym).news:
# # #         # art is a dict with keys: 'uuid','link','title','publisher','providerPublishTime','type'
# # #         link = art.get("link")
# # #         if not link or db.query(ArticleModel).filter_by(link=link).first():
# # #             continue

# # #         published = datetime.datetime.utcfromtimestamp(
# # #             art.get("providerPublishTime", datetime.datetime.utcnow().timestamp())
# # #         )

# # #         rec = ArticleModel(
# # #             id        = uuid.uuid4().hex,
# # #             ticker    = ticker,
# # #             title     = art.get("title","").strip(),
# # #             link      = link,
# # #             summary   = art.get("title","").strip(),   # no summary field here
# # #             published = published
# # #         )
# # #         db.add(rec)
# # #     db.commit()

# # # # ─── On startup, preload RSS news for existing holdings ─────────────────────────
# # # @app.on_event("startup")
# # # def load_portfolio_news():
# # #     db = SessionLocal()
# # #     try:
# # #         for tk in [h.ticker for h in db.query(HoldingModel).all()]:
# # #             print(tk)
# # #             fetch_and_store_ticker_news(tk, db)
# # #     finally:
# # #         db.close()

# # # def fetch_spark_for(ticker: str, period: str = "1d", interval: str = "1m"):
# # #     """
# # #     Returns (prices: List[float], times: List[datetime]) for the given ticker.
# # #     """
# # #     yf_sym = ticker.replace(".", "-")
# # #     df = yf.Ticker(yf_sym).history(period=period, interval=interval)
# # #     if df is None or df.empty:
# # #         return [], []
# # #     # list of closes
# # #     prices = df["Close"].tolist()
# # #     # list of actual timestamps
# # #     times  = df.index.to_pydatetime().tolist()
# # #     return prices, times

# # # # ─── Portfolio CRUD ────────────────────────────────────────────────────────────
# # # @app.get("/portfolio", response_model=List[Holding])
# # # def list_portfolio(db: Session = Depends(get_db)):
# # #     return [Holding.from_orm(h) for h in db.query(HoldingModel).all()]

# # # @app.post("/portfolio", response_model=Holding, status_code=201)
# # # def add_holding(h: HoldingCreate, db: Session = Depends(get_db)):
# # #     t = h.ticker.upper()
# # #     if db.query(HoldingModel).filter_by(ticker=t).first():
# # #         raise HTTPException(400, "Ticker already in portfolio")
# # #     try:
# # #         hist = yf.Ticker(t.replace(".", "-")).history(period="1d")
# # #         avg_price = float(hist["Close"].iloc[-1])
# # #     except:
# # #         avg_price = 0.0
# # #     new = HoldingModel(
# # #         id        = uuid.uuid4().hex,
# # #         ticker    = t,
# # #         quantity  = h.quantity,
# # #         avg_price = avg_price
# # #     )
# # #     db.add(new); db.commit(); db.refresh(new)
# # #     # fetch news immediately
# # #     fetch_and_store_ticker_news(t, db)
# # #     return Holding.from_orm(new)

# # # @app.put("/portfolio/{holding_id}", response_model=Holding)
# # # def update_holding(holding_id: str, h: HoldingCreate, db: Session = Depends(get_db)):
# # #     rec = db.query(HoldingModel).get(holding_id)
# # #     if not rec:
# # #         raise HTTPException(404, "Holding not found")
# # #     t = h.ticker.upper()
# # #     try:
# # #         hist = yf.Ticker(t.replace(".", "-")).history(period="1d")
# # #         avg_price = float(hist["Close"].iloc[-1])
# # #     except:
# # #         avg_price = rec.avg_price
# # #     rec.ticker, rec.quantity, rec.avg_price = t, h.quantity, avg_price
# # #     db.commit(); db.refresh(rec)
# # #     return Holding.from_orm(rec)

# # # @app.delete("/portfolio/{holding_id}", status_code=204)
# # # def delete_holding(holding_id: str, db: Session = Depends(get_db)):
# # #     db.query(HoldingModel).filter_by(id=holding_id).delete()
# # #     db.commit()

# # # # ─── List all tickers & names ──────────────────────────────────────────────────
# # # @app.get("/stocks", response_model=List[StockItem])
# # # def list_stocks():
# # #     return [{"ticker": tk, "name": stock_meta[tk]["name"]} for tk in stock_keys]

# # # def fetch_spark_for(ticker: str, period="1d", interval="1m"):
# # #     yf_sym = ticker.replace(".", "-")
# # #     df     = yf.Ticker(yf_sym).history(period=period, interval=interval)
# # #     if df is None or df.empty:
# # #         return [], []
# # #     return df["Close"].tolist(), df.index.to_pydatetime().tolist()

# # # @app.get("/recommendations", response_model=List[Recommendation])
# # # def recommend(db: Session = Depends(get_db)):
# # #     holdings = [h.ticker for h in db.query(HoldingModel).all()]
# # #     if not holdings:
# # #         raise HTTPException(400, "Portfolio is empty")

# # #     idxs = [stock_keys.index(t) for t in holdings if t in stock_keys]
# # #     if not idxs:
# # #         raise HTTPException(400, "No metadata for portfolio tickers")

# # #     centroid = tfidf_matrix[idxs].mean(axis=0)
# # #     centroid = centroid.A if hasattr(centroid, "A") else centroid.toarray()
# # #     sims     = cosine_similarity(centroid, tfidf_matrix).flatten()

# # #     recs = []
# # #     for i in sims.argsort()[::-1]:
# # #         tk = stock_keys[i]
# # #         if tk in holdings: 
# # #             continue

# # #         meta = stock_meta[tk]

# # #         # 1) load spark data as before
# # #         spark, times = fetch_spark_for(tk, period="1d")

# # #         # 2) try to read summaries/{TK}.txt
# # #         desc_file = SUMMARIES_DIR / f"{tk}.txt"
# # #         if desc_file.exists():
# # #             description = desc_file.read_text(encoding="utf-8").strip()
# # #         else:
# # #             # fallback to your wiki_intro you already populated
# # #             description = meta.get("wiki_intro","").strip()

# # #         recs.append(Recommendation(
# # #             ticker=tk,
# # #             name=meta["name"],
# # #             sector=meta["sector"],
# # #             description=description,
# # #             similarity_score=round(float(sims[i]), 3),
# # #             spark=spark,
# # #             spark_times=times
# # #         ))
# # #         if len(recs) >= 25:
# # #             break
# # #         # print(recs[-1].description)
# # #     return recs

# # # @app.get("/quotes", response_model=List[Quote])
# # # def get_quotes(
# # #     ticker: str = Query(None),
# # #     period: str = Query("1d"),
# # #     interval: str = Query("1m"),
# # #     db: Session = Depends(get_db)
# # # ):
# # #     symbols = [ticker] if ticker else [h.ticker for h in db.query(HoldingModel).all()]
# # #     out = []
# # #     for t in symbols:
# # #         sym = t.replace('.', '-')
# # #         df  = yf.Ticker(sym).history(period=period, interval=interval)
# # #         if df is None or df.empty: continue
# # #         opens, closes = df["Open"].tolist(), df["Close"].tolist()
# # #         delta = round(closes[-1] - opens[0], 2)
# # #         out.append({
# # #             "ticker": t,
# # #             "name": stock_meta[t]["name"],
# # #             "price": round(closes[-1], 2),
# # #             "change": delta,
# # #             "spark": closes,
# # #             "timestamps": [int(ts.timestamp()) for ts in df.index.to_pydatetime()]
# # #         })
# # #     if not out:
# # #         raise HTTPException(404, "No quote data")
# # #     return out

# # # from fastapi import Query

# # # @app.get("/portfolio-history", response_model=List[dict])
# # # def portfolio_history(
# # #     period: str = Query("1d", description="history period: 1d, 5d, 1mo, etc"),
# # #     interval: str = Query("1m", description="data interval: 1m, 5m, 60m, etc"),
# # #     db: Session = Depends(get_db),
# # # ):
# # #     # fetch all holdings
# # #     holdings = db.query(HoldingModel).all()
# # #     # accumulate per-minute (or per-interval) total value
# # #     # build a time‐indexed dict of sums
# # #     ts_map = {}
# # #     for h in holdings:
# # #         sym = h.ticker.replace(".", "-")
# # #         df = yf.Ticker(sym).history(period=period, interval=interval)
# # #         if df is None or df.empty:
# # #             continue
# # #         # multiply each close by quantity
# # #         for idx, row in df.iterrows():
# # #             ts = idx.isoformat()
# # #             ts_map.setdefault(ts, 0)
# # #             ts_map[ts] += row["Close"] * h.quantity
# # #     # sort by timestamp
# # #     series = [{"time": t, "value": round(v, 2)} for t, v in sorted(ts_map.items())]
# # #     return series

# # # # ─── Top Performers ────────────────────────────────────────────────────────────
# # # @app.get("/top-performers", response_model=List[Performer])
# # # def top_performers(
# # #     period: str = Query("1d", description="history period: 1d, 5d, 1mo, etc"),interval: str = Query("1m", description="data interval: 1m, 5m, 15m, etc")
# # # ):
# # #     perf = []
# # #     for tk in stock_keys:
# # #         sym = tk.replace(".", "-")
# # #         try:
# # #             df = yf.Ticker(sym).history(period=period, interval=interval)
# # #             if df is None or df.empty:
# # #                 continue
# # #             o, c = df["Open"].iloc[0], df["Close"].iloc[-1]
# # #             pct  = round((c - o) / o * 100, 2)
# # #             perf.append(Performer(
# # #                 ticker=tk,
# # #                 name=stock_meta[tk]["name"],
# # #                 sector=stock_meta[tk]["sector"],
# # #                 change_percent=pct,
# # #                 spark=df["Close"].tolist(),
# # #                 spark_times=df.index.to_pydatetime().tolist(),
# # #             ))
# # #         except:
# # #             continue
# # #     if not perf:
# # #         raise HTTPException(404, "No performance data")
# # #     return sorted(perf, key=lambda x: x.change_percent, reverse=True)[:10]

# # # @app.get("/news-impact", response_model=List[NewsImpactItem])
# # # def news_impact(
# # #     db: Session            = Depends(get_db),
# # #     window: int            = Query(30, ge=1),
# # #     min_sentiment: float   = Query(0.0, ge=0.0),
# # #     recent_hours: int      = Query(24, ge=1),
# # #     limit: int             = Query(20, ge=1, le=100),
# # # ):
# # #     # ─── 1) First, fetch & store the freshest headlines for each holding
# # #     holdings = db.query(HoldingModel).all()
# # #     for h in holdings:
# # #         print(h.ticker)
# # #         fetch_and_store_ticker_news(h.ticker, db)

# # #     # ─── 2) Then define the cutoff window & load from your ArticleModel table
# # #     cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=recent_hours)
# # #     raw = (
# # #         db.query(ArticleModel)
# # #           .filter(ArticleModel.published >= cutoff)
# # #           .order_by(ArticleModel.published.desc())
# # #           .all()
# # #     )

# # #     # ─── 3) Score them exactly as before
# # #     scored = []
# # #     for art in raw:
# # #         text = (art.title + " " + (art.summary or "")).upper()
# # #         pol  = TextBlob(text).sentiment.polarity
# # #         if abs(pol) < min_sentiment:
# # #             continue

# # #         sym   = art.ticker.replace(".", "-")
# # #         start = art.published - datetime.timedelta(minutes=5)
# # #         end   = art.published + datetime.timedelta(minutes=window+5)
# # #         try:
# # #             df = yf.Ticker(sym).history(start=start, end=end, interval="1m")
# # #             if df is None or df.empty:
# # #                 continue
# # #             closes = df["Close"].tolist()
# # #             before, after = closes[0], closes[min(window, len(closes)-1)]
# # #             delta = round(after - before, 2)
# # #         except:
# # #             continue

# # #         impact = round(abs(delta) * abs(pol), 4)
# # #         scored.append({
# # #             "ticker":    art.ticker,
# # #             "headline":  art.title,
# # #             "link":      art.link,
# # #             "source":    "Google News",
# # #             "delta":     delta,
# # #             "impact":    impact,
# # #             "spark":     closes,
# # #             "summary":   art.summary or "",
# # #             "published": art.published
# # #         })

# # #     # ─── 4) Return the top N by impact
# # #     return sorted(scored, key=lambda x: x["impact"], reverse=True)[:limit]


# # # # ─── Portfolio-News Flag ───────────────────────────────────────────────────────
# # # @app.get("/portfolio-news", response_model=List[PortfolioNewsFlag])
# # # def portfolio_news(
# # #     db: Session            = Depends(get_db),
# # #     recent_hours: int      = Query(24, ge=1),
# # # ):
# # #     cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=recent_hours)
# # #     holdings = [h.ticker for h in db.query(HoldingModel).all()]
# # #     seen = {
# # #         art.ticker
# # #         for art in db.query(ArticleModel)
# # #                      .filter(ArticleModel.ticker.in_(holdings))
# # #                      .filter(ArticleModel.published >= cutoff)
# # #                      .all()
# # #     }
# # #     return [
# # #         PortfolioNewsFlag(ticker=t, has_news=(t in seen))
# # #         for t in holdings
# # #     ]

# # #!/usr/bin/env python3
# # import os, ssl, uuid, datetime, glob, logging
# # from pathlib import Path
# # from typing    import List
# # import feedparser
# # import yfinance as yf
# # import pandas   as pd
# # from fastapi    import FastAPI, HTTPException, Depends, Query, BackgroundTasks
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses      import FileResponse
# # from pydantic               import BaseModel, ConfigDict
# # from sqlalchemy             import create_engine, Column, String, Float, DateTime, Text
# # from sqlalchemy.ext.declarative import declarative_base
# # from sqlalchemy.orm         import sessionmaker, Session
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise        import cosine_similarity
# # from textblob import TextBlob

# # # ─── SSL & Logging ─────────────────────────────────────────────────────────────
# # ssl._create_default_https_context = ssl._create_unverified_context
# # logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# # # ─── App & CORS ────────────────────────────────────────────────────────────────
# # app = FastAPI(title="StockRadar Core API")
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
# # )

# # @app.get("/", include_in_schema=False)
# # def serve_ui():
# #     return FileResponse("index.html")


# # # ─── Database setup ─────────────────────────────────────────────────────────────
# # DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stockradar.db")
# # engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# # SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
# # Base         = declarative_base()

# # class HoldingModel(Base):
# #     __tablename__ = "portfolio"
# #     id        = Column(String(36), primary_key=True, index=True)
# #     ticker    = Column(String(10), nullable=False, unique=True, index=True)
# #     quantity  = Column(Float, nullable=False)
# #     avg_price = Column(Float, nullable=False)

# # class ArticleModel(Base):
# #     __tablename__ = "articles"
# #     id        = Column(String(36), primary_key=True, index=True)
# #     ticker    = Column(String(10), index=True, nullable=False)
# #     title     = Column(Text, nullable=False)
# #     link      = Column(Text, nullable=False, unique=True)
# #     summary   = Column(Text)
# #     published = Column(DateTime, nullable=False)

# # Base.metadata.create_all(bind=engine)

# # def get_db():
# #     db = SessionLocal()
# #     try:
# #         yield db
# #     finally:
# #         db.close()


# # # ─── Pydantic Schemas ──────────────────────────────────────────────────────────
# # class Holding(BaseModel):
# #     id: str
# #     ticker: str
# #     quantity: float
# #     avg_price: float
# #     model_config = ConfigDict(from_attributes=True)

# # class HoldingCreate(BaseModel):
# #     ticker: str
# #     quantity: float

# # class Recommendation(BaseModel):
# #     ticker: str
# #     name: str
# #     sector: str
# #     description: str
# #     similarity_score: float
# #     spark: List[float]
# #     spark_times: List[datetime.datetime]
# #     model_config = ConfigDict(from_attributes=True)

# # class Quote(BaseModel):
# #     ticker: str
# #     name: str
# #     price: float
# #     change: float
# #     spark: List[float]

# # class Performer(BaseModel):
# #     ticker: str
# #     name: str
# #     sector: str
# #     change_percent: float
# #     spark: List[float]

# # class NewsImpactItem(BaseModel):
# #     ticker: str
# #     headline: str
# #     link: str
# #     source: str
# #     delta: float
# #     impact: float
# #     spark: List[float]
# #     summary: str
# #     published: datetime.datetime

# # class StockItem(BaseModel):
# #     ticker: str
# #     name: str

# # class PortfolioNewsFlag(BaseModel):
# #     ticker: str
# #     has_news: bool


# # # ─── Load S&P 500 metadata & summaries ────────────────────────────────────────
# # wiki_url   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# # df         = pd.read_html(wiki_url, flavor="lxml")[0]
# # stock_keys = df["Symbol"].tolist()

# # # cached Wikipedia summaries in data/summaries/*.txt
# # SUMMARIES_DIR = Path("data/summaries")
# # wiki_summaries = {}
# # for txt in SUMMARIES_DIR.glob("*.txt"):
# #     wiki_summaries[txt.stem] = txt.read_text(encoding="utf-8").strip()

# # stock_meta = {}
# # for _, row in df.iterrows():
# #     tk = row.Symbol
# #     stock_meta[tk] = {
# #         "name":        row.Security,
# #         "sector":      row["GICS Sector"],
# #         "description": wiki_summaries.get(tk, "")
# #     }

# # tfidf_corpus     = [
# #     f"{stock_meta[t]['name']} {stock_meta[t]['sector']} {stock_meta[t]['description'][:500]}"
# #     for t in stock_keys
# # ]
# # tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
# # tfidf_matrix     = tfidf_vectorizer.fit_transform(tfidf_corpus)


# # from urllib.parse import quote_plus
# # import feedparser
# # import requests

# # # def fetch_and_store_ticker_news(ticker: str, db: Session):
# # #     """
# # #     Fetch latest news for `ticker` from multiple sources (yfinance, Google RSS, NewsAPI)
# # #     dedupe by URL, and insert into ArticleModel.
# # #     """
# # #     # 1) gather all links we already have for this ticker
# # #     seen_links = {
# # #         link for (link,) in db.query(ArticleModel.link)
# # #                             .filter(ArticleModel.ticker == ticker)
# # #                             .all()
# # #     }

# # #     # —————————————————————————————————————————————————————
# # #     # Source A: Google News RSS (exact company name)
# # #     # —————————————————————————————————————————————————————
# # #     company = stock_meta[ticker]["name"]
# # #     q       = quote_plus(f'"{company}"')
# # #     rss_url = (
# # #         f"https://news.google.com/rss/search"
# # #         f"?q={q}&hl=en-US&gl=US&ceid=US:en"
# # #     )
# # #     feed = feedparser.parse(rss_url)
# # #     for entry in feed.entries:
# # #         link    = entry.get("link","").strip()
# # #         if not link or link in seen_links:
# # #             continue

# # #         title   = entry.get("title","").strip()
# # #         summary = entry.get("summary","").strip()
# # #         try:
# # #             published = datetime.datetime(*entry.published_parsed[:6])
# # #         except:
# # #             published = datetime.datetime.utcnow()

# # #         db.add(ArticleModel(
# # #             id        = uuid.uuid4().hex,
# # #             ticker    = ticker,
# # #             title     = title,
# # #             link      = link,
# # #             summary   = summary,
# # #             published = published
# # #         ))
# # #         seen_links.add(link)
        
# # #         # —————————————————————————————————————————————————————
# # #         # Source B: yfinance.news
# # #         # —————————————————————————————————————————————————————
# # #     yf_sym = ticker.replace(".", "-")
# # #     for art in yf.Ticker(yf_sym).news:
# # #         # yfinance puts most metadata under "content"
# # #         content = art.get("content", {}) or {}

# # #         # try the canonicalUrl first, then clickThroughUrl, then top‐level link
# # #         link = (
# # #             content.get("canonicalUrl", {}).get("url")
# # #             or content.get("clickThroughUrl", {}).get("url")
# # #             or art.get("link")
# # #         )
# # #         if not link or link in seen_links:
# # #             continue

# # #         title   = art.get("title", "").strip()
# # #         # sometimes yfinance gives a "summary" field, otherwise fall back to title
# # #         summary = art.get("summary") or title

# # #         # published time: yfinance gives providerPublishTime as a UNIX timestamp
# # #         tstamp = art.get("providerPublishTime")
# # #         if tstamp:
# # #             published = datetime.datetime.utcfromtimestamp(tstamp)
# # #         else:
# # #             published = datetime.datetime.utcnow()

# # #         db.add(ArticleModel(
# # #             id        = uuid.uuid4().hex,
# # #             ticker    = ticker,
# # #             title     = title,
# # #             link      = link,
# # #             summary   = summary,
# # #             published = published
# # #         ))
# # #         seen_links.add(link)

# # #     # —————————————————————————————————————————————————————
# # #     # Source C: NewsAPI.org (optional; requires NEWSAPI_KEY env var)
# # #     # —————————————————————————————————————————————————————
# # #     # key = os.getenv("NEWSAPI_KEY")
# # #     # if key:
# # #     #     resp = requests.get(
# # #     #         "https://newsapi.org/v2/everything",
# # #     #         params={
# # #     #             "q": ticker,
# # #     #             "language": "en",
# # #     #             "sortBy": "publishedAt",
# # #     #             "apiKey": key
# # #     #         },
# # #     #         timeout=5
# # #     #     ).json()
# # #     #     for art in resp.get("articles", []):
# # #     #         link = art["url"]
# # #     #         if not link or link in seen_links:
# # #     #             continue
# # #     #         try:
# # #     #             published = datetime.datetime.fromisoformat(
# # #     #                 art["publishedAt"].replace("Z", "+00:00")
# # #     #             )
# # #     #         except:
# # #     #             published = datetime.datetime.utcnow()
# # #     #         db.add(ArticleModel(
# # #     #             id        = uuid.uuid4().hex,
# # #     #             ticker    = ticker,
# # #     #             title     = art.get("title","").strip(),
# # #     #             link      = link,
# # #     #             summary   = art.get("description","").strip() or art.get("title","").strip(),
# # #     #             published = published
# # #     #         ))
# # #     #         seen_links.add(link)

# # #     db.commit()

# # 


# # # ─── Portfolio endpoints ──────────────────────────────────────────────────────
# # @app.get("/portfolio", response_model=List[Holding])
# # def list_portfolio(db: Session = Depends(get_db)):
# #     return [Holding.from_orm(h) for h in db.query(HoldingModel).all()]

# # @app.post("/portfolio", response_model=Holding, status_code=201)
# # def add_holding(
# #     h: HoldingCreate,
# #     background_tasks: BackgroundTasks,
# #     db: Session = Depends(get_db)
# # ):
# #     t = h.ticker.upper()
# #     if db.query(HoldingModel).filter_by(ticker=t).first():
# #         raise HTTPException(400, "Ticker already in portfolio")
# #     try:
# #         hist = yf.Ticker(t.replace(".", "-")).history(period="1d")
# #         avg_price = float(hist["Close"].iloc[-1])
# #     except:
# #         avg_price = 0.0

# #     new = HoldingModel(
# #         id        = uuid.uuid4().hex,
# #         ticker    = t,
# #         quantity  = h.quantity,
# #         avg_price = avg_price
# #     )
# #     db.add(new)
# #     db.commit()
# #     db.refresh(new)

# #     # kick off background news fetch
# #     background_tasks.add_task(fetch_and_store_ticker_news, t)

# #     return Holding.from_orm(new)

# # @app.put("/portfolio/{holding_id}", response_model=Holding)
# # def update_holding(holding_id: str, h: HoldingCreate, db: Session = Depends(get_db)):
# #     rec = db.query(HoldingModel).get(holding_id)
# #     if not rec:
# #         raise HTTPException(404, "Holding not found")
# #     t = h.ticker.upper()
# #     try:
# #         hist = yf.Ticker(t.replace(".", "-")).history(period="1d")
# #         avg_price = float(hist["Close"].iloc[-1])
# #     except:
# #         avg_price = rec.avg_price
# #     rec.ticker, rec.quantity, rec.avg_price = t, h.quantity, avg_price
# #     db.commit()
# #     db.refresh(rec)
# #     return Holding.from_orm(rec)

# # @app.delete("/portfolio/{holding_id}", status_code=204)
# # def delete_holding(holding_id: str, db: Session = Depends(get_db)):
# #     db.query(HoldingModel).filter_by(id=holding_id).delete()
# #     db.commit()


# # # ─── Stocks list for autocomplete ─────────────────────────────────────────────
# # @app.get("/stocks", response_model=List[StockItem])
# # def list_stocks():
# #     return [{"ticker": tk, "name": stock_meta[tk]["name"]} for tk in stock_keys]


# # # ─── Recommendations ───────────────────────────────────────────────────────────
# # def fetch_spark_for(ticker: str, period="1d", interval="1m"):
# #     sym = ticker.replace(".", "-")
# #     df  = yf.Ticker(sym).history(period=period, interval=interval)
# #     if df is None or df.empty: return [], []
# #     return df["Close"].tolist(), df.index.to_pydatetime().tolist()

# # # @app.get("/recommendations", response_model=List[Recommendation])
# # # def recommend(db: Session = Depends(get_db)):
# # #     holdings = [h.ticker for h in db.query(HoldingModel).all()]
# # #     if not holdings:
# # #         raise HTTPException(400, "Portfolio is empty")
# # #     idxs = [stock_keys.index(t) for t in holdings if t in stock_keys]
# # #     if not idxs:
# # #         raise HTTPException(400, "No metadata for portfolio tickers")

# # #     centroid = tfidf_matrix[idxs].mean(axis=0)
# # #     centroid = getattr(centroid, "A", centroid.toarray())
# # #     sims     = cosine_similarity(centroid, tfidf_matrix).flatten()

# # #     recs = []
# # #     for i in sims.argsort()[::-1]:
# # #         tk = stock_keys[i]
# # #         if tk in holdings:
# # #             continue
# # #         spark, times = fetch_spark_for(tk, period="1d")
# # #         desc_file = SUMMARIES_DIR / f"{tk}.txt"
# # #         description = desc_file.read_text(encoding="utf-8").strip() \
# # #                       if desc_file.exists() else stock_meta[tk]["description"]
# # #         recs.append(Recommendation(
# # #             ticker           = tk,
# # #             name             = stock_meta[tk]["name"],
# # #             sector           = stock_meta[tk]["sector"],
# # #             description      = description,
# # #             similarity_score = round(float(sims[i]), 3),
# # #             spark            = spark,
# # #             spark_times      = times
# # #         ))
# # #         if len(recs) >= 25:
# # #             break
# # #     return recs

# # from scipy import sparse
# # import numpy as np

# # @app.get("/recommendations", response_model=List[Recommendation])
# # def recommend(db: Session = Depends(get_db)):
# #     holdings = [h.ticker for h in db.query(HoldingModel).all()]
# #     if not holdings:
# #         raise HTTPException(400, "Portfolio is empty")

# #     idxs = [stock_keys.index(t) for t in holdings if t in stock_keys]
# #     if not idxs:
# #         raise HTTPException(400, "No metadata for portfolio tickers")

# #     # compute the centroid of your TF-IDF rows
# #     centroid = tfidf_matrix[idxs].mean(axis=0)   # could be sparse, matrix, or ndarray

# #     # safely convert to a flat numpy array:
# #     if sparse.issparse(centroid):
# #         centroid = centroid.toarray().ravel()
# #     elif hasattr(centroid, "A"):  # numpy.matrix
# #         centroid = centroid.A.ravel()
# #     else:
# #         centroid = np.asarray(centroid).ravel()

# #     sims = cosine_similarity(centroid.reshape(1, -1), tfidf_matrix).flatten()

# #     recs = []
# #     for i in sims.argsort()[::-1]:
# #         tk = stock_keys[i]
# #         if tk in holdings:
# #             continue

# #         meta = stock_meta[tk]
# #         spark, times = fetch_spark_for(tk, period="1d")
# #         desc_file = SUMMARIES_DIR / f"{tk}.txt"
# #         description = (
# #             desc_file.read_text(encoding="utf-8").strip()
# #             if desc_file.exists()
# #             else meta.get("description", "")
# #         )

# #         recs.append(Recommendation(
# #             ticker           = tk,
# #             name             = meta["name"],
# #             sector           = meta["sector"],
# #             description      = description,
# #             similarity_score = round(float(sims[i]), 3),
# #             spark            = spark,
# #             spark_times      = times
# #         ))
# #         if len(recs) >= 25:
# #             break

# #     return recs


# # # ─── Quotes ────────────────────────────────────────────────────────────────────
# # @app.get("/quotes", response_model=List[Quote])
# # def get_quotes(
# #     ticker: str = Query(None),
# #     period: str = Query("1d"),
# #     interval: str = Query("1m"),
# #     db: Session = Depends(get_db)
# # ):
# #     symbols = [ticker] if ticker else [h.ticker for h in db.query(HoldingModel).all()]
# #     out = []
# #     for t in symbols:
# #         sym = t.replace(".", "-")
# #         df  = yf.Ticker(sym).history(period=period, interval=interval)
# #         if df is None or df.empty: continue
# #         opens, closes = df["Open"].tolist(), df["Close"].tolist()
# #         delta = round(closes[-1] - opens[0], 2)
# #         out.append({
# #             "ticker": t,
# #             "name":   stock_meta[t]["name"],
# #             "price":  round(closes[-1], 2),
# #             "change": delta,
# #             "spark":  closes
# #         })
# #     if not out:
# #         raise HTTPException(404, "No quote data")
# #     return out


# # # ─── Portfolio History ─────────────────────────────────────────────────────────
# # @app.get("/portfolio-history", response_model=List[dict])
# # def portfolio_history(
# #     period: str = Query("1d", description="1d, 5d, 1mo, etc"),
# #     interval: str = Query("1m", description="1m, 5m, 60m, etc"),
# #     db: Session = Depends(get_db),
# # ):
# #     ts_map = {}
# #     for h in db.query(HoldingModel).all():
# #         sym = h.ticker.replace(".", "-")
# #         df  = yf.Ticker(sym).history(period=period, interval=interval)
# #         if df is None or df.empty:
# #             continue
# #         for idx, row in df.iterrows():
# #             ts = idx.isoformat()
# #             ts_map.setdefault(ts, 0)
# #             ts_map[ts] += row["Close"] * h.quantity

# #     series = [{"time": t, "value": round(v,2)} for t,v in sorted(ts_map.items())]
# #     return series


# # # ─── Top Performers ────────────────────────────────────────────────────────────
# # @app.get("/top-performers", response_model=List[Performer])
# # def top_performers(
# #     period: str = Query("1d", description="1d, 5d, 1mo, etc"),
# #     interval: str = Query("1m", description="1m, 5m, 15m, etc")
# # ):
# #     perf = []
# #     for tk in stock_keys:
# #         sym = tk.replace(".", "-")
# #         df  = yf.Ticker(sym).history(period=period, interval=interval)
# #         if df is None or df.empty:
# #             continue
# #         o, c = df["Open"].iloc[0], df["Close"].iloc[-1]
# #         pct  = round((c - o) / o * 100, 2)
# #         perf.append(Performer(
# #             ticker         = tk,
# #             name           = stock_meta[tk]["name"],
# #             sector         = stock_meta[tk]["sector"],
# #             change_percent = pct,
# #             spark          = df["Close"].tolist()
# #         ))
# #     if not perf:
# #         raise HTTPException(404, "No performance data")
# #     return sorted(perf, key=lambda x: x.change_percent, reverse=True)[:10]


# # from fastapi import Query
# # from sqlalchemy.orm import Session
# # from textblob import TextBlob
# # import datetime
# # import yfinance as yf

# # # @app.get("/news-impact", response_model=List[NewsImpactItem])
# # # def news_impact(
# # #     db: Session            = Depends(get_db),
# # #     window: int            = Query(30, ge=1),
# # #     min_sentiment: float   = Query(0.0, ge=0.0),
# # #     recent_hours: int      = Query(24, ge=1),
# # #     limit: int             = Query(20, ge=1, le=100),
# # # ):
# # #     # 1) Fetch & store fresh headlines only for current holdings
# # #     holdings = [h.ticker for h in db.query(HoldingModel).all()]
# # #     for tk in holdings:
# # #         fetch_and_store_ticker_news(tk, db)

# # #     # 2) Define cutoff time
# # #     cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=recent_hours)

# # #     # 3) Only query articles still in the portfolio and within the time window
# # #     raw = (
# # #         db.query(ArticleModel)
# # #           .filter(ArticleModel.ticker.in_(holdings))
# # #           .filter(ArticleModel.published >= cutoff)
# # #           .order_by(ArticleModel.published.desc())
# # #           .all()
# # #     )

# # #     # 4) Score each article by sentiment×price move and collect top N
# # #     scored = []
# # #     for art in raw:
# # #         # compute polarity
# # #         text = (art.title + " " + (art.summary or "")).upper()
# # #         pol  = TextBlob(text).sentiment.polarity
# # #         if abs(pol) < min_sentiment:
# # #             continue

# # #         # fetch pre/post closes around the publish time
# # #         sym   = art.ticker.replace(".", "-")
# # #         start = art.published - datetime.timedelta(minutes=5)
# # #         end   = art.published + datetime.timedelta(minutes=window + 5)
# # #         try:
# # #             df = yf.Ticker(sym).history(start=start, end=end, interval="1m")
# # #             if df is None or df.empty:
# # #                 continue
# # #             closes = df["Close"].tolist()
# # #             before, after = closes[0], closes[min(window, len(closes)-1)]
# # #             delta = round(after - before, 2)
# # #         except Exception:
# # #             continue

# # #         impact = round(abs(delta) * abs(pol), 4)
# # #         scored.append({
# # #             "ticker":    art.ticker,
# # #             "headline":  art.title,
# # #             "link":      art.link,
# # #             "source":    "Google News",
# # #             "delta":     delta,
# # #             "impact":    impact,
# # #             "spark":     closes,
# # #             "summary":   art.summary or "",
# # #             "published": art.published
# # #         })

# # #     # 5) Return the top `limit` articles, sorted by impact descending
# # #     return sorted(scored, key=lambda x: x["impact"], reverse=True)[:limit]

# # from fastapi import Depends, Query, HTTPException
# # from sqlalchemy.orm import Session
# # from textblob import TextBlob
# # import datetime
# # import yfinance as yf

# # from fastapi import HTTPException, Depends, Query
# # from sqlalchemy.orm import Session
# # from textblob import TextBlob
# # import datetime
# # import yfinance as yf

# # @app.get("/news-impact", response_model=List[NewsImpactItem])
# # def news_impact(
# #     db: Session            = Depends(get_db),
# #     window: int            = Query(30, ge=1),
# #     min_sentiment: float   = Query(0.0, ge=0.0),
# #     recent_hours: int      = Query(24, ge=1),
# #     limit: int             = Query(100, ge=1, le=5000),
# # ):
# #     # 1) Fetch fresh headlines for every current holding
# #     holdings = [h.ticker for h in db.query(HoldingModel).all()]
# #     if not holdings:
# #         raise HTTPException(400, "Portfolio is empty")
# #     for tk in holdings:
# #         fetch_and_store_ticker_news(tk, db)

# #     # 2) Only keep very recent articles
# #     cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=recent_hours)
# #     raw = (
# #         db.query(ArticleModel)
# #           .filter(ArticleModel.ticker.in_(holdings))
# #           .filter(ArticleModel.published >= cutoff)
# #           .order_by(ArticleModel.published.desc())
# #           .all()
# #     )

# #     # 3) Score *every* article
# #     scored: List[dict] = []
# #     for art in raw:
# #         text = (art.title + " " + (art.summary or "")).upper()
# #         pol  = TextBlob(text).sentiment.polarity
# #         if abs(pol) < min_sentiment:
# #             continue

# #         sym   = art.ticker.replace(".", "-")
# #         start = art.published - datetime.timedelta(minutes=5)
# #         end   = art.published + datetime.timedelta(minutes=window + 5)
# #         try:
# #             df = yf.Ticker(sym).history(start=start, end=end, interval="1m")
# #             if df is None or df.empty:
# #                 continue
# #             closes = df["Close"].tolist()
# #             before, after = closes[0], closes[min(window, len(closes)-1)]
# #             delta = round(after - before, 2)
# #         except Exception:
# #             continue

# #         impact = round(abs(delta) * abs(pol), 4)
# #         scored.append({
# #             "ticker":    art.ticker,
# #             "headline":  art.title,
# #             "link":      art.link,
# #             "source":    "Google News",
# #             "delta":     delta,
# #             "impact":    impact,
# #             "spark":     closes,
# #             "summary":   art.summary or "",
# #             "published": art.published
# #         })

# #     # 4) Sort *all* scored articles by impact, take up to `limit`
# #     result = sorted(scored, key=lambda x: x["impact"], reverse=True)[:limit]
# #     if not result:
# #         raise HTTPException(404, "No recent news-impact data")
# #     return result



# # # ─── Portfolio News Flag ───────────────────────────────────────────────────────
# # @app.get("/portfolio-news", response_model=List[PortfolioNewsFlag])
# # def portfolio_news(
# #     db: Session            = Depends(get_db),
# #     recent_hours: int      = Query(24, ge=1),
# # ):
# #     cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=recent_hours)
# #     holdings = [h.ticker for h in db.query(HoldingModel).all()]
# #     seen = {
# #         art.ticker
# #         for art in db.query(ArticleModel)
# #                      .filter(ArticleModel.ticker.in_(holdings))
# #                      .filter(ArticleModel.published >= cutoff)
# #                      .all()
# #     }
# #     return [
# #         PortfolioNewsFlag(ticker=t, has_news=(t in seen))
# #         for t in holdings
# #     ]

# #!/usr/bin/env python3
# import os, ssl, uuid, datetime, glob, logging
# from pathlib      import Path
# from typing       import List
# from urllib.parse import quote_plus

# import feedparser
# import requests
# import yfinance    as yf
# import pandas      as pd

# from fastapi                  import FastAPI, HTTPException, Depends, Query, BackgroundTasks
# from fastapi.middleware.cors  import CORSMiddleware
# from fastapi.responses        import FileResponse
# from pydantic                 import BaseModel, ConfigDict
# from sqlalchemy               import create_engine, Column, String, Float, DateTime, Text
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm           import sessionmaker, Session
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise        import cosine_similarity
# from textblob                 import TextBlob
# from scipy                    import sparse
# import numpy                    as np

# # ─── SSL & Logging ────────────────────────────────────────────────────────────
# ssl._create_default_https_context = ssl._create_unverified_context
# logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# # ─── App & CORS ────────────────────────────────────────────────────────────────
# app = FastAPI(title="StockRadar Core API")
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# @app.get("/", include_in_schema=False)
# def serve_ui():
#     return FileResponse("index.html")

# # ─── Database setup ─────────────────────────────────────────────────────────────
# DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stockradar.db")
# engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
# Base         = declarative_base()

# class HoldingModel(Base):
#     __tablename__ = "portfolio"
#     id        = Column(String(36), primary_key=True, index=True)
#     ticker    = Column(String(10), nullable=False, unique=True, index=True)
#     quantity  = Column(Float, nullable=False)
#     avg_price = Column(Float, nullable=False)

# class ArticleModel(Base):
#     __tablename__ = "articles"
#     id        = Column(String(36), primary_key=True, index=True)
#     ticker    = Column(String(10), index=True, nullable=False)
#     title     = Column(Text, nullable=False)
#     link      = Column(Text, nullable=False, unique=True)
#     summary   = Column(Text)
#     published = Column(DateTime, nullable=False)

# Base.metadata.create_all(bind=engine)
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # ─── Pydantic Schemas ──────────────────────────────────────────────────────────
# class Holding(BaseModel):
#     id: str; ticker: str; quantity: float; avg_price: float
#     model_config = ConfigDict(from_attributes=True)

# class HoldingCreate(BaseModel):
#     ticker: str; quantity: float

# class Recommendation(BaseModel):
#     ticker: str; name: str; sector: str; description: str
#     similarity_score: float; spark: List[float]; spark_times: List[datetime.datetime]
#     model_config = ConfigDict(from_attributes=True)

# class Quote(BaseModel):
#     ticker: str; name: str; price: float; change: float; spark: List[float]

# class Performer(BaseModel):
#     ticker: str; name: str; sector: str; change_percent: float; spark: List[float]

# class NewsImpactItem(BaseModel):
#     ticker: str; headline: str; link: str; source: str
#     delta: float; impact: float; spark: List[float]
#     summary: str; published: datetime.datetime

# class StockItem(BaseModel):
#     ticker: str; name: str

# class PortfolioNewsFlag(BaseModel):
#     ticker: str; has_news: bool

# # ─── Load S&P 500 metadata & cached summaries ─────────────────────────────────
# wiki_url   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# df         = pd.read_html(wiki_url, flavor="lxml")[0]
# stock_keys = df["Symbol"].tolist()

# SUMMARIES_DIR = Path("data/summaries")
# wiki_summaries = {}
# for txt in SUMMARIES_DIR.glob("*.txt"):
#     wiki_summaries[txt.stem] = txt.read_text(encoding="utf-8").strip()

# stock_meta = {}
# for _, row in df.iterrows():
#     tk = row.Symbol
#     stock_meta[tk] = {
#         "name":        row.Security,
#         "sector":      row["GICS Sector"],
#         "description": wiki_summaries.get(tk, "")
#     }

# # ─── TF-IDF for recommendations ───────────────────────────────────────────────
# tfidf_corpus     = [
#     f"{stock_meta[t]['name']} {stock_meta[t]['sector']} {stock_meta[t]['description'][:500]}"
#     for t in stock_keys
# ]
# tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
# tfidf_matrix     = tfidf_vectorizer.fit_transform(tfidf_corpus)


# # ─── Portfolio CRUD ────────────────────────────────────────────────────────────
# @app.get("/portfolio", response_model=List[Holding])
# def list_portfolio(db: Session = Depends(get_db)):
#     return [Holding.from_orm(h) for h in db.query(HoldingModel).all()]

# @app.post("/portfolio", response_model=Holding, status_code=201)
# def add_holding(
#     h: HoldingCreate,
#     background_tasks: BackgroundTasks,
#     db: Session = Depends(get_db)
# ):
#     t = h.ticker.upper()
#     if db.query(HoldingModel).filter_by(ticker=t).first():
#         raise HTTPException(400, "Ticker already in portfolio")
#     try:
#         avg_price = float(yf.Ticker(t).history(period="1d")["Close"].iloc[-1])
#     except:
#         avg_price = 0.0

#     new = HoldingModel(id=uuid.uuid4().hex, ticker=t, quantity=h.quantity, avg_price=avg_price)
#     db.add(new); db.commit(); db.refresh(new)

#     # fetch news in background
#     background_tasks.add_task(fetch_and_store_ticker_news, t, db)
#     return Holding.from_orm(new)

# @app.put("/portfolio/{holding_id}", response_model=Holding)
# def update_holding(holding_id: str, h: HoldingCreate, db: Session = Depends(get_db)):
#     rec = db.query(HoldingModel).get(holding_id)
#     if not rec: raise HTTPException(404, "Holding not found")
#     t = h.ticker.upper()
#     try:
#         avg_price = float(yf.Ticker(t).history(period="1d")["Close"].iloc[-1])
#     except:
#         avg_price = rec.avg_price
#     rec.ticker, rec.quantity, rec.avg_price = t, h.quantity, avg_price
#     db.commit(); db.refresh(rec)
#     background_tasks.add_task(fetch_and_store_ticker_news, t, db)
#     return Holding.from_orm(rec)

# @app.delete("/portfolio/{holding_id}", status_code=204)
# def delete_holding(holding_id: str, db: Session = Depends(get_db)):
#     db.query(HoldingModel).filter_by(id=holding_id).delete()
#     db.commit()


# # ─── Stocks list ───────────────────────────────────────────────────────────────
# @app.get("/stocks", response_model=List[StockItem])
# def list_stocks():
#     return [{"ticker": tk, "name": stock_meta[tk]["name"]} for tk in stock_keys]


# # ─── Recommendations ───────────────────────────────────────────────────────────
# def fetch_spark_for(tkr, period="1d", interval="1m"):
#     df = yf.Ticker(tkr.replace(".","-")).history(period=period, interval=interval)
#     return (df["Close"].tolist(), df.index.to_pydatetime().tolist()) if df is not None and not df.empty else ([],[])

# @app.get("/recommendations", response_model=List[Recommendation])
# def recommend(db: Session = Depends(get_db)):
#     holdings = [h.ticker for h in db.query(HoldingModel).all()]
#     if not holdings: raise HTTPException(400, "Portfolio is empty")
#     idxs = [stock_keys.index(t) for t in holdings if t in stock_keys]
#     if not idxs: raise HTTPException(400, "No metadata for portfolio tickers")

#     centroid = tfidf_matrix[idxs].mean(axis=0)
#     if sparse.issparse(centroid):
#         cent = centroid.toarray().ravel()
#     elif hasattr(centroid,"A"):
#         cent = centroid.A.ravel()
#     else:
#         cent = np.asarray(centroid).ravel()

#     sims = cosine_similarity(cent.reshape(1,-1), tfidf_matrix).flatten()

#     recs = []
#     for i in sims.argsort()[::-1]:
#         tk = stock_keys[i]
#         if tk in holdings: continue
#         meta = stock_meta[tk]
#         spark, times = fetch_spark_for(tk)
#         desc_file = SUMMARIES_DIR/f"{tk}.txt"
#         desc = desc_file.read_text(encoding="utf-8").strip() if desc_file.exists() else meta["description"]
#         recs.append(Recommendation(
#             ticker           = tk,
#             name             = meta["name"],
#             sector           = meta["sector"],
#             description      = desc,
#             similarity_score = round(float(sims[i]),3),
#             spark            = spark,
#             spark_times      = times
#         ))
#         if len(recs)>=25: break
#     return recs


# # ─── Quotes ────────────────────────────────────────────────────────────────────
# @app.get("/quotes", response_model=List[Quote])
# def get_quotes(
#     ticker: str = Query(None),
#     period: str = Query("1d"),
#     interval: str = Query("1m"),
#     db: Session = Depends(get_db)
# ):
#     symbols = [ticker] if ticker else [h.ticker for h in db.query(HoldingModel).all()]
#     out = []
#     for t in symbols:
#         df = yf.Ticker(t.replace(".","-")).history(period=period, interval=interval)
#         if df is None or df.empty: continue
#         opens, closes = df["Open"].tolist(), df["Close"].tolist()
#         out.append({
#             "ticker": t,
#             "name":   stock_meta[t]["name"],
#             "price":  round(closes[-1],2),
#             "change": round(closes[-1]-opens[0],2),
#             "spark":  closes
#         })
#     if not out: raise HTTPException(404, "No quote data")
#     return out


# # ─── Portfolio History ─────────────────────────────────────────────────────────
# @app.get("/portfolio-history", response_model=List[dict])
# def portfolio_history(
#     period: str = Query("1d"), interval: str = Query("1m"),
#     db: Session = Depends(get_db)
# ):
#     ts_map = {}
#     for h in db.query(HoldingModel).all():
#         df = yf.Ticker(h.ticker).history(period=period, interval=interval)
#         if df is None or df.empty: continue
#         for idx,row in df.iterrows():
#             ts_map.setdefault(idx.isoformat(),0.0)
#             ts_map[idx.isoformat()] += row["Close"]*h.quantity
#     return [{"time":t,"value":round(v,2)} for t,v in sorted(ts_map.items())]


# # ─── Top Performers ────────────────────────────────────────────────────────────
# @app.get("/top-performers", response_model=List[Performer])
# def top_performers(
#     period: str = Query("1d"), interval: str = Query("1m")
# ):
#     perf = []
#     for tk in stock_keys:
#         df = yf.Ticker(tk.replace(".","-")).history(period=period, interval=interval)
#         if df is None or df.empty: continue
#         o,c = df["Open"].iloc[0], df["Close"].iloc[-1]
#         perf.append(Performer(
#             ticker         = tk,
#             name           = stock_meta[tk]["name"],
#             sector         = stock_meta[tk]["sector"],
#             change_percent = round((c-o)/o*100,2),
#             spark          = df["Close"].tolist()
#         ))
#     if not perf: raise HTTPException(404, "No performance data")
#     return sorted(perf, key=lambda x: x.change_percent, reverse=True)[:10]


# # ─── News Impact ───────────────────────────────────────────────────────────────
# @app.get("/news-impact", response_model=List[NewsImpactItem])
# def news_impact(
#     db: Session            = Depends(get_db),
#     window: int            = Query(30, ge=1),
#     min_sentiment: float   = Query(0.0, ge=0.0),
#     recent_hours: int      = Query(24, ge=1),
#     limit: int             = Query(100, ge=1, le=5000),
# ):
#     holdings = [h.ticker for h in db.query(HoldingModel).all()]
#     if not holdings:
#         raise HTTPException(400, "Portfolio is empty")

#     # fetch fresh headlines only for current holdings
#     for tk in holdings:
#         fetch_and_store_ticker_news(tk, db)

#     cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=recent_hours)
#     raw    = (db.query(ArticleModel)
#                 .filter(ArticleModel.ticker.in_(holdings))
#                 .filter(ArticleModel.published >= cutoff)
#                 .order_by(ArticleModel.published.desc())
#                 .all())

#     scored = []
#     for art in raw:
#         text = (art.title + " " + (art.summary or "")).upper()
#         pol  = TextBlob(text).sentiment.polarity
#         if abs(pol) < min_sentiment: continue

#         start = art.published - datetime.timedelta(minutes=5)
#         end   = art.published + datetime.timedelta(minutes=window+5)
#         try:
#             df = yf.Ticker(art.ticker).history(start=start, end=end, interval="1m")
#             closes = df["Close"].tolist()
#             before, after = closes[0], closes[min(window,len(closes)-1)]
#             delta = round(after-before,2)
#         except:
#             continue

#         scored.append({
#             "ticker":    art.ticker,
#             "headline":  art.title,
#             "link":      art.link,
#             "source":    "GoogleRSS|yFinance|NewsAPI",
#             "delta":     delta,
#             "impact":    round(abs(delta)*abs(pol),4),
#             "spark":     closes,
#             "summary":   art.summary or "",
#             "published": art.published
#         })

#     if not scored:
#         raise HTTPException(404, "No recent news-impact data")
#     return sorted(scored, key=lambda x: x["impact"], reverse=True)[:limit]

# # ─── Portfolio-News Flag ───────────────────────────────────────────────────────
# @app.get("/portfolio-news", response_model=List[PortfolioNewsFlag])
# def portfolio_news(
#     db: Session            = Depends(get_db),
#     recent_hours: int      = Query(24, ge=1),
# ):
#     cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=recent_hours)
#     holdings = [h.ticker for h in db.query(HoldingModel).all()]
#     seen = {
#         art.ticker
#         for art in db.query(ArticleModel)
#                      .filter(ArticleModel.ticker.in_(holdings))
#                      .filter(ArticleModel.published >= cutoff)
#                      .all()
#     }
#     return [PortfolioNewsFlag(ticker=t, has_news=(t in seen)) for t in holdings]

#!/usr/bin/env python3
import os
import ssl
import uuid
import datetime
import logging
import feedparser
import requests

from pathlib import Path
from typing    import List

import yfinance as yf
import pandas   as pd
from fastapi    import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses      import FileResponse
from pydantic               import BaseModel, ConfigDict
from sqlalchemy             import (
    create_engine, Column, String, Float, DateTime, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm         import sessionmaker, Session

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity
from textblob import TextBlob

from apscheduler.schedulers.background import BackgroundScheduler

# ─── SSL & Logging ─────────────────────────────────────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ─── FastAPI setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="StockRadar Core API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse("index.html")


# ─── Database (SQLite + SQLAlchemy) ─────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stockradar.db")
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base         = declarative_base()

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
    id: str; ticker: str; quantity: float; avg_price: float
    model_config = ConfigDict(from_attributes=True)

class HoldingCreate(BaseModel):
    ticker: str; quantity: float

class Recommendation(BaseModel):
    ticker: str; name: str; sector: str; description: str
    similarity_score: float; spark: List[float]; spark_times: List[datetime.datetime]
    model_config = ConfigDict(from_attributes=True)

class Quote(BaseModel):
    ticker: str; name: str; price: float; change: float; spark: List[float]

class Performer(BaseModel):
    ticker: str; name: str; sector: str; change_percent: float; spark: List[float]

class NewsImpactItem(BaseModel):
    ticker: str; headline: str; link: str; source: str
    delta: float; impact: float; spark: List[float]
    summary: str; published: datetime.datetime

class StockItem(BaseModel):
    ticker: str; name: str

class PortfolioNewsFlag(BaseModel):
    ticker: str; has_news: bool


# ─── Load S&P 500 metadata & summaries ────────────────────────────────────────
wiki_url   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
df         = pd.read_html(wiki_url, flavor="lxml")[0]
stock_keys = df["Symbol"].tolist()

# cached Wikipedia summaries in data/summaries/*.txt
SUMMARIES_DIR = Path("data/summaries")
wiki_summaries = {
    p.stem: p.read_text(encoding="utf-8").strip()
    for p in SUMMARIES_DIR.glob("*.txt")
}

stock_meta = {}
for _, row in df.iterrows():
    tk = row.Symbol
    stock_meta[tk] = {
        "name":        row.Security,
        "sector":      row["GICS Sector"],
        "description": wiki_summaries.get(tk, "")
    }

# build TF-IDF corpus for recommendations
tfidf_corpus     = [
    f"{stock_meta[t]['name']} {stock_meta[t]['sector']} "
    f"{stock_meta[t]['description'][:500]}"
    for t in stock_keys
]
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
tfidf_matrix     = tfidf_vectorizer.fit_transform(tfidf_corpus)


from urllib.parse import quote_plus

# def fetch_and_store_ticker_news(ticker: str, db: Session):
#     """
#     Fetch latest news for a ticker from:
#        1) Google News RSS (company name)
#        2) Yahoo-Finance JSON
#        3) SEC EDGAR 8-K Atom feed
#     Deduplicate by URL, insert only novel articles.
#     """
#     seen = {link for (link,) in db.query(ArticleModel.link)
#                    .filter(ArticleModel.ticker==ticker).all()}

#     # — Google News RSS — exact company name
#     company = stock_meta[ticker]["name"]
#     q       = quote_plus(f'"{company}"')
#     rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
#     feed    = feedparser.parse(rss_url)
#     for e in feed.entries:
#         link    = e.get("link","").strip()
#         if link and link not in seen:
#             title   = e.get("title","").strip()
#             summary = e.get("summary","").strip()
#             try:
#                 published = datetime.datetime(*e.published_parsed[:6])
#             except:
#                 published = datetime.datetime.utcnow()
#             db.add(ArticleModel(
#                 id        = uuid.uuid4().hex,
#                 ticker    = ticker,
#                 title     = title, link=link,
#                 summary   = summary,
#                 published = published
#             ))
#             seen.add(link)

#     # — Yahoo-Finance JSON — yf.Ticker(...).news
#     yf_sym = ticker.replace(".", "-")
#     for art in yf.Ticker(yf_sym).news:
#         content = art.get("content",{}) or {}
#         link = (
#             content.get("canonicalUrl",{}).get("url")
#             or content.get("clickThroughUrl",{}).get("url")
#             or art.get("link")
#         )
#         if link and link not in seen:
#             title   = art.get("title","").strip()
#             summary = art.get("summary","") or title
#             tstamp  = art.get("providerPublishTime")
#             published = (
#                 datetime.datetime.utcfromtimestamp(tstamp)
#                 if tstamp else datetime.datetime.utcnow()
#             )
#             db.add(ArticleModel(
#                 id        = uuid.uuid4().hex,
#                 ticker    = ticker,
#                 title     = title,
#                 link      = link,
#                 summary   = summary,
#                 published = published
#             ))
#             seen.add(link)

#     # # — SEC EDGAR 8-K Feed — get current year’s feed
#     # edgar_url = (
#     #     "https://www.sec.gov/Archives/edgar/usgaap.rss"
#     #     # you may refine to only FD filings, see SEC documentation
#     # )
#     # feed = feedparser.parse(edgar_url)
#     # for e in feed.entries:
#     #     link = e.get("link","").strip()
#     #     if link in seen:
#     #         continue
#     #     # include only if the filing mentions our ticker
#     #     if ticker.lower() in (e.get("summary","")+e.get("title","")).lower():
#     #         try:
#     #             published = datetime.datetime(*e.published_parsed[:6])
#     #         except:
#     #             published = datetime.datetime.utcnow()
#     #         db.add(ArticleModel(
#     #             id        = uuid.uuid4().hex,
#     #             ticker    = ticker,
#     #             title     = e.get("title","").strip(),
#     #             link      = link,
#     #             summary   = e.get("summary","").strip(),
#     #             published = published
#     #         ))
#     #         seen.add(link)

#     db.commit()

from sqlalchemy.exc import IntegrityError

def fetch_and_store_ticker_news(ticker: str, db: Session):
    """
    Fetch latest news for `ticker` from multiple sources (Google RSS, yfinance.news),
    dedupe by URL, and insert into ArticleModel — skipping duplicates gracefully.
    """
    # Gather links we already have for this ticker
    seen_links = {
        link for (link,) in db.query(ArticleModel.link)
                            .filter(ArticleModel.ticker == ticker)
                            .all()
    }

    to_insert = []

    # — Source A: Google News RSS —
    company = stock_meta[ticker]["name"]
    q       = quote_plus(f'"{company}"')
    rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    feed    = feedparser.parse(rss_url)

    for entry in feed.entries:
        link    = entry.get("link","").strip()
        if not link or link in seen_links:
            continue
        title   = entry.get("title","").strip()
        summary = entry.get("summary","").strip()
        try:
            published = datetime.datetime(*entry.published_parsed[:6])
        except:
            published = datetime.datetime.utcnow()

        to_insert.append(ArticleModel(
            id        = uuid.uuid4().hex,
            ticker    = ticker,
            title     = title,
            link      = link,
            summary   = summary,
            published = published
        ))
        seen_links.add(link)

    # — Source B: yfinance.news —
    yf_sym = ticker.replace(".", "-")
    for art in yf.Ticker(yf_sym).news:
        content = art.get("content", {}) or {}
        link = (
            content.get("canonicalUrl", {}).get("url")
            or content.get("clickThroughUrl", {}).get("url")
            or art.get("link")
        )
        if not link or link in seen_links:
            continue

        title = art.get("title","").strip()
        summary = art.get("summary") or title
        tstamp = art.get("providerPublishTime")
        if tstamp:
            published = datetime.datetime.utcfromtimestamp(tstamp)
        else:
            published = datetime.datetime.utcnow()

        to_insert.append(ArticleModel(
            id        = uuid.uuid4().hex,
            ticker    = ticker,
            title     = title,
            link      = link,
            summary   = summary,
            published = published
        ))
        seen_links.add(link)
        
    # # — SEC EDGAR 8-K Feed — get current year’s feed
#     # edgar_url = (
#     #     "https://www.sec.gov/Archives/edgar/usgaap.rss"
#     #     # you may refine to only FD filings, see SEC documentation
#     # )
#     # feed = feedparser.parse(edgar_url)
#     # for e in feed.entries:
#     #     link = e.get("link","").strip()
#     #     if link in seen:
#     #         continue
#     #     # include only if the filing mentions our ticker
#     #     if ticker.lower() in (e.get("summary","")+e.get("title","")).lower():
#     #         try:
#     #             published = datetime.datetime(*e.published_parsed[:6])
#     #         except:
#     #             published = datetime.datetime.utcnow()
#     #         db.add(ArticleModel(
#     #             id        = uuid.uuid4().hex,
#     #             ticker    = ticker,
#     #             title     = e.get("title","").strip(),
#     #             link      = link,
#     #             summary   = e.get("summary","").strip(),
#     #             published = published
#     #         ))
#     #         seen.add(link)

    # Bulk‐insert but skip duplicates
    for article in to_insert:
        db.add(article)
        try:
            db.commit()
        except IntegrityError:
            # someone else inserted the same link just before us — skip it
            db.rollback()
        else:
            # committed successfully
            pass

# ─── Scheduler ────────────────────────────────────────────────────────────────
scheduler = BackgroundScheduler()
def crawl_all_portfolio():
    db = SessionLocal()
    try:
        tickers = [h.ticker for h in db.query(HoldingModel).all()]
        for tk in tickers:
            fetch_and_store_ticker_news(tk, db)
    finally:
        db.close()

# schedule every 5 minutes
scheduler.add_job(crawl_all_portfolio, 'interval', minutes=60, next_run_time=datetime.datetime.utcnow())
scheduler.start()


# ─── Utility for fetching spark data ──────────────────────────────────────────
def fetch_spark_for(ticker: str, period="1d", interval="1m"):
    df = yf.Ticker(ticker.replace(".", "-")).history(period=period, interval=interval)
    if df is None or df.empty:
        return [], []
    return df["Close"].tolist(), df.index.to_pydatetime().tolist()


# ─── Portfolio CRUD ────────────────────────────────────────────────────────────
@app.get("/portfolio", response_model=List[Holding])
def list_portfolio(db: Session = Depends(get_db)):
    return [Holding.from_orm(h) for h in db.query(HoldingModel).all()]

@app.post("/portfolio", response_model=Holding, status_code=201)
def add_holding(h: HoldingCreate, db: Session = Depends(get_db)):
    t = h.ticker.upper()
    if db.query(HoldingModel).filter_by(ticker=t).first():
        raise HTTPException(400, "Ticker already in portfolio")
    try:
        hist      = yf.Ticker(t.replace(".", "-")).history(period="1d")
        avg_price = float(hist["Close"].iloc[-1])
    except:
        avg_price = 0.0

    new = HoldingModel(id=uuid.uuid4().hex, ticker=t, quantity=h.quantity, avg_price=avg_price)
    db.add(new); db.commit(); db.refresh(new)
    # immediately crawl once for this new ticker
    fetch_and_store_ticker_news(t, db)
    return Holding.from_orm(new)

@app.put("/portfolio/{holding_id}", response_model=Holding)
def update_holding(holding_id: str, h: HoldingCreate, db: Session = Depends(get_db)):
    rec = db.query(HoldingModel).get(holding_id)
    if not rec:
        raise HTTPException(404, "Holding not found")
    t = h.ticker.upper()
    try:
        hist      = yf.Ticker(t.replace(".", "-")).history(period="1d")
        avg_price = float(hist["Close"].iloc[-1])
    except:
        avg_price = rec.avg_price
    rec.ticker, rec.quantity, rec.avg_price = t, h.quantity, avg_price
    db.commit(); db.refresh(rec)
    return Holding.from_orm(rec)

@app.delete("/portfolio/{holding_id}", status_code=204)
def delete_holding(holding_id: str, db: Session = Depends(get_db)):
    db.query(HoldingModel).filter_by(id=holding_id).delete()
    db.commit()


# ─── Stocks list for autocomplete ─────────────────────────────────────────────
@app.get("/stocks", response_model=List[StockItem])
def list_stocks():
    return [{"ticker": tk, "name": stock_meta[tk]["name"]} for tk in stock_keys]


# ─── Recommendations ───────────────────────────────────────────────────────────
@app.get("/recommendations", response_model=List[Recommendation])
def recommend(db: Session = Depends(get_db)):
    holdings = [h.ticker for h in db.query(HoldingModel).all()]
    if not holdings:
        raise HTTPException(400, "Portfolio is empty")
    idxs = [stock_keys.index(t) for t in holdings if t in stock_keys]
    if not idxs:
        raise HTTPException(400, "No metadata for portfolio tickers")

    # centroid of TF-IDF
    centroid = tfidf_matrix[idxs].mean(axis=0)
    # convert safely to 1d numpy
    import numpy as np
    from scipy import sparse
    if sparse.issparse(centroid):
        centroid = centroid.toarray().ravel()
    else:
        centroid = np.asarray(centroid).ravel()

    sims = cosine_similarity(centroid.reshape(1,-1), tfidf_matrix).flatten()
    recs = []
    for i in sims.argsort()[::-1]:
        tk = stock_keys[i]
        if tk in holdings:
            continue
        spark, times = fetch_spark_for(tk)
        desc_file    = SUMMARIES_DIR / f"{tk}.txt"
        description  = desc_file.read_text(encoding="utf-8").strip() \
                       if desc_file.exists() else stock_meta[tk]["description"]
        recs.append(Recommendation(
            ticker           = tk,
            name             = stock_meta[tk]["name"],
            sector           = stock_meta[tk]["sector"],
            description      = description,
            similarity_score = round(float(sims[i]), 3),
            spark            = spark,
            spark_times      = times
        ))
        if len(recs) >= 25:
            break
    return recs


# ─── Quotes ────────────────────────────────────────────────────────────────────
@app.get("/quotes", response_model=List[Quote])
def get_quotes(
    ticker: str = Query(None),
    period: str = Query("1d"),
    interval: str = Query("1m"),
    db: Session = Depends(get_db)
):
    symbols = [ticker] if ticker else [h.ticker for h in db.query(HoldingModel).all()]
    out = []
    for t in symbols:
        spark, _ = fetch_spark_for(t, period, interval)
        df       = yf.Ticker(t.replace(".", "-")).history(period=period, interval=interval)
        if df is None or df.empty:
            continue
        delta = round(df["Close"].iloc[-1] - df["Open"].iloc[0], 2)
        out.append({
            "ticker": t,
            "name":   stock_meta[t]["name"],
            "price":  round(df["Close"].iloc[-1], 2),
            "change": delta,
            "spark":  spark
        })
    if not out:
        raise HTTPException(404, "No quote data")
    return out


# ─── Portfolio History ─────────────────────────────────────────────────────────
@app.get("/portfolio-history", response_model=List[dict])
def portfolio_history(
    period: str = Query("1d"), interval: str = Query("1m"),
    db: Session = Depends(get_db)
):
    ts_map = {}
    for h in db.query(HoldingModel).all():
        df = yf.Ticker(h.ticker.replace(".", "-")).history(period=period, interval=interval)
        for idx, row in df.iterrows():
            ts = idx.isoformat()
            ts_map.setdefault(ts, 0)
            ts_map[ts] += row["Close"] * h.quantity
    return [{"time": t, "value": round(v,2)} for t,v in sorted(ts_map.items())]


# ─── Top Performers ────────────────────────────────────────────────────────────
@app.get("/top-performers", response_model=List[Performer])
def top_performers(
    period: str = Query("1d"), interval: str = Query("1m")
):
    perf = []
    for tk in stock_keys:
        df = yf.Ticker(tk.replace(".", "-")).history(period=period, interval=interval)
        if df is None or df.empty:
            continue
        o, c = df["Open"].iloc[0], df["Close"].iloc[-1]
        pct  = round((c-o)/o * 100, 2)
        perf.append(Performer(
            ticker         = tk,
            name           = stock_meta[tk]["name"],
            sector         = stock_meta[tk]["sector"],
            change_percent = pct,
            spark          = df["Close"].tolist()
        ))
    if not perf:
        raise HTTPException(404, "No performance data")
    return sorted(perf, key=lambda x: x.change_percent, reverse=True)[:10]


# ─── News Impact ───────────────────────────────────────────────────────────────
@app.get("/news-impact", response_model=List[NewsImpactItem])
def news_impact(
    db: Session            = Depends(get_db),
    window: int            = Query(30, ge=1),
    min_sentiment: float   = Query(0.0, ge=0.0),
    recent_hours: int      = Query(24, ge=1),
    limit: int             = Query(100, ge=1, le=5000),
):
    # 1) fetch & store fresh headlines for current holdings
    holdings = [h.ticker for h in db.query(HoldingModel).all()]
    if not holdings:
        raise HTTPException(400, "Portfolio is empty")
    for tk in holdings:
        fetch_and_store_ticker_news(tk, db)

    # 2) cutoff window
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=recent_hours)
    raw    = db.query(ArticleModel)\
               .filter(ArticleModel.ticker.in_(holdings))\
               .filter(ArticleModel.published >= cutoff)\
               .order_by(ArticleModel.published.desc())\
               .all()

    scored = []
    for art in raw:
        text = (art.title + " " + (art.summary or "")).upper()
        pol  = TextBlob(text).sentiment.polarity
        if abs(pol) < min_sentiment:
            continue
        start = art.published - datetime.timedelta(minutes=5)
        end   = art.published + datetime.timedelta(minutes=window+5)
        try:
            df = yf.Ticker(art.ticker.replace(".", "-"))\
                   .history(start=start, end=end, interval="1m")
            closes = df["Close"].tolist()
            before, after = closes[0], closes[min(window, len(closes)-1)]
            delta = round(after-before,2)
        except:
            continue
        impact = round(abs(delta)*abs(pol),4)
        scored.append({
            "ticker":    art.ticker,
            "headline":  art.title,
            "link":      art.link,
            "source":    "Google News and yfinance.news",
            "delta":     delta,
            "impact":    impact,
            "spark":     closes,
            "summary":   art.summary or "",
            "published": art.published
        })

    result = sorted(scored, key=lambda x: x["impact"], reverse=True)[:limit]
    if not result:
        raise HTTPException(404, "No recent news-impact data")
    return result


# ─── Portfolio-News Flag ───────────────────────────────────────────────────────
@app.get("/portfolio-news", response_model=List[PortfolioNewsFlag])
def portfolio_news(
    db: Session            = Depends(get_db),
    recent_hours: int      = Query(24, ge=1),
):
    cutoff   = datetime.datetime.utcnow() - datetime.timedelta(hours=recent_hours)
    holdings = [h.ticker for h in db.query(HoldingModel).all()]
    seen     = {
        art.ticker
        for art in db.query(ArticleModel)
                     .filter(ArticleModel.ticker.in_(holdings))
                     .filter(ArticleModel.published >= cutoff)
                     .all()
    }
    return [
        PortfolioNewsFlag(ticker=t, has_news=(t in seen))
        for t in holdings
    ]
