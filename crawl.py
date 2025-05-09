#!/usr/bin/env python3
"""
crawl_rss.py – portfolio‐focused RSS crawler + general finance feeds,
with summary/description fallback and dedupe of title==summary.
"""

import json
import time
from pathlib import Path
from datetime import datetime
import requests
import feedparser

# CONFIG
API_PORTFOLIO = "http://localhost:8000/portfolio"
GENERAL_FEEDS = {
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories",
    "Reuters":     "https://feeds.reuters.com/reuters/businessNews",
    "CNBC":        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
}
OUTPUT_DIR = Path("data/raw_articles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 10

# disable cert warnings
import urllib3
import warnings
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)


def fetch_rss(url: str, source: str):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, verify=False)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ❌ {source} HTTP error: {e}")
        return []

    feed = feedparser.parse(resp.content)
    if feed.bozo:
        print(f"  ❌ {source} parse error: {feed.bozo_exception}")
        return []

    out = []
    for e in feed.entries:
        title = e.get("title", "").strip()
        link  = e.get("link", "").strip()

        # published timestamp if available
        pub = ""
        if getattr(e, "published_parsed", None):
            pub = datetime.utcfromtimestamp(time.mktime(e.published_parsed))\
                  .strftime("%Y-%m-%dT%H:%M:%SZ")
        elif getattr(e, "updated_parsed", None):
            pub = datetime.utcfromtimestamp(time.mktime(e.updated_parsed))\
                  .strftime("%Y-%m-%dT%H:%M:%SZ")

        # summary vs description
        raw_sum = e.get("summary") or ""
        raw_des = e.get("description") or ""

        # choose non‐empty and not duplicate of title
        pick = raw_sum.strip() or raw_des.strip()
        if pick.strip() == title:
            pick = raw_des.strip() if raw_des.strip() != title else ""

        out.append({
            "title":     title,
            "link":      link,
            "published": pub,
            "summary":   pick,
            "source":    source
        })
    print(f"  ✅ {len(out)} articles from {source}")
    return out


def main():
    seen = set()
    all_articles = []

    # 1) portfolio‐specific Google News
    try:
        res = requests.get(API_PORTFOLIO, headers=HEADERS, timeout=5)
        res.raise_for_status()
        portfolio = [h["ticker"] for h in res.json()]
    except Exception as e:
        print(f"❌ Could not fetch portfolio: {e}")
        portfolio = []

    if portfolio:
        query = "+OR+".join(portfolio)
        url   = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        print("Fetching portfolio‐specific news…")
        for art in fetch_rss(url, "PortfolioNews"):
            if art["link"] not in seen:
                seen.add(art["link"])
                all_articles.append(art)
        time.sleep(1)

    # 2) general feeds
    for src, url in GENERAL_FEEDS.items():
        print(f"Fetching {src} feed…")
        for art in fetch_rss(url, src):
            if art["link"] not in seen:
                seen.add(art["link"])
                all_articles.append(art)
        time.sleep(1)

    # save out
    ts = int(time.time())
    out_path = OUTPUT_DIR / f"rss_articles_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_articles, f, indent=2)
    print(f"\n✨ Saved {len(all_articles)} unique articles to {out_path}")


if __name__ == "__main__":
    main()
