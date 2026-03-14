import asyncio
import logging
import pandas as pd
from data_fetcher import full_screening, get_autoscalping_candidates
from news_scraper import get_news_for_forex, get_macro_news
from ai_analyzer import analyze_sentiment, analyze_autoscalping
from db_manager import init_db
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_forex_flow():
    # 0. Init Database
    await init_db()
    
    # 1. Test Data Fetcher
    pair = "EURUSD=X"
    print(f"\n--- Testing Data Fetcher with {pair} ---")
    data = full_screening(pair)
    if data:
        print(f"Harga Terakhir: {data['harga_terakhir']}")
        print(f"Risk Management: {data['risk_management']}")
        print(f"Technical Score: {data['technical_score']}")
    else:
        print("Gagal fetch data")

    # 2. Test News Scraper
    print(f"\n--- Testing News Scraper for {pair} ---")
    headlines = await get_news_for_forex(pair)
    print(f"Headlines found: {len(headlines)}")
    for h in headlines[:3]:
        print(f"- {h}")

    # 3. Test AI Analyzer (Sentiment)
    if headlines:
        print(f"\n--- Testing AI Sentiment Analysis for {pair} ---")
        tech_ctx = {
            "technical_score": data.get("technical_score", 0),
            "rsi": data["kondisi"]["rsi"]["nilai"],
            "uptrend_daily": True,
            "bb_squeeze": False,
            "bb_breakout": False
        }
        sentiment = await analyze_sentiment(pair, headlines, tech_ctx)
        print(f"Sentimen: {sentiment['sentimen']}")
        print(f"Rekomendasi: {sentiment['rekomendasi']}")
        print(f"Alasan: {sentiment['alasan_singkat']}")

    # 4. Test AutoScalping Candidates
    print("\n--- Testing AutoScalping Candidates ---")
    candidates = get_autoscalping_candidates(config.FOREX_PAIRS_MAJOR, force=True)
    print(f"Candidates found: {[c['kode'] for c in candidates]}")

    # 5. Test Macro News
    print("\n--- Testing Macro News ---")
    macro = await get_macro_news(3)
    for m in macro:
        print(f"- {m}")

if __name__ == "__main__":
    asyncio.run(test_forex_flow())
