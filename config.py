"""
config.py - Forex Trading Bot Configuration v1.0
================================================
"""

import os
import pytz
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------
# KONFIGURASI TELEGRAM
# -------------------------------------------------------
TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# -------------------------------------------------------
# KONFIGURASI AI (GEMINI & GROQ)
# -------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-2.0-flash"

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# -------------------------------------------------------
# FOREX WATCHLIST
# -------------------------------------------------------
FOREX_PAIRS_MAJOR: list[str] = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", 
    "USDCAD=X", "USDCHF=X", "NZDUSD=X"
]

FOREX_PAIRS_CROSS: list[str] = [
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "EURCHF=X", 
    "AUDJPY=X", "GBPAUD=X", "CADJPY=X"
]

COMMODITIES: list[str] = [
    "GC=F",   # Gold
    "CL=F",   # Crude Oil
    "BZ=F"    # Brent Oil
]

FOREX_WATCHLIST: list[str] = FOREX_PAIRS_MAJOR + FOREX_PAIRS_CROSS + COMMODITIES

# Legacy support
WATCHLIST = FOREX_WATCHLIST
KOMPAS100 = FOREX_WATCHLIST 

# -------------------------------------------------------
# KONFIGURASI ANALISA TEKNIKAL (SCALPING)
# -------------------------------------------------------
EMA_FAST: int = 5
EMA_SLOW: int = 13
RSI_PERIOD: int = 14
VOLUME_SMA: int = 20
ATR_PERIOD: int = 14
ATR_MULTIPLIER: float = 1.5
RISK_REWARD_RATIO: float = 1.3

RSI_OVERBOUGHT: float = 70.0
VOLUME_SURGE_MULTIPLIER: float = 1.1  # Forex volume is different

YFINANCE_INTERVAL: str = "15m"
YFINANCE_PERIOD: str = "5d"

# -------------------------------------------------------
# FILTER VOLATILITAS (PIPS)
# -------------------------------------------------------
# Untuk Forex, kita filter berdasarkan pips atau minimal movement
VOLATILITY_MIN_PCT: float = 0.05  # Minimal pergerakan 0.05%
VOLATILITY_MAX_PCT: float = 1.5   # Maksimal pergerakan 1.5% (menghindari spike berita)

# Untuk /danger: drop tajam
DANGER_DROP_PCT: float = -0.5

# -------------------------------------------------------
# KONFIGURASI RADAR & SCORING
# -------------------------------------------------------
TECHNICAL_SCORE_BUY: int = 55      
TECHNICAL_SCORE_DANGER: int = 35   

# -------------------------------------------------------
# FOREX SESSIONS (WIB)
# -------------------------------------------------------
WIB = pytz.timezone("Asia/Jakarta")
# Tokyo: 07:00 - 15:00 WIB
SESSION_TOKYO_START = 7
SESSION_TOKYO_END = 15
# London: 14:00 - 22:00 WIB
SESSION_LONDON_START = 14
SESSION_LONDON_END = 22
# New York: 19:00 - 03:00 WIB
SESSION_NY_START = 19
SESSION_NY_END = 3

MARKET_OPEN_HOUR: int = 0   # Forex is 24h
MARKET_CLOSE_HOUR: int = 24
RADAR_INTERVAL_MINUTES: int = 15

# -------------------------------------------------------
# SCRAPER BERITA
# -------------------------------------------------------
MAX_NEWS_ARTICLES: int = 5
SENTIMENT_CACHE_TTL_MINUTES: int = 30

# -------------------------------------------------------
# VALIDASI
# -------------------------------------------------------
def validate_config() -> None:
    errors = []
    if not TELEGRAM_TOKEN:
        errors.append("TELEGRAM_TOKEN belum diset di file .env")
    if not TELEGRAM_CHAT_ID:
        errors.append("TELEGRAM_CHAT_ID belum diset di file .env")
    if not GEMINI_API_KEY and not GROQ_API_KEY:
        errors.append("Setidaknya GEMINI_API_KEY atau GROQ_API_KEY harus diset")

    if errors:
        print("[CONFIG] ⚠️ Warning: Konfigurasi tidak lengkap:")
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print(f"[CONFIG] ✅ Forex Config Loaded. Pairs: {len(FOREX_WATCHLIST)}")


if __name__ == "__main__":
    validate_config()
