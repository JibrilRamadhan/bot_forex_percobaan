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
# MONEY & RISK MANAGEMENT
# -------------------------------------------------------
DEFAULT_EQUITY: float = 1000.0        # Modal trading dalam USD (contoh $1,000)
RISK_PER_TRADE_PCT: float = 1.0       # Batas risiko kerugian maksimal per trade (1%)
ATR_PERIOD: int = 14
ATR_MULTIPLIER: float = 1.5           # Pengali ATR untuk Stop Loss
RISK_REWARD_RATIO: float = 1.5        # RR 1:1.5

# -------------------------------------------------------
# KONFIGURASI AI (GEMINI & GROQ)
# -------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-2.0-flash"

GROQ_API_KEYS_STR: str = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
GROQ_API_KEYS: list[str] = [k.strip() for k in GROQ_API_KEYS_STR.split(",")] if GROQ_API_KEYS_STR else []
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

# -------------------------------------------------------
# WEBSOCKET — THE GOLDEN 8 (TwelveData Real-time)
# -------------------------------------------------------
TWELVEDATA_API_KEY: str = os.getenv("TWELVEDATA_API_KEY", "")

# Map: TwelveData symbol → yfinance ticker (untuk full_screening)
WS_GOLDEN_8: dict[str, str] = {
    "EUR/USD":  "EURUSD=X",
    "GBP/USD":  "GBPUSD=X",
    "USD/JPY":  "USDJPY=X",
    "XAU/USD":  "GC=F",       # Gold — paling wajib untuk scalping!
    "AUD/USD":  "AUDUSD=X",
    "USD/CAD":  "USDCAD=X",
    "USD/CHF":  "USDCHF=X",
    "EUR/JPY":  "EURJPY=X",
}

# Pair non-WS yang tetap di-handle oleh yfinance polling 15 menit
WS_NON_STREAMING = [p for p in FOREX_WATCHLIST if p not in WS_GOLDEN_8.values()]

# --- Signal Engine Thresholds ---
WS_TRIGGER_PCT: float = 0.10       # Minimum pergerakan harga (0.10% = ±10 pips) dalam 5 mnt
WS_VOLUME_SPIKE: float = 3.0       # Minimum tick volume spike multiplier
WS_WINDOW_SECONDS: int = 300       # Lookback window untuk trigger (5 menit)
WS_COOLDOWN_MINUTES: int = 10      # Anti-spam: min jeda antar alert per pair (menit)
WS_ANALYSIS_QUEUE_MAX: int = 5     # Max antrian analisa AI bersamaan


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
DEFAULT_EQUITY: float = 1000.0  # Default balance user $1000
RISK_PER_TRADE_PCT: float = 1.0 # Risiko 1% per posisi

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

# Golden Hours (London/NY Overlap): 19:00 - 22:00 WIB
SESSION_GOLDEN_START = 19
SESSION_GOLDEN_END = 22

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
    if not GEMINI_API_KEY and not GROQ_API_KEYS:
        errors.append("Setidaknya GEMINI_API_KEY atau GROQ_API_KEYS harus diset")

    if errors:
        print("[CONFIG] ⚠️ Warning: Konfigurasi tidak lengkap:")
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print(f"[CONFIG] ✅ Forex Config Loaded. Pairs: {len(FOREX_WATCHLIST)}")


if __name__ == "__main__":
    validate_config()
