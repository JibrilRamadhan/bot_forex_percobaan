"""
config.py - Konfigurasi Pusat v3.0
====================================
v3.0: Watchlist Kompas100 (~100 saham liquid BEI), filter volatilitas +2%-+5%
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
# KONFIGURASI GEMINI API (Fallback)
# -------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-2.0-flash"

# -------------------------------------------------------
# KONFIGURASI GROQ API (Primary - 30 RPM gratis)
# -------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# -------------------------------------------------------
# WATCHLIST INTI (digunakan oleh Radar Otomatis 15 menit)
# Sekarang menggunakan seluruh Kompas100 untuk jangkauan lebih luas
# -------------------------------------------------------
WATCHLIST: list[str] = []  # Diisi setelah KOMPAS100 di-define (lihat bawah)

# -------------------------------------------------------
# KOMPAS100 - ~100 Saham Paling Liquid BEI (Mode Radar Sapu Bersih)
# Update quarterly → sesuaikan dengan publikasi Kompas100 terbaru
# -------------------------------------------------------
KOMPAS100: list[str] = [
    # --- PERBANKAN ---
    "BBCA", "BBRI", "BMRI", "BBNI", "BBTN", "BRIS", "NISP", "MEGA", "BNGA",
    # --- TELEKOMUNIKASI ---
    "TLKM", "EXCL", "ISAT", "MTEL", "TBIG", "TOWR",
    # --- ENERGI & PERTAMBANGAN ---
    "ADRO", "PTBA", "HRUM", "ITMG", "INDY", "MEDC", "PGAS", "ELSA",
    "MDKA", "ANTM", "TINS", "INCO", "NCKL", "MBMA",
    # --- CONSUMER & RETAIL ---
    "UNVR", "ICBP", "INDF", "MYOR", "GGRM", "HMSP", "SIDO", "KLBF",
    "KAEF", "HEAL", "MIKA", "SILO", "ROTI", "ACES", "MAPI", "AMRT",
    "RALS", "MAP",
    # --- PROPERTI & KONSTRUKSI ---
    "BSDE", "CTRA", "LPKR", "PWON", "SMGR", "INTP", "SMCB",
    "ADHI", "PTPP", "WSKT", "WTON", "DMAS",
    # --- INFRASTRUKTUR ---
    "JSMR", "WIKA", "IPCC", "PJAA",
    # --- TEKNOLOGI & DIGITAL ---
    "GOTO", "BUKA", "EMTK", "SCMA", "MNCN", "FILM", "NFCX", "WIFI",
    # --- INDUSTRI & MANUFAKTUR ---
    "ASII", "AUTO", "AALI", "LSIP", "TAPG", "JPFA", "CPIN", "ISSP",
    "TPIA", "BRPT", "INKP",
    # --- KEUANGAN NON-BANK ---
    "ESSA", "SRTG", "PNLF", "AKRA", "ERAA",
    # --- PROPERTI & KAWASAN INDUSTRI ---
    "KIJA", "MDIY", "DOID",
    # --- TRANSPORTASI & LOGISTIK ---
    "RAJA", "GJTL",
    # --- TAMBAHAN SAHAM AKTIF ---
    "INET", "AMMN", "BREN", "PGEO", "NFCX",
    # --- LQ45 LAINNYA ---
    "UNTR", "IMAS", "MPPA",
]
# Hapus duplikat
KOMPAS100 = list(dict.fromkeys(KOMPAS100))

# Radar otomatis menggunakan seluruh Kompas100
WATCHLIST: list[str] = KOMPAS100

# -------------------------------------------------------
# KONFIGURASI ANALISA TEKNIKAL
# -------------------------------------------------------
EMA_FAST: int = 5
EMA_SLOW: int = 13
RSI_PERIOD: int = 14
VOLUME_SMA: int = 20

RSI_OVERBOUGHT: float = 70.0
VOLUME_SURGE_MULTIPLIER: float = 2.0

YFINANCE_INTERVAL: str = "15m"
YFINANCE_PERIOD: str = "5d"

# -------------------------------------------------------
# FILTER VOLATILITAS (v3.0)
# Sinyal BUY hanya untuk saham yang naik "sweet spot"
# -------------------------------------------------------
VOLATILITY_MIN_PCT: float = 2.0   # Harga naik minimal +2% (already moving)
VOLATILITY_MAX_PCT: float = 8.0   # Harga naik maksimal +8% (belum terlambat, bukan "gorengan")

# Untuk /danger: saham turun lebih dari ini dianggap berbahaya
DANGER_DROP_PCT: float = -2.5

# -------------------------------------------------------
# KONFIGURASI RADAR & SCORING
# -------------------------------------------------------
TECHNICAL_SCORE_BUY: int = 60      # /rekomendasi: minimum score untuk masuk daftar BUY
TECHNICAL_SCORE_DANGER: int = 30   # /danger: score di bawah ini dianggap lemah

# -------------------------------------------------------
# JADWAL & WAKTU
# -------------------------------------------------------
WIB = pytz.timezone("Asia/Jakarta")
MARKET_OPEN_HOUR: int = int(os.getenv("MARKET_OPEN_HOUR", "9"))
MARKET_CLOSE_HOUR: int = int(os.getenv("MARKET_CLOSE_HOUR", "16"))
RADAR_INTERVAL_MINUTES: int = int(os.getenv("RADAR_INTERVAL_MINUTES", "15"))

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
        raise EnvironmentError(
            f"[CONFIG] Konfigurasi tidak lengkap:\n"
            + "\n".join(f"  ❌ {e}" for e in errors)
            + "\nSalin .env.example ke .env dan isi nilainya."
        )
    print(f"[CONFIG] ✅ Konfigurasi dimuat. Kompas100: {len(KOMPAS100)} saham.")


if __name__ == "__main__":
    validate_config()
    print(f"Kompas100 ({len(KOMPAS100)} saham): {', '.join(KOMPAS100[:10])}...")
