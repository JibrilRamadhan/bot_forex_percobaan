"""
data_fetcher.py - Data & Indikator Teknikal (v2.0)
====================================================
Perubahan v2.0:
- ATR-14 untuk Stop Loss dinamis
- Bollinger Bands (20,2) + Squeeze detection
- Multi-Timeframe: konfirmasi tren dari data Daily
- Technical Scoring System (0-100)
- Kalkulasi Stop Loss, Target Price, Risk/Reward otomatis
"""

import time
import logging
import pandas as pd
import ta
import yfinance as yf
import numpy as np
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# UTILITAS
# ----------------------------------------------------------------
def format_ticker(kode_saham: str) -> str:
    kode_saham = kode_saham.strip().upper()
    if not kode_saham.endswith(".JK"):
        kode_saham = f"{kode_saham}.JK"
    return kode_saham


def get_clean_code(ticker: str) -> str:
    return ticker.replace(".JK", "").upper()


# ----------------------------------------------------------------
# FETCH DATA
# ----------------------------------------------------------------
def fetch_ohlcv(
    kode_saham: str,
    interval: str = config.YFINANCE_INTERVAL,
    period: str = config.YFINANCE_PERIOD,
    max_retry: int = 3,
) -> Optional[pd.DataFrame]:
    ticker = format_ticker(kode_saham)
    for attempt in range(1, max_retry + 1):
        try:
            logger.info(f"[FETCHER] Mengambil data {ticker} interval={interval} (percobaan {attempt})")
            df = yf.download(ticker, interval=interval, period=period,
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                if attempt < max_retry:
                    time.sleep(2 ** attempt)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            logger.info(f"[FETCHER] ✅ {len(df)} candle untuk {ticker}")
            return df
        except Exception as e:
            logger.error(f"[FETCHER] Error percobaan {attempt}: {e}")
            if attempt < max_retry:
                time.sleep(2 ** attempt)
    return None


def fetch_daily(kode_saham: str) -> Optional[pd.DataFrame]:
    """Ambil data harian (60 hari) untuk multi-timeframe analysis."""
    return fetch_ohlcv(kode_saham, interval="1d", period="60d")


def fetch_info(kode_saham: str) -> dict:
    ticker = format_ticker(kode_saham)
    try:
        info = yf.Ticker(ticker).info
        return {
            "nama_perusahaan": info.get("longName", get_clean_code(ticker)),
            "sektor": info.get("sector", "N/A"),
        }
    except Exception:
        return {"nama_perusahaan": get_clean_code(ticker), "sektor": "N/A"}


# ----------------------------------------------------------------
# KALKULASI INDIKATOR (15 MENIT)
# ----------------------------------------------------------------
def calculate_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Hitung semua indikator teknikal pada data 15 menit:
    EMA5, EMA13, RSI14, Volume SMA20, ATR14, Bollinger Bands(20,2)
    """
    try:
        min_rows = max(config.EMA_SLOW, config.RSI_PERIOD, config.VOLUME_SMA, 20) * 2
        if len(df) < min_rows:
            logger.warning(f"[INDICATOR] Data tidak cukup: {len(df)} baris")
            return None

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]

        # --- EMA ---
        df[f"EMA_{config.EMA_FAST}"] = ta.trend.EMAIndicator(close, config.EMA_FAST, fillna=False).ema_indicator()
        df[f"EMA_{config.EMA_SLOW}"] = ta.trend.EMAIndicator(close, config.EMA_SLOW, fillna=False).ema_indicator()

        # --- RSI ---
        df[f"RSI_{config.RSI_PERIOD}"] = ta.momentum.RSIIndicator(close, config.RSI_PERIOD, fillna=False).rsi()

        # --- Volume SMA ---
        df[f"VOL_SMA_{config.VOLUME_SMA}"] = vol.rolling(config.VOLUME_SMA).mean()

        # --- ATR-14 (untuk Stop Loss) ---
        df["ATR_14"] = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14, fillna=False
        ).average_true_range()

        # --- Bollinger Bands (20, 2) ---
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2, fillna=False)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_mid"] = bb.bollinger_mavg()
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]

        df.dropna(inplace=True)
        if df.empty:
            return None

        logger.info(f"[INDICATOR] ✅ {len(df)} candle valid dengan semua indikator")
        return df

    except Exception as e:
        logger.error(f"[INDICATOR] Error: {e}")
        return None


# ----------------------------------------------------------------
# MULTI-TIMEFRAME: CEK TREN DAILY
# ----------------------------------------------------------------
def get_daily_trend(kode_saham: str) -> dict:
    """
    Ambil data harian dan hitung EMA-20 Daily.
    Return: apakah harga saat ini berada di atas EMA-20 Daily (uptrend).
    """
    df_daily = fetch_daily(kode_saham)
    default = {"uptrend_daily": False, "harga_vs_ema20d": 0.0, "ema20_daily": 0.0}

    if df_daily is None or df_daily.empty or len(df_daily) < 25:
        return default

    try:
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = df_daily.columns.get_level_values(0)
        df_daily.columns = [c.lower() for c in df_daily.columns]

        close_daily = df_daily["close"]
        ema20d = ta.trend.EMAIndicator(close_daily, window=20, fillna=False).ema_indicator().dropna()

        if ema20d.empty:
            return default

        harga_terakhir = float(close_daily.iloc[-1])
        nilai_ema20d = float(ema20d.iloc[-1])
        uptrend = harga_terakhir > nilai_ema20d
        selisih_pct = ((harga_terakhir - nilai_ema20d) / nilai_ema20d) * 100

        logger.info(f"[MTF] Daily trend: harga={harga_terakhir:,.0f} EMA20d={nilai_ema20d:,.0f} uptrend={uptrend}")
        return {
            "uptrend_daily": uptrend,
            "harga_vs_ema20d": round(selisih_pct, 2),
            "ema20_daily": round(nilai_ema20d, 2),
        }

    except Exception as e:
        logger.warning(f"[MTF] Error daily trend: {e}")
        return default


# ----------------------------------------------------------------
# DETEKSI SINYAL
# ----------------------------------------------------------------
def detect_signal(df: pd.DataFrame) -> dict:
    """Deteksi semua kondisi sinyal dari candle terakhir."""
    col_ef = f"EMA_{config.EMA_FAST}"
    col_es = f"EMA_{config.EMA_SLOW}"
    col_rsi = f"RSI_{config.RSI_PERIOD}"
    col_vsma = f"VOL_SMA_{config.VOLUME_SMA}"

    row_prev = df.iloc[-2]
    row_last = df.iloc[-1]

    # EMA Crossover
    ema_was_below = row_prev[col_ef] < row_prev[col_es]
    ema_now_above = row_last[col_ef] > row_last[col_es]
    is_crossover = ema_was_below and ema_now_above
    ema_bullish = row_last[col_ef] > row_last[col_es]  # EMA fast masih di atas

    # Volume
    vol_current = float(row_last["volume"])
    vol_sma = float(row_last[col_vsma]) if col_vsma in df.columns else 1
    vol_ratio = vol_current / vol_sma if vol_sma > 0 else 0
    is_vol_surge = vol_ratio >= config.VOLUME_SURGE_MULTIPLIER

    # RSI
    rsi_val = float(row_last[col_rsi])
    is_rsi_safe = rsi_val < config.RSI_OVERBOUGHT

    # Bollinger Bands Squeeze + Breakout
    bb_squeeze = False
    bb_breakout = False
    bb_width_now = float(row_last.get("BB_width", 0))
    bb_upper = float(row_last.get("BB_upper", 0))
    close_now = float(row_last["close"])

    if "BB_width" in df.columns and len(df) >= 20:
        bb_width_hist = df["BB_width"].tail(20)
        bb_percentile_20 = float(bb_width_hist.quantile(0.25))
        bb_squeeze = bb_width_now < bb_percentile_20
        bb_breakout = (close_now > bb_upper) and is_vol_surge and bb_squeeze

    # Sinyal utama valid jika SEMUA 3 kondisi dasar terpenuhi
    is_valid = is_crossover and is_vol_surge and is_rsi_safe

    return {
        "sinyal_valid": is_valid,
        "harga_terakhir": round(close_now, 2),
        "kondisi": {
            "crossover": {
                "status": is_crossover,
                "ema_bullish": ema_bullish,
                "ema_fast_sebelum": round(float(row_prev[col_ef]), 2),
                "ema_slow_sebelum": round(float(row_prev[col_es]), 2),
                "ema_fast_sekarang": round(float(row_last[col_ef]), 2),
                "ema_slow_sekarang": round(float(row_last[col_es]), 2),
            },
            "volume": {
                "status": is_vol_surge,
                "volume_sekarang": int(vol_current),
                "volume_sma": round(float(vol_sma), 0),
                "rasio": round(vol_ratio, 2),
            },
            "rsi": {
                "status": is_rsi_safe,
                "nilai": round(rsi_val, 2),
                "batas_overbought": config.RSI_OVERBOUGHT,
            },
            "bollinger": {
                "squeeze": bb_squeeze,
                "breakout": bb_breakout,
                "bb_width": round(bb_width_now, 4),
                "bb_upper": round(bb_upper, 2),
            },
        },
    }


# ----------------------------------------------------------------
# ATR STOP LOSS & TARGET
# ----------------------------------------------------------------
def calculate_risk_management(df: pd.DataFrame, harga: float) -> dict:
    """
    Hitung Stop Loss dinamis berdasarkan ATR dan Target Price.
    Stop Loss = harga - (1.5 × ATR)
    Target    = harga + (2.0 × ATR)  → Risk/Reward minimal 1:1.3
    """
    try:
        atr = float(df["ATR_14"].iloc[-1])
        stop_loss = harga - (1.5 * atr)
        target = harga + (2.0 * atr)
        risiko = harga - stop_loss
        potensi = target - harga
        rr = round(potensi / risiko, 2) if risiko > 0 else 0
        return {
            "atr": round(atr, 2),
            "stop_loss": round(stop_loss, 2),
            "target_price": round(target, 2),
            "risiko_per_saham": round(risiko, 2),
            "potensi_per_saham": round(potensi, 2),
            "risk_reward": rr,
        }
    except Exception as e:
        logger.warning(f"[RISK] Error kalkulasi risk management: {e}")
        return {"atr": 0, "stop_loss": 0, "target_price": 0, "risiko_per_saham": 0, "potensi_per_saham": 0, "risk_reward": 0}


# ----------------------------------------------------------------
# TECHNICAL SCORING (0-100)
# ----------------------------------------------------------------
def calculate_technical_score(sinyal: dict, daily_trend: dict) -> int:
    """
    Sistem scoring teknikal 0-100 berdasarkan semua kondisi.
    Digunakan untuk menentukan rekomendasi BUY / SELL / HOLD.
    """
    score = 0
    k = sinyal["kondisi"]

    # EMA Crossover (25 poin)
    if k["crossover"]["status"]:
        score += 25  # Crossover baru terjadi
    elif k["crossover"]["ema_bullish"]:
        score += 12  # EMA fast masih di atas (tren berlanjut)

    # Volume (20 poin)
    vol_rasio = k["volume"]["rasio"]
    if vol_rasio >= 3.0:
        score += 20
    elif vol_rasio >= 2.0:
        score += 15
    elif vol_rasio >= 1.2:
        score += 8

    # RSI (20 poin)
    rsi = k["rsi"]["nilai"]
    if 40 <= rsi <= 60:
        score += 20   # RSI zona netral terbaik
    elif 30 <= rsi < 40:
        score += 15   # Mendekati oversold, peluang rebound
    elif 60 < rsi <= 70:
        score += 10   # Masih aman tapi mulai panas
    elif rsi < 30:
        score += 18   # Oversold kuat, potensi reversal

    # Bollinger Bands Breakout (20 poin)
    if k["bollinger"]["breakout"]:
        score += 20   # Breakout dari squeeze = sinyal A++
    elif k["bollinger"]["squeeze"]:
        score += 10   # Squeeze saja = energi terkumpul, siap meledak

    # Multi-Timeframe Konfirmasi Daily (15 poin)
    if daily_trend["uptrend_daily"]:
        score += 15   # Tren utama naik = sinyal lebih valid

    return min(score, 100)


# ----------------------------------------------------------------
# PIVOT POINT
# ----------------------------------------------------------------
def calculate_pivot_points(df: pd.DataFrame) -> dict:
    try:
        df_work = df.copy()
        if not isinstance(df_work.index, pd.DatetimeIndex):
            df_work = df_work.set_index(pd.to_datetime(df_work.index))
        df_daily = df_work.resample("D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        prev = df_daily.iloc[-2] if len(df_daily) >= 2 else df_work.iloc[-2]
        h, l, c = float(prev["high"]), float(prev["low"]), float(prev["close"])
        pp = (h + l + c) / 3
        return {
            "PP": round(pp, 2),
            "R1": round((2 * pp) - l, 2),
            "R2": round(pp + (h - l), 2),
            "S1": round((2 * pp) - h, 2),
            "S2": round(pp - (h - l), 2),
        }
    except Exception as e:
        logger.warning(f"[PIVOT] Error: {e}")
        return {"PP": 0, "R1": 0, "R2": 0, "S1": 0, "S2": 0}


# ----------------------------------------------------------------
# FULL SCREENING (main function)
# ----------------------------------------------------------------
def full_screening(kode_saham: str) -> Optional[dict]:
    """
    Screening lengkap v2.0:
    Data 15m + Daily + semua indikator + risk management + scoring
    """
    ticker = format_ticker(kode_saham)
    kode_bersih = get_clean_code(kode_saham)

    # 1. Data 15 menit
    df = fetch_ohlcv(ticker)
    if df is None:
        return None

    # 2. Hitung indikator
    df = calculate_indicators(df)
    if df is None:
        return None

    # 3. Deteksi sinyal
    sinyal = detect_signal(df)
    harga = sinyal["harga_terakhir"]

    # 4. Multi-Timeframe (daily trend)
    daily_trend = get_daily_trend(ticker)

    # 5. Risk Management (ATR-based)
    risk = calculate_risk_management(df, harga)

    # 6. Pivot Points
    pivots = calculate_pivot_points(df)

    # 7. Technical Score
    tech_score = calculate_technical_score(sinyal, daily_trend)

    # 8. Perubahan harga vs candle sebelumnya
    harga_prev = float(df.iloc[-2]["close"])
    perubahan_pct = ((harga - harga_prev) / harga_prev) * 100

    # 9. Info perusahaan
    info = fetch_info(ticker)

    return {
        "ticker": ticker,
        "kode": kode_bersih,
        "nama_perusahaan": info.get("nama_perusahaan", kode_bersih),
        "harga_terakhir": harga,
        "perubahan_pct": round(perubahan_pct, 2),
        "pivot_points": pivots,
        "daily_trend": daily_trend,
        "risk_management": risk,
        "technical_score": tech_score,
        "df": df,  # Dibutuhkan untuk generate chart
        **sinyal,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hasil = full_screening("BBCA")
    if hasil:
        print(f"Saham: {hasil['kode']} | Harga: Rp {hasil['harga_terakhir']:,.0f}")
        print(f"Score Teknikal: {hasil['technical_score']}/100")
        print(f"ATR Stop Loss: Rp {hasil['risk_management']['stop_loss']:,.0f}")
        print(f"Target: Rp {hasil['risk_management']['target_price']:,.0f}")
        print(f"Daily Uptrend: {hasil['daily_trend']['uptrend_daily']}")
        print(f"BB Squeeze: {hasil['kondisi']['bollinger']['squeeze']}")
        print(f"BB Breakout: {hasil['kondisi']['bollinger']['breakout']}")
