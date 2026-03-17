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
from datetime import datetime
import ta
import yfinance as yf
import numpy as np
from typing import Optional

import config
from news_scraper import is_kill_switch_active

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# UTILITAS
# ----------------------------------------------------------------
def format_ticker(kode: str) -> str:
    kode = kode.strip().upper()
    # Forex major/cross often ends with =X in Yahoo Finance
    if any(x in kode for x in ["=X", "USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD"]):
        if not kode.endswith("=X") and len(kode) == 6:
            return f"{kode}=X"
        return kode
    # Commodities like GC=F (Gold), CL=F (Oil)
    if "=F" in kode:
        return kode
    # Fallback to .JK for stocks if any remains (though this bot is for forex)
    if not any(x in kode for x in ["=X", "=F", "^", ".", "="]):
         return f"{kode}.JK"
    return kode


def get_clean_code(ticker: str) -> str:
    return ticker.replace(".JK", "").replace("=X", "").upper()


def get_pip_multiplier(ticker: str) -> float:
    """Helper untuk PIP calculation (Forex 4/5 digit vs JPY 2/3 digit)."""
    if "JPY" in ticker:
        return 100.0  # 1 pip = 0.01
    return 10000.0    # 1 pip = 0.0001


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

        logger.info(f"[MTF] Daily trend: harga={harga_terakhir:.5f} EMA20d={nilai_ema20d:.5f} uptrend={uptrend}")
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
        "harga_terakhir": round(close_now, 5),
        "kondisi": {
            "crossover": {
                "status": is_crossover,
                "ema_bullish": ema_bullish,
                "ema_fast_sebelum": round(float(row_prev[col_ef]), 5),
                "ema_slow_sebelum": round(float(row_prev[col_es]), 5),
                "ema_fast_sekarang": round(float(row_last[col_ef]), 5),
                "ema_slow_sekarang": round(float(row_last[col_es]), 5),
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
# ATR STOP LOSS & TARGET & LOT CALCULATION (v7.0)
# ----------------------------------------------------------------
def calculate_lot_size(equity: float, risk_pct: float, sl_pips: float, ticker: str) -> float:
    """
    Hitung Lot Size agar risiko tetap (Risk Management).
    Formula: RiskAmount / (SL_Pips * PipValue_per_standard_lot)
    Standard Lot = 100,000 units.
    Pip Value per Standard Lot (untuk XXX/USD) = $10.
    """
    if sl_pips <= 0:
        return 0.01
        
    risk_amount = equity * (risk_pct / 100)
    
    # Estimasi Pip Value (Standard Lot 100k)
    # Untuk XXX/USD = $10. Untuk JPY pairs = variable tapi bisa di-proxied $10.
    pip_value_std = 10.0 
    
    # Lot = Risk / (SL * PipValue)
    lot = risk_amount / (sl_pips * pip_value_std)
    
    # Pembulatan di Forex biasanya 0.01 (Micro Lot)
    return max(0.01, round(lot, 2))

def calculate_risk_management(df: pd.DataFrame, harga: float, ticker: str = "EURUSD=X") -> dict:
    """
    Hitung Stop Loss dinamis berdasarkan ATR dan Target Price.
    Stop Loss = harga - (ATR_MULTIPLIER × ATR)
    Target    = harga + (ATR_MULTIPLIER × ATR × RISK_REWARD)
    """
    try:
        col_atr = f"ATR_{config.ATR_PERIOD}"
        atr = float(df[col_atr].iloc[-1]) if col_atr in df.columns else float(df.filter(like='ATR').iloc[-1])
        
        stop_loss = harga - (config.ATR_MULTIPLIER * atr)
        target = harga + (config.ATR_MULTIPLIER * atr * config.RISK_REWARD_RATIO)
        
        # Hitung jarak dalam Pip
        multiplier = get_pip_multiplier(ticker)
        
        risiko_pips = abs(harga - stop_loss) * multiplier
        potensi_pips = abs(target - harga) * multiplier
        
        rr = round(potensi_pips / risiko_pips, 2) if risiko_pips > 0 else 0
        
        # v7.0 Position Sizing
        recommended_lot = calculate_lot_size(
            config.DEFAULT_EQUITY, 
            config.RISK_PER_TRADE_PCT, 
            risiko_pips, 
            ticker
        )
        
        return {
            "atr": round(atr, 6),
            "stop_loss": round(stop_loss, 5),
            "target_price": round(target, 5),
            "stop_pips": round(risiko_pips, 1),
            "target_pips": round(potensi_pips, 1),
            "risk_reward": rr,
            "recommended_lot": recommended_lot
        }
    except Exception as e:
        logger.warning(f"[RISK] Error kalkulasi risk management: {e}")
        return {
            "atr": 0, "stop_loss": harga, "target_price": harga,
            "stop_pips": 0, "target_pips": 0, "risk_reward": 0, "recommended_lot": 0.01
        }


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
            "PP": round(pp, 5),
            "R1": round((2 * pp) - l, 5),
            "R2": round(pp + (h - l), 5),
            "S1": round((2 * pp) - h, 5),
            "S2": round(pp - (h - l), 5),
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
    risk = calculate_risk_management(df, harga, ticker)

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


# ----------------------------------------------------------------
# BULK DOWNLOAD (v3.0) - Tarik 100 Saham Sekaligus
# ----------------------------------------------------------------
def bulk_fetch_ohlcv(kode_list: list[str], interval: str = "15m", period: str = "2d", min_len: int = 2) -> dict:
    """
    Download data OHLCV untuk banyak saham SEKALIGUS dalam 1 request yfinance.
    Jauh lebih cepat daripada looping satu per satu.

    Returns:
        dict: { 'BBCA': DataFrame, 'TLKM': DataFrame, ... }
    """
    tickers = [format_ticker(k) for k in kode_list]
    tickers_str = " ".join(tickers)

    try:
        logger.info(f"[BULK] Mengunduh {len(tickers)} saham sekaligus...")
        raw = yf.download(
            tickers=tickers_str,
            interval=interval,
            period=period,
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
        result = {}
        for ticker in tickers:
            kode = get_clean_code(ticker)
            try:
                if len(tickers) == 1:
                    # Single ticker: tidak ada multi-index
                    df = raw.copy()
                else:
                    df = raw[ticker].copy()

                if df is None or df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                df.dropna(subset=["close", "volume"], inplace=True)
                if len(df) >= min_len:
                    result[kode] = df
            except Exception:
                continue

        logger.info(f"[BULK] ✅ {len(result)}/{len(tickers)} saham berhasil diunduh")
        return result

    except Exception as e:
        logger.error(f"[BULK] Error bulk download: {e}")
        return {}


def quick_scan(kode_saham: str, df: pd.DataFrame) -> dict | None:
    """
    Versi ringan dari full_screening — tanpa fetch info, tanpa daily trend.
    Digunakan untuk scan massal /rekomendasi dan /danger.
    Input df sudah di-lowercase kolomnya.
    """
    try:
        df_ind = calculate_indicators(df)
        if df_ind is None or len(df_ind) < 3:
            return None

        sinyal = detect_signal(df_ind)
        harga = sinyal["harga_terakhir"]

        # Perubahan harga vs candle pertama hari ini
        # Ambil close sekitar 27 candle lalu (≈ 1 hari kalau interval 15m)
        lookback = min(27, len(df_ind) - 1)
        harga_ref = float(df_ind.iloc[-lookback]["close"])
        perubahan_pct = ((harga - harga_ref) / harga_ref) * 100 if harga_ref > 0 else 0

        risk = calculate_risk_management(df_ind, harga, kode_saham)

        # Score ringan tanpa daily trend
        score_ringan = calculate_technical_score(sinyal, {"uptrend_daily": False})

        return {
            "kode": kode_saham,
            "harga_terakhir": harga,
            "perubahan_pct": round(perubahan_pct, 2),
            "technical_score": score_ringan,
            "risk_management": risk,
            "sinyal_valid": sinyal["sinyal_valid"],
            "kondisi": sinyal["kondisi"],
        }

    except Exception as e:
        logger.warning(f"[QUICK_SCAN] Error {kode_saham}: {e}")
        return None


def scan_forex_buy(kode_list: list[str], calendar: list = None) -> list[dict]:
    """
    Scan semua instrumen di kode_list untuk mencari kandidat BUY.
    Filter: technical_score >= config.TECHNICAL_SCORE_BUY
            AND perubahan_pct dalam range VOLATILITY_MIN - VOLATILITY_MAX
            AND volume surge
            AND No Red Folder News (Kill-Switch)
    """
    logger.info(f"[REKO] Mulai scan {len(kode_list)} pair untuk kandidat BUY...")
    data_map = bulk_fetch_ohlcv(kode_list, min_len=10)
    candidates = []

    for kode, df in data_map.items():
        result = quick_scan(kode, df)
        if result is None:
            continue
        score = result["technical_score"]
        pct = result["perubahan_pct"]
        vol_surge = result["kondisi"]["volume"]["status"]

        # Filter volatilitas sweet spot
        in_sweet_spot = config.VOLATILITY_MIN_PCT <= pct <= config.VOLATILITY_MAX_PCT
        
        # Check Kill-Switch (Red Folder Filter)
        killed = False
        kill_reason = ""
        if calendar:
            ticker = format_ticker(kode)
            # Check both base and quote currency
            base = ticker[:3]
            quote = ticker[3:6]
            
            kill_base, msg_base = is_kill_switch_active(calendar, base)
            kill_quote, msg_quote = is_kill_switch_active(calendar, quote)
            
            if kill_base or kill_quote:
                killed = True
                kill_reason = msg_base or msg_quote

        if score >= config.TECHNICAL_SCORE_BUY and in_sweet_spot and vol_surge and not killed:
            candidates.append(result)
        elif killed:
            logger.warning(f"[REKO] 🛡 {kode} diabaikan: {kill_reason} (Kill-Switch)")

    candidates.sort(key=lambda x: x["technical_score"], reverse=True)
    logger.info(f"[REKO] ✅ {len(candidates)} kandidat BUY ditemukan")
    return candidates[:10]  # Top 10


def scan_forex_danger(kode_list: list[str]) -> list[dict]:
    """
    Scan semua instrumen di kode_list untuk mendeteksi pergerakan BERBAHAYA.
    Filter: perubahan_pct <= DANGER_DROP_PCT
            OR (RSI overbought AND volume turun)
    """
    logger.info(f"[DANGER] Mulai scan {len(kode_list)} pair untuk deteksi bahaya...")
    data_map = bulk_fetch_ohlcv(kode_list, min_len=10)
    dangerous = []

    for kode, df in data_map.items():
        result = quick_scan(kode, df)
        if result is None:
            continue
        pct = result["perubahan_pct"]
        rsi = result["kondisi"]["rsi"]["nilai"]
        score = result["technical_score"]

        # Instrumen dianggap berbahaya jika:
        # 1. Turun drastis dalam sehari, ATAU
        # 2. RSI overbought (> 75) dengan score lemah (potensi reversal turun)
        is_dropping = pct <= config.DANGER_DROP_PCT
        is_overbought_weak = rsi > 75 and score < 40

        if is_dropping or is_overbought_weak:
            # Semakin merah semakin atas daftar
            result["danger_score"] = abs(pct) if is_dropping else rsi / 10
            result["danger_type"] = "DROP" if is_dropping else "OVERBOUGHT"
            dangerous.append(result)

    dangerous.sort(key=lambda x: x["danger_score"], reverse=True)
    logger.info(f"[DANGER] ✅ {len(dangerous)} bahaya terdeteksi")
    return dangerous[:10]  # Top 10


# ----------------------------------------------------------------
# CURRENCY STRENGTH METER (v7.0 HOLY GRAIL)
# ----------------------------------------------------------------
def calculate_csm(all_data: list[dict]) -> dict:
    """
    Kalkulasi kekuatan mata uang relatif (8 major currencies).
    Input: all_data dari get_market_leaders (change_pct harian).
    """
    majors = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]
    strength = {m: {"total": 0.0, "count": 0} for m in majors}
    
    for d in all_data:
        kode = d["kode"]
        if len(kode) != 6: continue
        
        base, quote = kode[:3], kode[3:6]
        change = d["change_pct"]
        
        # Base currency: performance sejalan dengan pair
        if base in strength:
            strength[base]["total"] += change
            strength[base]["count"] += 1
            
        # Quote currency: performance berlawanan dengan pair
        if quote in strength:
            strength[quote]["total"] -= change
            strength[quote]["count"] += 1
            
    # Hitung rata-rata
    results = {}
    for m, val in strength.items():
        avg = val["total"] / val["count"] if val["count"] > 0 else 0.0
        results[m] = round(avg, 2)
        
    # Sort dari terkuat ke terlemah
    sorted_csm = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    return sorted_csm


def get_market_leaders(kode_list: list[str]) -> dict:
    """
    (v4.0) Ambil data market leaders: Top Gainer, Top Volume, Top Value, dan Live Rebound.
    Hanya butuh data OHLCV dasar, sangat cepat menggunakan bulk_fetch_ohlcv.
    """
    logger.info(f"[MARKET] Mengambil data market dari {len(kode_list)} pair...")
    data_map = bulk_fetch_ohlcv(kode_list, interval="1d", period="2d", min_len=2) # Pakai daily untuk gain hari ini
    
    all_data = []
    
    for kode, df in data_map.items():
        try:
            if len(df) < 2:
                continue
            
            # Hari ini
            current = df.iloc[-1]
            close = float(current["close"])
            open_price = float(current["open"])
            low = float(current["low"])
            high = float(current["high"])
            vol = float(current["volume"])
            
            # Kemarin
            prev = df.iloc[-2]
            prev_close = float(prev["close"])
            prev_high = float(prev["high"])
            prev_low = float(prev["low"])
            
            # Kalkulasi Pivot Point standar klasik
            pp = (prev_high + prev_low + prev_close) / 3
            s1 = (2 * pp) - prev_high
            
            # Metric
            change_pct = ((close - prev_close) / prev_close) * 100
            value = close * vol
            
            # Rebound logic: sempat turun kena Support 1 tapi ditutup (current) hijau di atas open
            is_rebound = low <= s1 and close > open_price and change_pct > 0
            
            all_data.append({
                "kode": kode,
                "harga": close,
                "change_pct": change_pct,
                "volume": vol,
                "value": value,
                "is_rebound": is_rebound
            })
        except Exception:
            pass
            
    if not all_data:
        return {}

    # Sort data
    top_gainers = sorted(all_data, key=lambda x: x["change_pct"], reverse=True)[:5]
    top_volume = sorted(all_data, key=lambda x: x["volume"], reverse=True)[:5]
    top_value = sorted(all_data, key=lambda x: x["value"], reverse=True)[:5]
    live_rebound = [d for d in all_data if d["is_rebound"]]
    # Sort rebound berdasarkan change terkuat
    live_rebound = sorted(live_rebound, key=lambda x: x["change_pct"], reverse=True)[:5]
    
    # v7.0 Generate CSM
    csm_data = calculate_csm(all_data)
    
    logger.info("[MARKET] ✅ Berhasil mendapatkan data market leaders")
    return {
        "top_gainer": top_gainers,
        "top_volume": top_volume,
        "top_value": top_value,
        "live_rebound": live_rebound,
        "csm": csm_data
    }


def get_autoscalping_candidates(kode_list: list[str], force: bool = False, calendar: list = None) -> list[dict]:
    """
    (v5.0 & v6.0) Cari 1-3 kandidat TERBAIK secara kuantitatif untuk Auto Scalping.
    Jika force=True, filter lebih longgar untuk memastikan ada kandidat.
    """
    logger.info(f"[AUTOSCALP] Memulai filter kuantitatif scalping dari {len(kode_list)} pair... (Force: {force})")
    
    # -- V1.0 FOREX MACRO WEATHER CHECK (DXY Correlation) --
    is_dxy_rallying = False
    try:
        dxy_df = yf.Ticker("DX-Y.NYB").history(period="2d", interval="1h")
        if len(dxy_df) >= 2:
            prev_close = dxy_df['Close'].iloc[-2]
            curr_close = dxy_df['Close'].iloc[-1]
            dxy_pct = ((curr_close - prev_close) / prev_close) * 100
            
            if dxy_pct >= 0.2:
                is_dxy_rallying = True
                logger.warning(f"[AUTOSCALP] ⚠️ DXY RALLY (+{dxy_pct:.2f}%): USD menguat tajam, hati-hati pair XXXUSD.")
            else:
                logger.info(f"[AUTOSCALP] 🌤 Cuaca DXY Normal ({dxy_pct:.2f}%).")
    except Exception as e:
        logger.error(f"[AUTOSCALP] Gagal cek DXY: {e}")

    data_map = bulk_fetch_ohlcv(kode_list)
    candidates = []

    for kode, df in data_map.items():
        quick = quick_scan(kode, df)
        if not quick:
            continue
            
        score = quick["technical_score"]
        vol_surge = quick["kondisi"]["volume"]["rasio"]
        rsi = quick["kondisi"]["rsi"]["nilai"]
        
        # -- SESSION AWARE VOLUME FILTER (v7.5) --
        curr_hour = datetime.now(config.WIB).hour
        is_golden = config.SESSION_GOLDEN_START <= curr_hour < config.SESSION_GOLDEN_END
        is_tokyo = config.SESSION_TOKYO_START <= curr_hour < config.SESSION_TOKYO_END
        
        # Jika Sesi Tokyo (Pagi), perketat filter volume krn liquidity rendah/sideways
        if is_tokyo and not force:
            if vol_surge < 3.0: 
                logger.info(f"[AUTOSCALP] 😴 {kode} diabaikan: Volume Tokyo rendah (×{vol_surge:.1f} < 3x)")
                continue

        # Filter awal
        if not force:
            min_score = 65 if is_tokyo else 50 # Tokyo butuh konfirmasi extra
            if score < min_score or vol_surge < 1.0 or rsi >= 75:
                continue
        else:
            # Force mode: filter lebih longgar
            if score < 35 or rsi >= 85:
                continue
            
        full_data = full_screening(kode)
        if not full_data:
            continue
            
        daily_trend_ok = full_data.get("daily_trend", {}).get("uptrend_daily", False)
        if not force and not daily_trend_ok:
            continue
            
        is_squeeze_break = full_data["kondisi"]["bollinger"]["breakout"]
        scalp_power = score + (vol_surge * 10) + (20 if is_squeeze_break else 0)
        
        # -- SESSION SCORING ADJUSTMENTS (v7.5) --
        if is_golden:
            scalp_power += 20
            logger.info(f"[AUTOSCALP] 🏆 {kode} mendapat Bonus Golden Hour (+20)")
        elif is_tokyo:
            scalp_power -= 15 # Penalti Tokyo (False Breakout risk)
            logger.info(f"[AUTOSCALP] 🐌 {kode} terkena Penalti Sesi Tokyo (-15)")
            
        # Jika DXY rally, pair XXXUSD (EURUSD, GBPUSD) biasanya drop
        if is_dxy_rallying and "USD" in full_data["kode"] and not full_data["kode"].startswith("USD"):
            scalp_power -= 20
            
        # -- RED FOLDER KILL SWITCH (v7.0) --
        if calendar:
            base, quote = full_data["kode"][:3], full_data["kode"][3:6]
            k_base, _ = is_kill_switch_active(calendar, base)
            k_quote, _ = is_kill_switch_active(calendar, quote)
            if k_base or k_quote:
                logger.warning(f"[AUTOSCALP] 🛡 {full_data['kode']} dieliminasi karena Red Folder News.")
                continue

        full_data["scalp_power"] = scalp_power
        candidates.append(full_data)

    # v7.5 MAXIMUM EXPOSURE & CORRELATION MANAGER
    # Mencegah duplikasi mata uang dasar (e.g. jgn EURUSD & EURJPY bersamaan)
    candidates.sort(key=lambda x: x.get("scalp_power", 0), reverse=True)
    
    final_candidates = []
    used_bases = set()
    
    for c in candidates:
        base = c["kode"][:3]
        if base in used_bases and not force:
            continue
            
        if force or c["scalp_power"] > 70:
            final_candidates.append(c)
            used_bases.add(base)
            
        if len(final_candidates) >= 3:
            break
    
    logger.info(f"[AUTOSCALP] ✅ Ditemukan {len(final_candidates)} top class scalping (Exposure Filter Active).")
    return final_candidates


