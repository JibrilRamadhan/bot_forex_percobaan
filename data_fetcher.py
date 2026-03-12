"""
data_fetcher.py - Pengambil Data & Kalkulasi Indikator Teknikal
===============================================================
Modul ini bertanggung jawab untuk:
1. Mengambil data OHLCV (Open, High, Low, Close, Volume) dari yfinance.
2. Menghitung semua indikator teknikal menggunakan pandas_ta.
3. Mendeteksi kondisi sinyal (crossover, volume surge, RSI).
4. Menghitung level Support & Resistance menggunakan Pivot Point Klasik.
"""

import time
import logging
import pandas as pd
import ta
import yfinance as yf
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# FUNGSI UTILITAS
# ----------------------------------------------------------------
def format_ticker(kode_saham: str) -> str:
    """
    Memastikan kode saham memiliki suffix '.JK' untuk pasar Indonesia.
    
    Contoh:
        'INET'    -> 'INET.JK'
        'INET.JK' -> 'INET.JK' (sudah benar, tidak berubah)
        'inet'    -> 'INET.JK' (case-insensitive)
    """
    kode_saham = kode_saham.strip().upper()
    if not kode_saham.endswith(".JK"):
        kode_saham = f"{kode_saham}.JK"
    return kode_saham


def get_clean_code(ticker: str) -> str:
    """Mengembalikan kode saham bersih tanpa suffix .JK (misal: 'INET.JK' -> 'INET')."""
    return ticker.replace(".JK", "").upper()


# ----------------------------------------------------------------
# PENGAMBILAN DATA DARI YFINANCE
# ----------------------------------------------------------------
def fetch_ohlcv(
    kode_saham: str,
    interval: str = config.YFINANCE_INTERVAL,
    period: str = config.YFINANCE_PERIOD,
    max_retry: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Mengambil data OHLCV dari yfinance dengan penanganan error dan retry.

    Args:
        kode_saham: Kode saham (dengan atau tanpa .JK).
        interval: Interval candlestick, default '15m'.
        period: Periode data, default '5d'.
        max_retry: Jumlah percobaan ulang jika gagal.

    Returns:
        DataFrame pandas dengan data OHLCV, atau None jika gagal.
    """
    ticker = format_ticker(kode_saham)

    for attempt in range(1, max_retry + 1):
        try:
            logger.info(f"[FETCHER] Mengambil data {ticker} (percobaan {attempt}/{max_retry})")
            df = yf.download(
                ticker,
                interval=interval,
                period=period,
                progress=False,   # Nonaktifkan progress bar
                auto_adjust=True, # Sesuaikan harga untuk dividen/split
            )

            # Periksa apakah DataFrame kosong
            if df is None or df.empty:
                logger.warning(f"[FETCHER] Data kosong untuk {ticker} pada percobaan {attempt}")
                if attempt < max_retry:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue

            # Flatten MultiIndex columns jika ada (yfinance kadang menghasilkan ini)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            logger.info(f"[FETCHER] ✅ Berhasil mengambil {len(df)} candle untuk {ticker}")
            return df

        except Exception as e:
            logger.error(f"[FETCHER] Error pada percobaan {attempt} untuk {ticker}: {e}")
            if attempt < max_retry:
                time.sleep(2 ** attempt)

    logger.error(f"[FETCHER] ❌ Gagal mengambil data untuk {ticker} setelah {max_retry} percobaan")
    return None


def fetch_info(kode_saham: str) -> dict:
    """
    Mengambil informasi umum saham (nama perusahaan, sektor, dll.).
    
    Returns:
        Dictionary berisi info saham, atau dict kosong jika gagal.
    """
    ticker = format_ticker(kode_saham)
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "nama_perusahaan": info.get("longName", get_clean_code(ticker)),
            "sektor": info.get("sector", "N/A"),
            "industri": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "currency": info.get("currency", "IDR"),
        }
    except Exception as e:
        logger.warning(f"[FETCHER] Gagal mengambil info untuk {ticker}: {e}")
        return {"nama_perusahaan": get_clean_code(ticker), "sektor": "N/A"}


# ----------------------------------------------------------------
# KALKULASI INDIKATOR TEKNIKAL
# ----------------------------------------------------------------
def calculate_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Menghitung semua indikator teknikal yang dibutuhkan menggunakan library ta.

    Indikator yang dihitung:
        - EMA_5 & EMA_13 (Exponential Moving Average)
        - RSI_14 (Relative Strength Index)
        - SMA_Volume_20 (Simple Moving Average volume)

    Args:
        df: DataFrame OHLCV dari yfinance.

    Returns:
        DataFrame yang sudah ditambahkan kolom indikator, atau None jika gagal.
    """
    try:
        # Pastikan cukup data untuk kalkulasi
        min_rows_needed = max(config.EMA_SLOW, config.RSI_PERIOD, config.VOLUME_SMA) * 2
        if len(df) < min_rows_needed:
            logger.warning(f"[INDICATOR] Data tidak cukup: {len(df)} baris (butuh minimal {min_rows_needed})")
            return None

        df = df.copy()

        # Pastikan nama kolom standar (lowercase)
        df.columns = [col.lower() for col in df.columns]

        close = df["close"]
        volume = df["volume"]

        # --- EMA Cepat & Lambat (menggunakan library ta) ---
        df[f"EMA_{config.EMA_FAST}"] = ta.trend.EMAIndicator(
            close=close, window=config.EMA_FAST, fillna=False
        ).ema_indicator()

        df[f"EMA_{config.EMA_SLOW}"] = ta.trend.EMAIndicator(
            close=close, window=config.EMA_SLOW, fillna=False
        ).ema_indicator()

        # --- RSI (menggunakan library ta) ---
        df[f"RSI_{config.RSI_PERIOD}"] = ta.momentum.RSIIndicator(
            close=close, window=config.RSI_PERIOD, fillna=False
        ).rsi()

        # --- Volume SMA (rolling mean via pandas — lebih ringan) ---
        df[f"VOL_SMA_{config.VOLUME_SMA}"] = volume.rolling(window=config.VOLUME_SMA).mean()

        # Hapus baris dengan nilai NaN (akibat perhitungan rolling)
        df.dropna(inplace=True)

        if df.empty:
            logger.warning("[INDICATOR] DataFrame kosong setelah dropna()")
            return None

        logger.info(f"[INDICATOR] ✅ Indikator berhasil dihitung, {len(df)} candle valid.")
        return df

    except Exception as e:
        logger.error(f"[INDICATOR] Error saat kalkulasi indikator: {e}")
        return None


# ----------------------------------------------------------------
# DETEKSI SINYAL TEKNIKAL
# ----------------------------------------------------------------
def detect_signal(df: pd.DataFrame) -> dict:
    """
    Mendeteksi kondisi sinyal beli (HAKA/Bullish Crossover) pada candle terakhir.

    Kondisi SEMUANYA harus terpenuhi untuk sinyal VALID:
        1. EMA_5 crossover ke atas EMA_13 (candle sebelumnya EMA_5 < EMA_13,
           candle terbaru EMA_5 > EMA_13).
        2. Volume candle terakhir > (SMA_Volume_20 × 2.0).
        3. RSI_14 < 70 (belum overbought).

    Returns:
        Dictionary berisi status sinyal dan detail setiap kondisi.
    """
    col_ema_fast = f"EMA_{config.EMA_FAST}"
    col_ema_slow = f"EMA_{config.EMA_SLOW}"
    col_rsi = f"RSI_{config.RSI_PERIOD}"
    col_vol_sma = f"VOL_SMA_{config.VOLUME_SMA}"

    # Ambil 2 candle terakhir untuk deteksi crossover
    row_prev = df.iloc[-2]  # Candle sebelumnya
    row_last = df.iloc[-1]  # Candle terakhir (terkini)

    # --- Kondisi 1: EMA Crossover ---
    ema_was_below = row_prev[col_ema_fast] < row_prev[col_ema_slow]
    ema_now_above = row_last[col_ema_fast] > row_last[col_ema_slow]
    is_crossover = ema_was_below and ema_now_above

    # --- Kondisi 2: Volume Surge ---
    vol_current = row_last["volume"]
    vol_sma = row_last[col_vol_sma]
    vol_ratio = vol_current / vol_sma if vol_sma > 0 else 0
    is_volume_surge = vol_ratio >= config.VOLUME_SURGE_MULTIPLIER

    # --- Kondisi 3: RSI Aman (tidak overbought) ---
    rsi_value = row_last[col_rsi]
    is_rsi_safe = rsi_value < config.RSI_OVERBOUGHT

    # --- Sinyal Valid Hanya Jika Semua Kondisi Terpenuhi ---
    is_valid_signal = is_crossover and is_volume_surge and is_rsi_safe

    return {
        "sinyal_valid": is_valid_signal,
        "harga_terakhir": round(float(row_last["close"]), 2),
        "kondisi": {
            "crossover": {
                "status": is_crossover,
                "ema_fast_sebelum": round(float(row_prev[col_ema_fast]), 2),
                "ema_slow_sebelum": round(float(row_prev[col_ema_slow]), 2),
                "ema_fast_sekarang": round(float(row_last[col_ema_fast]), 2),
                "ema_slow_sekarang": round(float(row_last[col_ema_slow]), 2),
            },
            "volume": {
                "status": is_volume_surge,
                "volume_sekarang": int(vol_current),
                "volume_sma": round(float(vol_sma), 0),
                "rasio": round(vol_ratio, 2),
                "threshold": config.VOLUME_SURGE_MULTIPLIER,
            },
            "rsi": {
                "status": is_rsi_safe,
                "nilai": round(float(rsi_value), 2),
                "batas_overbought": config.RSI_OVERBOUGHT,
            },
        },
    }


# ----------------------------------------------------------------
# PERHITUNGAN SUPPORT & RESISTANCE (PIVOT POINT KLASIK)
# ----------------------------------------------------------------
def calculate_pivot_points(df: pd.DataFrame) -> dict:
    """
    Menghitung Support dan Resistance menggunakan metode Pivot Point Klasik.
    
    Rumus Pivot Point Klasik (berdasarkan candle sebelumnya):
        PP = (High + Low + Close) / 3
        R1 = (2 × PP) - Low
        R2 = PP + (High - Low)
        S1 = (2 × PP) - High
        S2 = PP - (High - Low)

    Args:
        df: DataFrame OHLCV yang sudah dihitung indikatornya.

    Returns:
        Dictionary berisi nilai PP, R1, R2, S1, S2.
    """
    try:
        # Gunakan data dari hari sebelumnya (daily) untuk perhitungan pivot
        # Ambil dari data 15m, konversi ke daily untuk akurasi lebih baik
        df_daily = df.resample("D", on=df.index.name if df.index.name else None).agg(
            {"high": "max", "low": "min", "close": "last"}
        ).dropna()

        # Jika tidak bisa resample, gunakan candle terakhir sebagai referensi
        if df_daily.empty or len(df_daily) < 2:
            prev = df.iloc[-2]
        else:
            prev = df_daily.iloc[-2]

        high = float(prev["high"])
        low = float(prev["low"])
        close = float(prev["close"])

        pp = (high + low + close) / 3
        r1 = (2 * pp) - low
        r2 = pp + (high - low)
        s1 = (2 * pp) - high
        s2 = pp - (high - low)

        return {
            "PP": round(pp, 2),
            "R1": round(r1, 2),
            "R2": round(r2, 2),
            "S1": round(s1, 2),
            "S2": round(s2, 2),
        }
    except Exception as e:
        logger.warning(f"[PIVOT] Error kalkulasi pivot point: {e}")
        return {"PP": 0, "R1": 0, "R2": 0, "S1": 0, "S2": 0}


# ----------------------------------------------------------------
# FUNGSI LENGKAP: SCREENING SAHAM (dipakai oleh command /screening)
# ----------------------------------------------------------------
def full_screening(kode_saham: str) -> Optional[dict]:
    """
    Fungsi screening lengkap untuk satu saham.
    Menggabungkan fetch data, kalkulasi indikator, deteksi sinyal, dan pivot.
    
    Returns:
        Dictionary berisi semua data screening, atau None jika gagal.
    """
    ticker = format_ticker(kode_saham)
    kode_bersih = get_clean_code(kode_saham)

    # 1. Ambil data OHLCV
    df = fetch_ohlcv(ticker)
    if df is None:
        return None

    # 2. Hitung indikator
    df = calculate_indicators(df)
    if df is None:
        return None

    # 3. Deteksi sinyal
    sinyal = detect_signal(df)

    # 4. Hitung pivot point
    pivots = calculate_pivot_points(df)

    # 5. Hitung perubahan harga (%) dibanding candle sebelumnya
    harga_sekarang = float(df.iloc[-1]["close"])
    harga_kemarin = float(df.iloc[-2]["close"])
    perubahan_pct = ((harga_sekarang - harga_kemarin) / harga_kemarin) * 100

    # 6. Ambil info perusahaan
    info = fetch_info(ticker)

    return {
        "ticker": ticker,
        "kode": kode_bersih,
        "nama_perusahaan": info.get("nama_perusahaan", kode_bersih),
        "harga_terakhir": harga_sekarang,
        "perubahan_pct": round(perubahan_pct, 2),
        "pivot_points": pivots,
        **sinyal,  # Gabungkan semua data sinyal
    }


if __name__ == "__main__":
    # Test modul secara standalone
    logging.basicConfig(level=logging.INFO)
    hasil = full_screening("INET")
    if hasil:
        print(f"Saham: {hasil['kode']}")
        print(f"Harga: Rp {hasil['harga_terakhir']:,.0f} ({hasil['perubahan_pct']:+.2f}%)")
        print(f"Sinyal Valid: {hasil['sinyal_valid']}")
        print(f"Pivot: {hasil['pivot_points']}")
