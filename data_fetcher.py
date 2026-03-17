"""
data_fetcher.py - Data & Indikator Kuantitatif (v8.0 MT5)
=========================================================
Perombakan total dari arsitektur lama (yfinance) ke Direct Market Access
menggunakan MetaTrader5 (MT5). Fokus pada kecepatan, stabilitas eksekusi,
serta implementasi strategi "Triple Screen" oleh Alexander Elder + VWAP & Vol Surge.
"""

import logging
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# PENGATURAN LOGGING
# ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# ----------------------------------------------------------------
# UTILITAS
# ----------------------------------------------------------------
def get_pip_multiplier(symbol: str) -> float:
    """Helper untuk rasio jarak PIP (Forex vs JPY)."""
    symbol_upper = symbol.upper()
    if "JPY" in symbol_upper:
        return 100.0  # 1 pip = 0.01
    elif "XAU" in symbol_upper or "GOLD" in symbol_upper:
        return 10.0   # Asumsi standar untuk XAUUSD (10 pips per dollar)
    return 10000.0    # 1 pip = 0.0001 (Mayoritas pair forex)


# ----------------------------------------------------------------
# FUNGSI DETEKSI PRICE ACTION (Candlestick Pattern)
# ----------------------------------------------------------------
def detect_candlestick_pattern(df: "pd.DataFrame") -> str:
    """
    Mendeteksi pola candlestick Price Action dari 2 candle terakhir di DataFrame.
    Menggunakan aritmatika OHLC murni—tidak memerlukan library tambahan.

    Pola yang Dideteksi:
    - BULLISH ENGULFING  : Candle hijau besar "menelan" seluruh candle merah sebelumnya.
    - BEARISH ENGULFING  : Candle merah besar "menelan" seluruh candle hijau sebelumnya.
    - HAMMER / PIN BAR   : Ekor bawah >= 2x body, body kecil di 2/3 atas candle.
    - SHOOTING STAR      : Ekor atas >= 2x body, body kecil di 1/3 bawah candle.
    - NEUTRAL            : Tidak ada pola signifikan.
    """
    if df is None or len(df) < 2:
        return "NEUTRAL"

    try:
        prev = df.iloc[-2]
        curr = df.iloc[-1]

        prev_open, prev_close = prev['open'], prev['close']
        curr_open, curr_close = curr['open'], curr['close']
        curr_high, curr_low = curr['high'], curr['low']

        curr_body       = abs(curr_close - curr_open)
        curr_body_low   = min(curr_open, curr_close)
        curr_body_high  = max(curr_open, curr_close)
        curr_range      = curr_high - curr_low
        lower_wick      = curr_body_low - curr_low
        upper_wick      = curr_high - curr_body_high

        # Hindari pembagian nol
        if curr_range == 0 or curr_body == 0:
            return "NEUTRAL"

        # ── BULLISH ENGULFING ──────────────────────────────────────────
        prev_was_bearish = prev_close < prev_open
        curr_is_bullish  = curr_close > curr_open
        if (prev_was_bearish and curr_is_bullish
                and curr_open  < prev_close
                and curr_close > prev_open):
            return "BULLISH ENGULFING"

        # ── BEARISH ENGULFING ──────────────────────────────────────────
        prev_was_bullish = prev_close > prev_open
        curr_is_bearish  = curr_close < curr_open
        if (prev_was_bullish and curr_is_bearish
                and curr_open  > prev_close
                and curr_close < prev_open):
            return "BEARISH ENGULFING"

        # ── HAMMER / BULLISH PIN BAR ───────────────────────────────────
        # Ekor bawah panjang >= 2x body, body kecil di 2/3 atas candle
        body_position_from_low = curr_body_low - curr_low  # = lower_wick
        if (lower_wick >= 2.0 * curr_body
                and body_position_from_low >= (curr_range * 0.5)):
            return "HAMMER / PIN BAR (BULLISH)"

        # ── SHOOTING STAR / BEARISH PIN BAR ───────────────────────────
        # Ekor atas panjang >= 2x body, body kecil di 1/3 bawah candle
        body_position_from_high = curr_high - curr_body_high  # = upper_wick
        if (upper_wick >= 2.0 * curr_body
                and body_position_from_high >= (curr_range * 0.5)):
            return "SHOOTING STAR / PIN BAR (BEARISH)"

    except Exception:
        return "NEUTRAL"

    return "NEUTRAL"


# ----------------------------------------------------------------
# FUNGSI BARU v9.0: Synthetic DXY Strength
# ----------------------------------------------------------------
def get_synthetic_dxy_trend(num_candles: int = 20) -> str:
    """
    Menghitung arah USD Dollar Index secara sintetis dari 4 pair korelatif.
    Menggunakan hanya 20 candle M15 agar proses seringan mungkin.

    Logika:
    - Jika EMA5 Close > EMA20 Close  → pair itu Uptrend
    - EURUSD/GBPUSD Uptrend → USD LEMAH (bearish untuk DXY)
    - USDJPY/USDCHF Uptrend → USD KUAT (bullish untuk DXY)

    Returns:
        str: "STRONG" (USD strength), "WEAK", atau "NEUTRAL"
    """
    # Pair correlated dan bobot arahnya terhadap kekuatan USD
    # True  = jika pair uptrend, berarti USD KUAT
    # False = jika pair uptrend, berarti USD LEMAH
    DXY_PAIRS = [
        ("EURUSD", False),
        ("GBPUSD", False),
        ("USDJPY", True),
        ("USDCHF", True),
    ]

    bullish_usd_votes = 0
    valid_pairs = 0

    for pair_symbol, usd_bullish_when_up in DXY_PAIRS:
        try:
            # Penarikan data minimalis: hanya 20 candle M15, kolom close saja
            rates = mt5.copy_rates_from_pos(pair_symbol, mt5.TIMEFRAME_M15, 0, num_candles)
            if rates is None or len(rates) < num_candles:
                continue  # Pair tidak tersedia / data kurang

            closes = pd.Series([r['close'] for r in rates], dtype=float)
            ema5  = closes.ewm(span=5,  adjust=False).mean().iloc[-1]
            ema20 = closes.ewm(span=20, adjust=False).mean().iloc[-1]

            is_uptrend = ema5 > ema20
            valid_pairs += 1

            if (is_uptrend and usd_bullish_when_up) or (not is_uptrend and not usd_bullish_when_up):
                bullish_usd_votes += 1

        except Exception as e:
            logger.debug(f"[DXY] Gagal tarik data {pair_symbol}: {e}")
            continue

    if valid_pairs == 0:
        return "UNKNOWN"

    ratio = bullish_usd_votes / valid_pairs
    if ratio >= 0.75:
        return "STRONG"    # 3 atau 4 dari 4 pair konfirmasi USD menguat
    elif ratio <= 0.25:
        return "WEAK"      # 3 atau 4 dari 4 pair konfirmasi USD melemah
    else:
        return "NEUTRAL"   # Mixed signal


# ----------------------------------------------------------------
# FUNGSI BARU v9.0: Fair Value Gap (FVG) Detector
# ----------------------------------------------------------------
def detect_fvg(df: pd.DataFrame, num_candles: int = 15) -> dict:
    """
    Mendeteksi Fair Value Gap (FVG) dari N candle terakhir pada sebuah DataFrame.
    FVG adalah 'jejak kaki' Smart Money berupa celah harga yang belum terisi.

    Logika:
    - Bearish FVG: High candle[i] < Low candle[i-2]  → gap bearish di atas harga
    - Bullish FVG: Low candle[i]  > High candle[i-2] → gap bullish di bawah harga

    Hanya FVG terdekat dan belum 'dimitigasi' (harga belum kembali menutupnya)
    yang dilaporkan—inilah zona paling magnetis bagi harga.
    """
    result = {
        "fvg_bearish_unmitigated": False,
        "fvg_bearish_zone": None,
        "fvg_bullish_unmitigated": False,
        "fvg_bullish_zone": None,
    }

    if df is None or len(df) < 3:
        return result

    try:
        # Scan hanya N candle terbaru untuk kecepatan
        subset = df.tail(num_candles).reset_index(drop=True)
        current_price = subset['close'].iloc[-1]
        latest_bearish_fvg = None
        latest_bullish_fvg = None

        for i in range(2, len(subset)):
            h_i   = subset['high'].iloc[i]
            l_i   = subset['low'].iloc[i]
            h_i2  = subset['high'].iloc[i - 2]
            l_i2  = subset['low'].iloc[i - 2]

            # Bearish FVG: candle ke-3 seluruhnya di bawah candle ke-1
            if h_i < l_i2:
                zone_top = l_i2   # Batas bawah candle ke-1
                zone_bot = h_i    # Batas atas candle ke-3
                # Unmitigated jika harga saat ini BELUM menyentuh zona gap
                if current_price > zone_top:
                    latest_bearish_fvg = (zone_top, zone_bot)

            # Bullish FVG: candle ke-3 seluruhnya di atas candle ke-1
            if l_i > h_i2:
                zone_top = l_i    # Batas bawah candle ke-3
                zone_bot = h_i2   # Batas atas candle ke-1
                # Unmitigated jika harga saat ini BELUM menyentuh zona gap
                if current_price < zone_top:
                    latest_bullish_fvg = (zone_top, zone_bot)

        if latest_bearish_fvg:
            result["fvg_bearish_unmitigated"] = True
            result["fvg_bearish_zone"] = f"{round(latest_bearish_fvg[0], 5)} - {round(latest_bearish_fvg[1], 5)}"

        if latest_bullish_fvg:
            result["fvg_bullish_unmitigated"] = True
            result["fvg_bullish_zone"] = f"{round(latest_bullish_fvg[0], 5)} - {round(latest_bullish_fvg[1], 5)}"

    except Exception as e:
        logger.debug(f"[FVG] Error deteksi FVG: {e}")

    return result


# ----------------------------------------------------------------
# FUNGSI BARU v9.0: Order Block (OB) Detector — Bonus Weapon
# ----------------------------------------------------------------
def detect_order_block(df: pd.DataFrame) -> dict:
    """
    Mendeteksi Order Block (OB) — zona konsolidasi institusi sebelum impulse kuat.
    Logika:
    - Bullish OB: Candle bearish (merah) besar, langsung disusul oleh kenaikan impulse.
      Harga sering kembali ke body candle merah ini sebelum rally lebih jauh.
    - Bearish OB: Candle bullish (hijau) besar, langsung disusul oleh penurunan impulse.
    Scan 10 candle terakhir. Hanya cari OB yang masih 'di zona' (belum dilewati).
    """
    result = {
        "ob_bullish_zone": None,
        "ob_bearish_zone": None,
    }

    if df is None or len(df) < 5:
        return result

    try:
        subset = df.tail(10).reset_index(drop=True)
        current_price = subset['close'].iloc[-1]

        for i in range(1, len(subset) - 1):
            o, c, h, l = (subset['open'].iloc[i], subset['close'].iloc[i],
                          subset['high'].iloc[i], subset['low'].iloc[i])
            body = abs(c - o)

            # Ambang batas: body candle harus cukup signifikan
            avg_body = abs(subset['close'] - subset['open']).mean()
            if body < avg_body * 1.2:
                continue

            next_close = subset['close'].iloc[i + 1]

            # Bullish OB: candle BEARISH disusul candle bullish (next > open OB)
            if c < o and next_close > o:
                ob_top = o   # atas body candle bearish
                ob_bot = c   # bawah body candle bearish
                if current_price > ob_bot:   # Belum terlewati ke bawah
                    result["ob_bullish_zone"] = f"{round(ob_bot, 5)} - {round(ob_top, 5)}"

            # Bearish OB: candle BULLISH disusul candle bearish (next < open OB)
            if c > o and next_close < o:
                ob_top = c   # atas body candle bullish
                ob_bot = o   # bawah body candle bullish
                if current_price < ob_top:   # Belum terlewati ke atas
                    result["ob_bearish_zone"] = f"{round(ob_bot, 5)} - {round(ob_top, 5)}"

    except Exception as e:
        logger.debug(f"[OB] Error deteksi Order Block: {e}")

    return result


def get_mt5_data(symbol: str, timeframe: int, num_candles: int = 100) -> pd.DataFrame:
    """
    Mengambil data historical (OHLCV) langung dari MT5 terminal yang terhubung.
    Operasi ini thread-safe secara natural pada level C API python-MT5 untuk READ.
    
    Args:
        symbol (str): Nama pair atau instrumen, misal 'EURUSD', 'XAUUSDm'.
        timeframe (int): Konstan dari mt5, misal mt5.TIMEFRAME_H1.
        num_candles (int): Jumlah bar/candle ke belakang.
        
    Returns:
        pd.DataFrame: DataFrame berisi Open, High, Low, Close, Tick Volume, 
                      atau empty DataFrame jika gagal.
    """
    try:
        # Menarik data dari candle yang baru tutup / berjalan (start_pos = 0)
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
        
        if rates is None or len(rates) == 0:
            logger.error(f"[MT5] Tidak dapat mengambil data rates untuk {symbol} pada timeframe {timeframe}.")
            return pd.DataFrame()
            
        # Konversi array of tuples (records) ke Pandas DataFrame
        df = pd.DataFrame(rates)
        
        # Ekstrak waktu dan konversi (Unix seconds to datetime timezone-aware)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

        # Memastikan kolom-kolom yang diminta saja yang kita pakai untuk efisiensi
        if 'tick_volume' not in df.columns and 'real_volume' in df.columns:
            df.rename(columns={'real_volume': 'tick_volume'}, inplace=True)

        # Return mapping Open, High, Low, Close, Tick Volume sesuai requirements
        df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
        
        # Pastikan data diurutkan dari terlama ke terbaru
        df.sort_values(by='time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df

    except Exception as e:
        logger.error(f"[MT5] Exception saat mengambil data {symbol}: {e}")
        return pd.DataFrame()


# ----------------------------------------------------------------
# FUNGSI UTAMA 2: Analisis Kuantitatif Triple Screen
# ----------------------------------------------------------------
def analyze_triple_screen(symbol: str) -> dict:
    """
    Otak Kuantitatif: Menerapkan Alexander Elder's Triple Screen System
    + VWAP Harian HFT Style + Volume Surge + Risk Management Dinamis.
    
    Args:
        symbol (str): Instrumen trading MT5.
        
    Returns:
        dict: Eksekutif summary dari layar-layar analisis beserta risk-reward, 
              (kosong / dict default jika gagal).
    """
    logger.info(f"[TRIPLE SCREEN] Memulai analisis untuk {symbol}")
    
    # Init default failed result
    failed_result = {
        "symbol": symbol,
        "valid": False,
        "error": "Gagal fetch data / data tidak mencukupi"
    }

    # Tarik 3 Timeframe (H1, M15, M1) masing-masing 100 candle
    # Untuk H1 di-pull 250 agar EMA 200 bisa dikalkulasikan dengan valid.
    df_h1 = get_mt5_data(symbol, mt5.TIMEFRAME_H1, 250) 
    df_m15 = get_mt5_data(symbol, mt5.TIMEFRAME_M15, 100)
    df_m1 = get_mt5_data(symbol, mt5.TIMEFRAME_M1, 100)
    
    if df_h1.empty or df_m15.empty or df_m1.empty:
        return failed_result

    try:
        current_price = df_m1['close'].iloc[-1]
        
        # ==========================================================
        # LAYER 1 (H1) - MACRO TREND (Arah Arus Utama)
        # ==========================================================
        df_h1['EMA_50'] = ta.trend.ema_indicator(df_h1['close'], window=50, fillna=False)
        df_h1['EMA_200'] = ta.trend.ema_indicator(df_h1['close'], window=200, fillna=False)
        
        last_h1 = df_h1.iloc[-1]
        c_h1 = last_h1['close']
        ema50_h1 = last_h1['EMA_50']
        ema200_h1 = last_h1['EMA_200']
        
        trend = "SIDEWAYS"
        if pd.notna(ema50_h1) and pd.notna(ema200_h1):
            if c_h1 > ema50_h1 and ema50_h1 > ema200_h1:
                trend = "BULLISH"
            elif c_h1 < ema50_h1 and ema50_h1 < ema200_h1:
                trend = "BEARISH"

        # [v9.0] Synthetic DXY Strength — cek apakah ada konflik makro
        dxy_trend = get_synthetic_dxy_trend(num_candles=20)
        is_gold = any(kw in symbol.upper() for kw in ["XAU", "GOLD"])
        dxy_conflict = is_gold and dxy_trend == "STRONG"
        if dxy_conflict:
            logger.warning(f"[v9.0] DXY CONFLICT aktif! DXY={dxy_trend}, {symbol} adalah Emas → BUY DIBLOKIR oleh Makro!")

        # ==========================================================
        # LAYER 2 (M15) - MOMENTUM & VWAP (Diskon Harga)
        # ==========================================================
        df_m15['RSI_14'] = ta.momentum.rsi(df_m15['close'], window=14, fillna=False)
        last_rsi_14 = df_m15['RSI_14'].iloc[-1]
        
        # VWAP Harian (Intraday VWAP)
        df_m15['date'] = df_m15['time'].dt.date
        df_m15['typical_price'] = (df_m15['high'] + df_m15['low'] + df_m15['close']) / 3
        df_m15['vol_x_typ'] = df_m15['typical_price'] * df_m15['tick_volume']
        
        groupby_date = df_m15.groupby('date')
        df_m15['cum_vol_typ'] = groupby_date['vol_x_typ'].cumsum()
        df_m15['cum_vol'] = groupby_date['tick_volume'].cumsum()
        df_m15['VWAP'] = df_m15['cum_vol_typ'] / df_m15['cum_vol']
        
        last_m15 = df_m15.iloc[-1]
        c_m15 = last_m15['close']
        vwap_m15 = last_m15['VWAP']
        # Jarak persentase harga Close M15 saat ini terhadap VWAP
        vwap_distance_pct = ((c_m15 - vwap_m15) / vwap_m15) * 100 if vwap_m15 else 0
        
        vwap_status = "DISCOUNT (Below VWAP)" if c_m15 < vwap_m15 else "PREMIUM (Above VWAP)"

        # ==========================================================
        # SUPPORT & RESISTANCE M15 (Donchian Channel 50 Candle)
        # ==========================================================
        # Menggunakan 50 candle terakhir (tidak termasuk candle terkini)
        lookback = min(50, len(df_m15) - 1)
        resistance_m15 = float(df_m15['high'].iloc[-lookback-1:-1].max())
        support_m15    = float(df_m15['low'].iloc[-lookback-1:-1].min())
        
        pip_multiplier_sr = get_pip_multiplier(symbol)
        dist_to_resistance_pips = (resistance_m15 - c_m15) * pip_multiplier_sr
        dist_to_support_pips    = (c_m15 - support_m15) * pip_multiplier_sr
        
        # Status posisi harga terhadap S/R
        s_r_zone_pct = 0.05  # Dalam 5% dari range = zona S/R
        sr_range = resistance_m15 - support_m15
        sr_status = "MID RANGE"
        if sr_range > 0:
            if (c_m15 - support_m15) / sr_range <= s_r_zone_pct:
                sr_status = "AT SUPPORT (Potential Bounce)"
            elif (resistance_m15 - c_m15) / sr_range <= s_r_zone_pct:
                sr_status = "AT RESISTANCE (Potential Rejection)"

        # Deteksi Price Action di M15
        pa_m15 = detect_candlestick_pattern(df_m15)

        # [v9.0] Smart Money Concepts: FVG + Order Block
        fvg_data = detect_fvg(df_m15, num_candles=15)
        ob_data  = detect_order_block(df_m15)
        smart_money = {**fvg_data, **ob_data}

        # ==========================================================
        # LAYER 3 (M1) - EXECUTION TRIGGER (Pelatuk Entri)
        # ==========================================================
        last_m1 = df_m1.iloc[-1]
        current_m1_vol = last_m1['tick_volume']
        
        # Rata-rata 10 candle terakhir sebelum candle terkini
        avg_vol_10 = df_m1['tick_volume'].iloc[-11:-1].mean() if len(df_m1) > 10 else df_m1['tick_volume'].mean()
        
        # Deteksi *Volume Surge*
        is_volume_surge = current_m1_vol > (2.5 * avg_vol_10) if avg_vol_10 > 0 else False
        
        # ATR(14)
        df_m1['ATR_14'] = ta.volatility.average_true_range(
            high=df_m1['high'],
            low=df_m1['low'],
            close=df_m1['close'],
            window=14,
            fillna=False
        )
        last_atr_14 = df_m1['ATR_14'].iloc[-1]

        # Deteksi Price Action di M1
        pa_m1 = detect_candlestick_pattern(df_m1)
        
        # ==========================================================
        # KALKULASI PIPS & RISK/REWARD (1:1.5)
        # ==========================================================
        pip_multiplier = get_pip_multiplier(symbol)
        
        ATR_MULTIPLIER = 2.5
        base_atr = last_atr_14 if pd.notna(last_atr_14) else (current_price * 0.001)
        sl_raw_distance = base_atr * ATR_MULTIPLIER
        
        # Perhitungan matematis risk di hitung sebagai pips
        sl_pips = sl_raw_distance * pip_multiplier
        tp_pips = sl_pips * 1.5
        
        # Hitung Kedua Skenario (Long & Short)
        long_sl = current_price - sl_raw_distance
        long_tp = current_price + (sl_raw_distance * 1.5)
        
        short_sl = current_price + sl_raw_distance
        short_tp = current_price - (sl_raw_distance * 1.5)
            
        # ==========================================================
        # RANGKUMAN (DICTIONARY UNTUK LLM)
        # ==========================================================
        executive_summary = {
            "symbol": symbol,
            "valid": True,
            "current_price": round(current_price, 5),
            "macro_h1": {
                "trend": trend,
                "close": round(c_h1, 5),
                "ema_50": round(ema50_h1, 5) if pd.notna(ema50_h1) else None,
                "ema_200": round(ema200_h1, 5) if pd.notna(ema200_h1) else None,
                "dxy_trend": dxy_trend,
                "dxy_conflict": dxy_conflict
            },
            "momentum_m15": {
                "rsi_14": round(last_rsi_14, 2) if pd.notna(last_rsi_14) else None,
                "vwap_daily": round(vwap_m15, 5) if pd.notna(vwap_m15) else None,
                "vwap_status": vwap_status,
                "vwap_distance_pct": round(vwap_distance_pct, 4),
                "price_action": pa_m15,
                "support_m15": round(support_m15, 5),
                "resistance_m15": round(resistance_m15, 5),
                "dist_to_support_pips": round(dist_to_support_pips, 1),
                "dist_to_resistance_pips": round(dist_to_resistance_pips, 1),
                "sr_status": sr_status,
                "smart_money": smart_money
            },
            "execution_m1": {
                "volume_surge_detected": bool(is_volume_surge),
                "current_tick_volume": int(current_m1_vol),
                "average_tick_volume_10": round(avg_vol_10, 2),
                "atr_14": round(last_atr_14, 6) if pd.notna(last_atr_14) else None,
                "price_action": pa_m1
            },
            "risk_management": {
                "raw_atr_distance_pips": round(sl_pips, 2),
                "scenario_long": {
                    "entry": round(current_price, 5),
                    "sl": round(long_sl, 5),
                    "tp": round(long_tp, 5)
                },
                "scenario_short": {
                    "entry": round(current_price, 5),
                    "sl": round(short_sl, 5),
                    "tp": round(short_tp, 5)
                }
            }
        }
        
        logger.info(
            f"[TRIPLE SCREEN] Analisis sukses untuk {symbol}. "
            f"Trend: {trend} | DXY: {dxy_trend} | Conflict: {dxy_conflict} | "
            f"FVG Bear: {fvg_data['fvg_bearish_unmitigated']} | FVG Bull: {fvg_data['fvg_bullish_unmitigated']} | "
            f"Vol Surge: {is_volume_surge}"
        )
        return executive_summary
        
    except Exception as e:
        logger.error(f"[TRIPLE SCREEN] Runtime error kalkulasi untuk {symbol}: {e}")
        return failed_result

if __name__ == "__main__":
    # Test call - Jangan panggil jika tidak terkoneksi ke MT5
    # Pastikan MetaTrader5 terinisialisasi
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
        
    res = analyze_triple_screen("EURUSD")
    print("Hasil Analisis:", res)
    mt5.shutdown()
