"""
ai_analyzer.py - Groq (Primary) + Gemini (Fallback) dengan Circuit Breaker (v8.0 MT5)
=====================================================================================
Arsitektur:
  1. Groq LLaMA-3.3-70B (Primary, 30 RPM gratis, < 1 detik)
  2. Gemini 2.0-Flash (Fallback otomatis + Failover 60 menit)
  3. Circuit Breaker: 3 gagal berturut-turut → Groq diblokir 60 menit
  4. Cache 30 menit per instrumen di SQLite (Opsional/Tergantung db_manager)
"""

import json
import logging
import time
import re
import asyncio
from datetime import datetime, timedelta

from groq import Groq
from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

import config

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# CLIENT INITIALIZATION
# ----------------------------------------------------------------
_groq_clients: list[Groq] = []
_active_groq_index = 0
_gemini_client = None

def get_groq_client() -> Groq:
    global _groq_clients, _active_groq_index
    if not _groq_clients:
        if not config.GROQ_API_KEYS:
            raise ValueError("GROQ_API_KEYS belum diset di environment variables")
        for key in config.GROQ_API_KEYS:
            if key.strip():
                _groq_clients.append(Groq(api_key=key.strip()))
        if not _groq_clients:
             raise ValueError("GROQ_API_KEYS kosong/tidak valid")
    return _groq_clients[_active_groq_index]

def rotate_groq_client() -> bool:
    """Memutar ke API Key Groq berikutnya jika terdeteksi Token Limit / 429."""
    global _active_groq_index, _groq_clients
    if len(_groq_clients) <= 1:
        return False
    _active_groq_index = (_active_groq_index + 1) % len(_groq_clients)
    logger.warning(f"[CIRCUIT] 🔄 Memutar Groq API Key ke antrian #{_active_groq_index + 1}/{len(_groq_clients)}.")
    return True


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY belum diset di environment variables")
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


from typing import Dict, Any

# ----------------------------------------------------------------
# CIRCUIT BREAKER STATE
# ----------------------------------------------------------------
_circuit_state: Dict[str, Any] = {
    "groq_failure_count": 0,
    "groq_disabled_until": None
}
_GROQ_FAILURE_THRESHOLD = 3
_GROQ_COOLDOWN_MINUTES = 60


def _is_groq_circuit_open() -> bool:
    """Return True jika Groq sedang dalam kondisi diblokir (circuit open)."""
    disabled_until = _circuit_state["groq_disabled_until"]
    if disabled_until is None:
        return False
    if datetime.now() >= disabled_until: # type: ignore
        # Cooldown selesai — reset circuit
        logger.info("[CIRCUIT] ✅ Groq circuit CLOSED kembali setelah cooldown.")
        _reset_groq_circuit()
        return False
    return True


def _record_groq_failure():
    """Catat satu kegagalan Groq dan buka circuit jika threshold tercapai."""
    _circuit_state["groq_failure_count"] += 1
    count = _circuit_state["groq_failure_count"]
    logger.warning(f"[CIRCUIT] ⚡ Groq failure #{count}/{_GROQ_FAILURE_THRESHOLD}")
    if count >= _GROQ_FAILURE_THRESHOLD: # type: ignore
        _circuit_state["groq_disabled_until"] = datetime.now() + timedelta(minutes=_GROQ_COOLDOWN_MINUTES)
        logger.error(
            f"[CIRCUIT] 🔴 Groq circuit OPEN! Diblokir {_GROQ_COOLDOWN_MINUTES} menit "
            f"hingga {_circuit_state['groq_disabled_until'].strftime('%H:%M:%S')}. Semua request → Gemini." # type: ignore
        )


def _record_groq_success():
    """Reset failure counter saat Groq berhasil."""
    if _circuit_state["groq_failure_count"] > 0: # type: ignore
        _circuit_state["groq_failure_count"] = 0


def _reset_groq_circuit():
    _circuit_state["groq_failure_count"] = 0
    _circuit_state["groq_disabled_until"] = None


# ----------------------------------------------------------------
# SENTIMENT + RECOMMENDATION CACHE (DIPINDAHKAN)
# ----------------------------------------------------------------
# Operasi database cache dipindahkan ke pemanggil (signal_engine/bot.py)
# untuk mencegah SQLite database locks & RuntimeError.


# ----------------------------------------------------------------
# PROMPT SISTEM (v8.0 MT5 Triple Screen)
# ----------------------------------------------------------------
TRIPLE_SCREEN_PROMPT = """Kamu adalah Institutional Quant Trader senior untuk Hedge Fund.
Tugasmu adalah menganalisa sinyal 'Triple Screen' (Alexander Elder) dikombinasikan dengan VWAP harian, dinamika Volume Surge (M1), Support/Resistance, Price Action Candlestick, Korelasi Makro DXY, Smart Money Concepts (FVG), serta berita makro, untuk mengambil keputusan trading skala institusi.

ATURAN PENGAMBILAN KEPUTUSAN (STRICT):
1. Jika H1 Trend BULLISH, kamu HANYA BOLEH mencari peluang LONG (BUY). Entri terbaik adalah ketika harga berada di DISCOUNT (di bawah VWAP M15).
2. Jika H1 Trend BEARISH, kamu HANYA BOLEH mencari peluang SHORT (SELL). Entri terbaik adalah ketika harga berada di PREMIUM (di atas VWAP M15).
3. KORELASI DXY (US Dollar Index): Jika DXY STRONG, instrumen yang memiliki USD sebagai arah sebaliknya (seperti XAUUSD / Emas, EURUSD) pasti tertekan turun, DILARANG BUY. Jika DXY WEAK, Emas/EURUSD berpotensi meroket. Evaluasilah korelasi wajar antara jenis instrumen dan DXY.
4. SMART MONEY CONCEPTS (FVG): Perhatikan area FVG dan Order Block yang unmitigated sebagai magnet harga. Jika terdapat FVG kosong di bawah harga saat ini, hindari melakukan BUY karena harga dapat tersedot turun untuk menutupi gap tersebut (liquidity grab).
5. Jika H1 SIDEWAYS, kamu WAJIB melihat arah berita (News) dan DXY. Jika News memberikan katalis kuat, kamu boleh melakukan counter-trend trading. Jika News tidak ada atau ambigu, rekomendasikan WAIT (HOLD).
6. Untuk rekomendasi tingkat "STRONG" (STRONG BUY / STRONG SELL), "Volume Surge Detected" di time-frame M1 adalah SYARAT MUTLAK (Wajib True). Jika False, maksimal rekomendasi hanyalah BUY / SELL biasa.
7. EXCEPTION untuk aturan 6: Jika terdeteksi pola Price Action pembalikan (seperti BULLISH ENGULFING, HAMMER/PIN BAR) tepat di area "AT SUPPORT" atau menyentuh "ORDER BLOCK BULLISH", kamu BOLEH memberikan STRONG BUY meski Volume Surge False. Sebaliknya untuk BEARISH ENGULFING atau SHOOTING STAR di area "AT RESISTANCE" atau "ORDER BLOCK BEARISH" → STRONG SELL.

ATURAN MANAJEMEN POSISI v11.2 (HYPER-AGGRESSIVE SCALPING & REVERSAL EXIT):
- Kamu kini punya mata untuk melihat "STATUS OPEN POSITIONS SAAT INI" (Jumlah layer dan Total Profit).
- 🛑 ANTI-HEDGING (STRICT BAN): Jika ada posisi BUY terbuka, kamu HARAM merekomendasikan SELL. Begitupun jika ada posisi SELL terbuka, kamu HARAM merekomendasikan BUY. Mesin akan menge-block tindakanmu jika kamu nekat Hedging.
- ⚡ EARLY REVERSAL EXIT (PENTING!): Jika kamu sedang memegang posisi (misal BUY), lalu tiba-tiba terdeteksi pola Price Action pembalikan (seperti BEARISH PIN BAR / ENGULFING di M1) atau harga gagal menembus VWAP, JANGAN TUNGGU TREND H1 BERBALIK! Segera keluarkan "CLOSE_ALL_LONG" untuk mengamankan profit sekecil apapun atau memotong loss secepatnya. Begitupun untuk posisi SELL, gunakan "CLOSE_ALL_SHORT".
- Boleh merekomendasikan "STRONG BUY" meski sudah ada posisi BUY terbuka maksimal 10 Layer (Aggressive Layering). Setara untuk SELL. Target terminal adalah Hit and Run scalping yang cepat.

ATURAN OUTPUT:
- Jawab HANYA dengan format JSON yang valid. Jangan tambahkan teks apa pun di luar JSON.

FORMAT JSON WAJIB:
{
  "arah_trading": "LONG" | "SHORT" | "WAIT" | "CLOSE",
  "rekomendasi": "STRONG BUY" | "BUY" | "SELL" | "STRONG SELL" | "HOLD" | "CLOSE_ALL_LONG" | "CLOSE_ALL_SHORT",
  "alasan_singkat": "Maksimal 3 kalimat. Sebutkan konfirmasi H1, DXY, FVG, posisi VWAP M15, dan status Open Positions.",
  "skor_keyakinan": 8,
  "faktor_risiko": "Risiko utama dari trade ini"
}
"""


def _build_triple_screen_prompt(symbol: str, headlines: list[str], ts_data: dict) -> str:
    """Merakit dictionary eksekutif dari data_fetcher MT5 menjadi string untuk LLM."""
    numbered_news = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines)) if headlines else "Tidak ada berita makro terbaru."
    
    # Ekstrak data
    h1 = ts_data.get("macro_h1", {})
    m15 = ts_data.get("momentum_m15", {})
    m1 = ts_data.get("execution_m1", {})
    
    fvg = m15.get("fvg_data", {})
    op = ts_data.get("open_positions", {})
    tg = ts_data.get("absolute_trend_guard", {})
    
    ctx = (
        f"STATUS OPEN POSITIONS SAAT INI:\n"
        f"   - Total Layers: {op.get('total_positions', 0)} (Buy: {op.get('total_buy', 0)}, Sell: {op.get('total_sell', 0)})\n"
        f"   - Total Floating Profit: {op.get('total_profit_pips', 0.0)} pips\n\n"
        f"ABSOLUTE TREND GUARD (MESIN):\n"
        f"   - Izin Buka LONG: {'Ya' if tg.get('is_allowed_long', False) else 'TIDAK (DILARANG KERAS)'}\n"
        f"   - Izin Buka SHORT: {'Ya' if tg.get('is_allowed_short', False) else 'TIDAK (DILARANG KERAS)'}\n\n"
        f"SIMPULAN DATA KUANTITATIF (MT5 TRIPLE SCREEN):\n"
        f"1. MACRO TREND (H1):\n"
        f"   - Trend Status: {h1.get('trend', 'UNKNOWN')}\n"
        f"   - Synthetic DXY Status: {h1.get('dxy_status', 'UNKNOWN')}\n"
        f"   - Close Price: {h1.get('close', 'N/A')}\n"
        f"   - EMA 50 / 200: {h1.get('ema_50', 'N/A')} / {h1.get('ema_200', 'N/A')}\n"
        f"2. MOMENTUM, VWAP & SMART MONEY (M15):\n"
        f"   - VWAP Status: {m15.get('vwap_status', 'UNKNOWN')} ({m15.get('vwap_distance_pct', 0)}%)\n"
        f"   - RSI 14: {m15.get('rsi_14', 'N/A')}\n"
        f"   - Support M15: {m15.get('support_m15', 'N/A')} | Resistance M15: {m15.get('resistance_m15', 'N/A')}\n"
        f"   - Posisi thd S/R: {m15.get('sr_status', 'N/A')} (Dist to Support: {m15.get('dist_to_support_pips', 'N/A')} pips, Dist to Resistance: {m15.get('dist_to_resistance_pips', 'N/A')} pips)\n"
        f"   - Price Action M15: {m15.get('price_action', 'NEUTRAL')}\n"
        f"   - Unmitigated FVG M15: {fvg.get('fvg_type', 'NONE')} (Top: {fvg.get('fvg_top', 'N/A')}, Bottom: {fvg.get('fvg_bottom', 'N/A')})\n"
        f"   - Order Block M15: Top: {fvg.get('order_block_top', 'N/A')} | Bottom: {fvg.get('order_block_bottom', 'N/A')}\n"
        f"3. EXECUTION TRIGGER (M1):\n"
        f"   - Volume Surge Detected: {m1.get('volume_surge_detected', False)}\n"
        f"   - Current Tick Vol vs Avg 10: {m1.get('current_tick_volume', 0)} vs {m1.get('average_tick_volume_10', 0)}\n"
        f"   - Price Action M1: {m1.get('price_action', 'NEUTRAL')}\n\n"
    )
    
    return (
        f"Analisa instrumen {symbol} berdasarkan gabungan data kuantitatif dan headline berita.\n\n"
        f"{ctx}"
        f"HEADLINE BERITA:\n{numbered_news}\n\n"
        f"Sebagai Institutional Quant Trader, tentukan keputusan trading dalam format JSON baku."
    )


# ----------------------------------------------------------------
# GROQ INFERENCE — With Circuit Breaker + tenacity Retry
# ----------------------------------------------------------------
class GroqRateLimitError(Exception):
    pass


class GroqAPIError(Exception):
    pass


def _call_groq_api(messages: list, max_tokens: int = 600, response_format: dict | None = None) -> str:
    """
    Raw Groq API call. Raises specific exceptions untuk retry logic.
    """
    client = get_groq_client()
    kwargs = {
        "model": config.GROQ_MODEL,
        "messages": messages,
        "temperature": 0.15,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


@retry(
    retry=retry_if_exception_type(GroqRateLimitError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=False,
)
def _groq_with_retry(messages: list, max_tokens: int = 600, response_format: dict | None = None) -> str | None:
    """Groq call dengan rotasi kunci untuk token exhaust / rate limit."""
    try:
        return _call_groq_api(messages, max_tokens, response_format)
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ["rate", "429", "too many", "quota", "tokens", "limit"]):
            logger.warning(f"[CIRCUIT] Groq token/rate limit error: {e}")
            if rotate_groq_client():
                raise GroqRateLimitError(f"Memutar kunci karena: {e}")
            else:
                logger.warning("[CIRCUIT] Tidak ada kunci Groq cadangan yang tersisa.")
                raise GroqRateLimitError(str(e))
        # Error lain
        raise GroqAPIError(str(e)) from e


def _analyze_mt5_groq(symbol: str, headlines: list[str], ts_data: dict) -> dict | None:
    """
    Groq inference untuk MT5 Triple Screen.
    """
    if not config.GROQ_API_KEYS:
        return None

    if _is_groq_circuit_open():
        logger.warning(f"[CIRCUIT] 🔴 Groq circuit OPEN, skip untuk {symbol} → Gemini")
        return None

    messages = [
        {"role": "system", "content": TRIPLE_SCREEN_PROMPT},
        {"role": "user", "content": _build_triple_screen_prompt(symbol, headlines, ts_data)},
    ]

    try:
        logger.info(f"[AI] 🔵 Groq {config.GROQ_MODEL} → {symbol}")
        text = _groq_with_retry(messages, max_tokens=600, response_format={"type": "json_object"})
        if text is None:
            _record_groq_failure()
            return None
        result = _parse_mt5_json(text)
        _record_groq_success()
        logger.info(f"[AI] ✅ Groq → {symbol}: {result['arah_trading']} | {result['rekomendasi']} ({result['skor_keyakinan']}/10)")
        return result
    except (RetryError, GroqRateLimitError):
        logger.warning(f"[CIRCUIT] Groq habis retry untuk {symbol}")
        _record_groq_failure()
    except Exception as e:
        logger.error(f"[AI] Groq non-retryable error untuk {symbol}: {e}")
        _record_groq_failure()

    return None


# ----------------------------------------------------------------
# GEMINI INFERENCE (Fallback)
# ----------------------------------------------------------------
def _analyze_mt5_gemini(symbol: str, headlines: list[str], ts_data: dict) -> dict | None:
    if not config.GEMINI_API_KEY:
        logger.warning("[AI] GEMINI_API_KEY tidak diset — skip Gemini")
        return None

    full_prompt = f"{TRIPLE_SCREEN_PROMPT}\n\n{_build_triple_screen_prompt(symbol, headlines, ts_data)}"

    for attempt in range(1, 4):
        try:
            logger.info(f"[AI] 🟡 Gemini {config.GEMINI_MODEL} fallback → {symbol} (try {attempt})")
            client = get_gemini_client()
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.15,
                    top_p=0.8,
                    max_output_tokens=600,
                    response_mime_type="application/json"
                ),
            )
            result = _parse_mt5_json(response.text.strip())
            logger.info(f"[AI] ✅ Gemini → {symbol}: {result['arah_trading']} | {result['rekomendasi']}")
            return result

        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ["rate", "429", "quota", "resource_exhausted"]):
                wait = 15 * attempt
                logger.warning(f"[AI] Gemini rate limit try {attempt}/3, tunggu {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"[AI] Gemini error try {attempt}/3: {e}")
                time.sleep(3)

    return None


# ----------------------------------------------------------------
# FUNGSI UTAMA (MT5 ANALYZER)
# ----------------------------------------------------------------
async def analyze_mt5_signal(
    symbol: str,
    headlines: list[str],
    ts_data: dict,
) -> dict:
    """
    Analisa Triple Screen MT5 + Berita menggunakan Double-Layer AI.
    (Operasi caching dipindah ke fungsi pemanggil di signal_engine/bot.py)
    """
    loop = asyncio.get_event_loop()

    # 1. Groq (primary) - Jalan di Executor
    result = await loop.run_in_executor(None, _analyze_mt5_groq, symbol, headlines, ts_data)

    # 2. Gemini (fallback) - Jalan di Executor
    if result is None:
        result = await loop.run_in_executor(None, _analyze_mt5_gemini, symbol, headlines, ts_data)

    # 3. Semua gagal
    if result is None:
        logger.error(f"[AI] ❌ Semua AI provider gagal untuk {symbol}")
        result = _mt5_neutral_result(symbol, headlines)

    result["headlines_dianalisa"] = len(headlines)
    
    if "risk_management" in ts_data:
        result["risk_management"] = ts_data["risk_management"]
        result["current_price"] = ts_data.get("current_price")
    
    return result


def _mt5_neutral_result(symbol: str, headlines: list) -> dict:
    return {
        "arah_trading": "WAIT",
        "rekomendasi": "HOLD",
        "alasan_singkat": "Analisis AI gagal atau API Limit tercapai. Harap perhatikan data teknikal H1/M15 secara manual.",
        "skor_keyakinan": 0,
        "faktor_risiko": "Analisis terputus",
        "headlines_dianalisa": len(headlines)
    }


# ----------------------------------------------------------------
# JSON PARSER (MT5 FORMAT)
# ----------------------------------------------------------------
def _parse_mt5_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        return _validate_mt5_json(json.loads(cleaned))
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if m:
        try:
            return _validate_mt5_json(json.loads(m.group()))
        except json.JSONDecodeError:
            pass
            
    # Fallback primitif
    tl = text.lower()
    arah = "LONG" if "long" in tl or "buy" in tl else ("SHORT" if "short" in tl or "sell" in tl else "WAIT")
    reko = "BUY" if arah == "LONG" else ("SELL" if arah == "SHORT" else "HOLD")
    return {
        "arah_trading": arah,
        "rekomendasi": reko,
        "alasan_singkat": "Analisa (text fallback karena JSON rusak).",
        "skor_keyakinan": 3,
        "faktor_risiko": "Parsing manual"
    }


def _validate_mt5_json(data: dict) -> dict:
    valid_a = {"LONG", "SHORT", "WAIT", "CLOSE"}
    valid_r = {"STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL", "CLOSE_ALL_LONG", "CLOSE_ALL_SHORT"}
    
    a = data.get("arah_trading", "WAIT").upper()
    a = next((v for v in valid_a if v == a), "WAIT")
    
    r = data.get("rekomendasi", "HOLD").upper()
    r = next((v for v in valid_r if v == r), "HOLD")
    
    return {
        "arah_trading": a,
        "rekomendasi": r,
        "alasan_singkat": str(data.get("alasan_singkat", "Tidak ada keterangan.")),
        "skor_keyakinan": max(0, min(10, int(data.get("skor_keyakinan", 5)))),
        "faktor_risiko": str(data.get("faktor_risiko", "N/A")),
    }


# ----------------------------------------------------------------
# AUTOSCALPING INFERENCE (Legacy/Placeholder for MT5 CSM later)
# ----------------------------------------------------------------
# Note: Dibiarkan agar tidak mematahkan import modul lain jika masih digunakan.
# Bisa diarahkan ulang/diubah sesuai logika yang sama.

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Dummy Test Call
    dummy_ts_data = {
        "macro_h1": {"trend": "BULLISH", "dxy_status": "DXY STRONG", "close": 1.0850, "ema_50": 1.0840, "ema_200": 1.0800},
        "momentum_m15": {
            "vwap_status": "DISCOUNT (Below VWAP)", "vwap_distance_pct": -0.05, "rsi_14": 45,
            "fvg_data": {"fvg_type": "BEARISH", "fvg_top": 1.0860, "fvg_bottom": 1.0855, "order_block_top": 1.0870, "order_block_bottom": 1.0865}
        },
        "execution_m1": {"volume_surge_detected": True, "current_tick_volume": 550, "average_tick_volume_10": 120}
    }
    
    hasil = asyncio.run(analyze_mt5_signal("EURUSD", ["Dollar Melemah karena Data NFP Buruk"], dummy_ts_data))
    print(json.dumps(hasil, indent=2))
