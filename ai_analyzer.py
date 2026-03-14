"""
ai_analyzer.py - Groq (Primary) + Gemini (Fallback) dengan Buy/Sell Recommendation (v2.0)
==========================================================================================
Arsitektur:
  1. Groq LLaMA-3.3-70B (Primary, 30 RPM gratis, < 1 detik)
  2. Gemini 2.0-Flash (Fallback, 15 RPM)
  3. Cache 30 menit per saham
  4. AI memberikan: sentimen + rekomendasi BUY/HOLD/SELL + alasan
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

import config

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# CLIENT INITIALIZATION
# ----------------------------------------------------------------
_groq_client = None
_gemini_client = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY belum diset di environment variables")
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY belum diset di environment variables")
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


# ----------------------------------------------------------------
# SENTIMENT + RECOMMENDATION CACHE (Database Migration v6.0)
# ----------------------------------------------------------------
from db_manager import get_cached_sentiment, save_cached_sentiment


# ----------------------------------------------------------------
# PROMPT SISTEM (v3.0 - Ahli Forex & Makro Global)
# ----------------------------------------------------------------
SYSTEM_PROMPT = """Kamu adalah analis Forex & Makro Ekonomi Global senior yang ahli dalam pergerakan pasangan mata uang (Major & Cross) serta korelasi instrumen seperti Gold (XAUUSD) dan DXY (Dollar Index).

Tugasmu: Analisa gabungan berita + data teknikal untuk menghasilkan rekomendasi trading forex yang akurat.

ATURAN KETAT:
- Jawab HANYA dengan format JSON yang valid.
- 'sentimen' HANYA boleh: "Bullish", "Bearish", atau "Neutral".
- 'rekomendasi' HANYA boleh: "STRONG BUY", "BUY", "HOLD", "SELL", atau "STRONG SELL".
- 'alasan_singkat': maksimal 3 kalimat Bahasa Indonesia yang jelas dan actionable.
- 'skor_keyakinan': angka 1-10.
- 'faktor_risiko': sebutkan 1-2 risiko utama.

FORMAT JSON WAJIB (ikuti persis, tidak ada teks di luar JSON):
{
  "sentimen": "Bullish",
  "rekomendasi": "BUY",
  "alasan_singkat": "Penjelasan 2-3 kalimat yang actionable",
  "skor_keyakinan": 8,
  "kata_kunci": ["kata1", "kata2"],
  "faktor_risiko": "Risiko utama yang perlu diperhatikan"
}

PANDUAN REKOMENDASI:
- STRONG BUY: Fundamental sangat positif + teknikal semua hijau
- BUY: Sentimen positif/neutral + teknikal mendukung
- HOLD: Sinyal campur atau dirasa masih tunggu konfirmasi  
- SELL: Berita negatif atau teknikal melemah
- STRONG SELL: Berita sangat buruk atau teknikal breakdown"""

AUTOSCALP_SYSTEM_PROMPT = """Kamu adalah Prop Trader Forex Kuantitatif spesialis Scalping/Intraday.
Tugasmu: Diberikan data berita Makro Ekonomi Global (terutama terkait USD/Fed/ECB dll) dan 1-3 kandidat pair forex dengan momentum volatilitas sempurna. Pilih SATU pair paling berpotensi untuk discalping hari ini.

ATURAN KETAT:
- WAJIB output dalam JSON.
- Evaluasi pengaruh berita Makro terhadap pasar forex hari ini (Risk On / Risk Off).
- Jika ada rilis data "Red Folder" (NFP, CPI, Fed Rate) yang bertentangan tajam dengan setup, coret kandidat tersebut!

FORMAT JSON WAJIB:
{
  "market_view": "Analisa singkat cuaca market forex (DXY/Risk Sentiment) hari ini berdasarkan berita makro (1 kalimat)",
  "pemenang_kode": "EURUSD=X",
  "pemenang_nama": "Euro / US Dollar",
  "alasan_menang": "Alasan solid mengapa pair ini menang vs kandidat lain (2 kalimat, sebutkan efek volatilitas/news)",
  "trading_plan": {
    "entry_area": "Area harga masuk yang aman (kisaran presisi)",
    "target_1": "Target take profit awal (+10 hingga +20 Pips)",
    "target_2": "Target take profit maksimal (+30 hingga +50 Pips)",
    "stop_loss": "Angka cut loss ketat (-10 hingga -15 Pips)"
  },
  "pesan_psikologi": "Satu pesan disiplin singkat untuk trader"
}"""


def _build_prompt(kode: str, headlines: list[str], tech_context: dict | None = None) -> str:
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    ctx = ""
    if tech_context:
        score = tech_context.get("technical_score", 0)
        rsi = tech_context.get("rsi", 0)
        uptrend = tech_context.get("uptrend_daily", False)
        bb_squeeze = tech_context.get("bb_squeeze", False)
        bb_breakout = tech_context.get("bb_breakout", False)
        ctx = (
            f"\n\nDATA TEKNIKAL SAAT INI:\n"
            f"- Technical Score: {score}/100\n"
            f"- RSI: {rsi:.1f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Normal'})\n"
            f"- Trend Harian: {'📈 UPTREND' if uptrend else '📉 DOWNTREND/SIDEWAYS'}\n"
            f"- Bollinger Bands: {'🔥 BREAKOUT dari Squeeze!' if bb_breakout else '⚡ SQUEEZE (energi terkumpul)' if bb_squeeze else 'Normal'}\n"
        )

    return (
        f"Analisa instrumen {kode} berdasarkan {len(headlines)} headline berita makro/fundamental:{ctx}\n\n"
        f"HEADLINE BERITA:\n{numbered}\n\n"
        f"Berikan analisa dan rekomendasi trading dalam format JSON."
    )


# ----------------------------------------------------------------
# GROQ INFERENCE
# ----------------------------------------------------------------
def _analyze_groq(kode: str, headlines: list[str], tech_ctx: dict | None) -> dict | None:
    if not config.GROQ_API_KEY:
        logger.warning("[AI] GROQ_API_KEY tidak diset — skip Groq")
        return None

    user_prompt = _build_prompt(kode, headlines, tech_ctx)

    for attempt in range(1, 3):
        try:
            logger.info(f"[AI] 🔵 Groq {config.GROQ_MODEL} → {kode} (try {attempt})")
            client = get_groq_client()
            chat = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.15,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            text = chat.choices[0].message.content.strip()
            result = _parse(text)
            logger.info(f"[AI] ✅ Groq → {kode}: {result['sentimen']} | {result['rekomendasi']} ({result['skor_keyakinan']}/10)")
            return result

        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ["rate", "429", "too many", "quota", "tokens"]):
                logger.warning(f"[AI] Groq rate limit try {attempt}/2, tunggu 10s")
                time.sleep(10)
            else:
                logger.error(f"[AI] Groq error try {attempt}/2: {e}")
                time.sleep(2)

    logger.warning("[AI] Groq gagal → pindah ke Gemini fallback")
    return None


# ----------------------------------------------------------------
# GEMINI INFERENCE (Fallback)
# ----------------------------------------------------------------
def _analyze_gemini(kode: str, headlines: list[str], tech_ctx: dict | None) -> dict | None:
    if not config.GEMINI_API_KEY:
        logger.warning("[AI] GEMINI_API_KEY tidak diset — skip Gemini")
        return None

    full_prompt = f"{SYSTEM_PROMPT}\n\n{_build_prompt(kode, headlines, tech_ctx)}"

    for attempt in range(1, 3):
        try:
            logger.info(f"[AI] 🟡 Gemini {config.GEMINI_MODEL} fallback → {kode} (try {attempt})")
            client = get_gemini_client()
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.15,
                    top_p=0.8,
                    max_output_tokens=600,
                ),
            )
            result = _parse(response.text.strip())
            logger.info(f"[AI] ✅ Gemini → {kode}: {result['sentimen']} | {result['rekomendasi']}")
            return result

        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ["rate", "429", "quota", "resource_exhausted"]):
                wait = 15 * attempt
                logger.warning(f"[AI] Gemini rate limit try {attempt}/2, tunggu {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"[AI] Gemini error try {attempt}/2: {e}")
                time.sleep(3)

    return None


# ----------------------------------------------------------------
# FUNGSI UTAMA
# ----------------------------------------------------------------
async def analyze_sentiment(
    kode_saham: str,
    headlines: list[str],
    tech_context: dict | None = None,
    max_retry: int = 3,
) -> dict:
    """
    Analisa sentimen + rekomendasi trading dengan double-layer AI + SQLite cache (v6.0).
    """
    kode = kode_saham.upper().replace(".JK", "")

    # 1. Cek cache DB (async)
    cached = await get_cached_sentiment(kode, config.SENTIMENT_CACHE_TTL_MINUTES)
    if cached is not None:
        return {**cached, "dari_cache": True}

    # 2. Tidak ada berita → Neutral
    if not headlines:
        logger.warning(f"[AI] Tidak ada berita untuk {kode}")
        return _neutral_result(kode, len(headlines), reason="no_news")

    loop = asyncio.get_event_loop()

    # 3. Groq (primary) - Jalan di Executor
    result = await loop.run_in_executor(None, _analyze_groq, kode, headlines, tech_context)

    # 4. Gemini (fallback) - Jalan di Executor
    if result is None:
        result = await loop.run_in_executor(None, _analyze_gemini, kode, headlines, tech_context)

    # 5. Semua gagal
    if result is None:
        logger.error(f"[AI] ❌ Semua AI provider gagal untuk {kode}")
        return _neutral_result(kode, len(headlines), reason="api_error")

    result["headlines_dianalisa"] = len(headlines)
    
    # 6. Simpan ke Cache DB (async)
    await save_cached_sentiment(kode, result)
    return result


def _neutral_result(kode: str, n_headlines: int, reason: str = "") -> dict:
    if reason == "no_news":
        alasan = "Tidak ada berita terbaru ditemukan. Keputusan berdasarkan teknikal saja."
    elif reason == "api_error":
        alasan = "Analisa AI tidak tersedia saat ini. Perhatikan indikator teknikal."
    else:
        alasan = "Tidak ada sinyal fundamental yang kuat. Pantau perkembangan berita."
    return {
        "sentimen": "Neutral",
        "rekomendasi": "HOLD",
        "alasan_singkat": alasan,
        "skor_keyakinan": 0 if reason == "api_error" else 3,
        "kata_kunci": [],
        "faktor_risiko": "Data fundamental tidak tersedia",
        "headlines_dianalisa": n_headlines,
    }


def is_signal_approved(sentiment_result: dict) -> bool:
    return sentiment_result.get("sentimen", "Neutral") in ("Bullish", "Neutral")


# ----------------------------------------------------------------
# AUTOSCALPING INFERENCE (v5.0 & v6.0)
# ----------------------------------------------------------------
async def analyze_autoscalping(candidates: list[dict], macro_news: list[str]) -> dict | None:
    """
    Kirim semua kandidat dan berita makro sekaligus ke Llama-3 (Groq) untuk dipilih pemenangnya.
    Fallbak ke Gemini jika Groq limit.
    """
    logger.info("[AI_SCALP] Memulai perumusan Trading Plan AutoScalping (Async)...")
    
    macro_text = "\n".join(f"- {n}" for n in macro_news) if macro_news else "Tidak ada berita makro terbaru."
    
    cand_text = ""
    for c in candidates:
        kode = c["kode"]
        harga = c["harga_terakhir"]
        pct = c["perubahan_pct"]
        vol = c["kondisi"]["volume"]["rasio"]
        rsi = c["kondisi"]["rsi"]["nilai"]
        score = c["technical_score"]
        bb = "BREAKOUT SQUEEZE" if c["kondisi"]["bollinger"]["breakout"] else "Normal"
        
# Ambil 3 headline berita khusus pair ini
        from news_scraper import get_news_for_forex
        stock_news = await get_news_for_forex(kode, max_articles=3)
        news_str = "\n  - ".join(stock_news) if stock_news else "Tidak ada berita fundamental spesifik instrumen ini."
        
        cand_text += f"\nKANDIDAT: {kode} (Harga: {harga:.5f} | Gerak: {pct:.2f}%)\n"
        cand_text += f"Teknikal: Score {score}/100, RSI {rsi:.1f}, Volume {vol:.1f}x rata-rata, BB {bb}\n"
        cand_text += f"Berita Instrumen/Fundamental:\n  - {news_str}\n"

    user_prompt = (
        f"KONDISI MAKRO GLOBAL SAAT INI (Fokus sentimen USD/Risk):\n{macro_text}\n\n"
        f"DATA KANDIDAT PAIR FOREX SCALPING:\n{cand_text}\n\n"
        f"Tugas: Tentukan 1 saham pemenang dan buatkan Trading Plan presisi. Output WAJIB JSON."
    )

    # Coba Groq
    if config.GROQ_API_KEY:
        try:
            client = get_groq_client()
            loop = asyncio.get_event_loop()
            chat = await loop.run_in_executor(None, lambda: client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": AUTOSCALP_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2, # Sedikit dinaikkan agar analitis
                max_tokens=800,
                response_format={"type": "json_object"},
            ))
            text = chat.choices[0].message.content.strip()
            return json.loads(text)
        except Exception as e:
            logger.error(f"[AI_SCALP] Groq error: {e}")
            
    # Coba Gemini
    if config.GEMINI_API_KEY:
        try:
            client = get_gemini_client()
            full_prompt = f"{AUTOSCALP_SYSTEM_PROMPT}\n\n{user_prompt}"
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.8,
                    max_output_tokens=800,
                    response_mime_type="application/json"
                ),
            ))
            return json.loads(response.text.strip())
        except Exception as e:
            logger.error(f"[AI_SCALP] Gemini error: {e}")

    logger.error("[AI_SCALP] Gagal mendapatkan Trading Plan dari semua AI.")
    return None


# ----------------------------------------------------------------
# RECOMMENDATION LABEL (Gabungan Score Teknikal + AI)
# ----------------------------------------------------------------
def get_final_recommendation(technical_score: int, sentiment_result: dict) -> dict:
    """
    Gabungkan score teknikal (0-100) + sentimen AI menjadi rekomendasi final.
    Teknikal bobot 60%, AI bobot 40%.
    """
    sentimen = sentiment_result.get("sentimen", "Neutral")
    ai_reko = sentiment_result.get("rekomendasi", "HOLD")

    # Konversi sentimen AI ke score numerik
    sentimen_score = {"Bullish": 100, "Neutral": 50, "Bearish": 0}.get(sentimen, 50)

    # Gabungkan (60% teknikal, 40% AI)
    final_score = (technical_score * 0.6) + (sentimen_score * 0.4)

    if final_score >= 80 and sentimen == "Bullish":
        label = "🚀 STRONG BUY"
        warna = "🟢"
    elif final_score >= 65:
        label = "✅ BUY"
        warna = "🟢"
    elif final_score >= 45:
        label = "⏳ HOLD"
        warna = "🟡"
    elif final_score >= 25:
        label = "⚠️ SELL"
        warna = "🔴"
    else:
        label = "🚨 STRONG SELL"
        warna = "🔴"

    return {
        "label": label,
        "warna": warna,
        "final_score": round(final_score, 1),
        "ai_rekomendasi": ai_reko,
    }


# ----------------------------------------------------------------
# JSON PARSER
# ----------------------------------------------------------------
def _parse(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        return _validate(json.loads(cleaned))
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if m:
        try:
            return _validate(json.loads(m.group()))
        except json.JSONDecodeError:
            pass
    tl = text.lower()
    sentimen = "Bullish" if "bullish" in tl else ("Bearish" if "bearish" in tl else "Neutral")
    reko = "BUY" if "buy" in tl else ("SELL" if "sell" in tl else "HOLD")
    return {"sentimen": sentimen, "rekomendasi": reko,
            "alasan_singkat": "Analisa (text fallback).", "skor_keyakinan": 3,
            "kata_kunci": [], "faktor_risiko": "Parsing manual"}


def _validate(data: dict) -> dict:
    valid_s = {"Bullish", "Bearish", "Neutral"}
    valid_r = {"STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"}
    s = data.get("sentimen", "Neutral")
    s = next((v for v in valid_s if v.lower() == s.lower()), "Neutral")
    r = data.get("rekomendasi", "HOLD").upper()
    r = next((v for v in valid_r if v == r), "HOLD")
    return {
        "sentimen": s,
        "rekomendasi": r,
        "alasan_singkat": str(data.get("alasan_singkat", "Tidak ada keterangan.")),
        "skor_keyakinan": max(0, min(10, int(data.get("skor_keyakinan", 5)))),
        "kata_kunci": data.get("kata_kunci", []),
        "faktor_risiko": str(data.get("faktor_risiko", "N/A")),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from news_scraper import get_news_for_forex
    hasil = asyncio.run(analyze_sentiment("EURUSD=X", ["ECB Keeps Rates Unchanged", "Euro Drops as Dollar Rallies"],
                               tech_context={"technical_score": 72, "rsi": 45.0, "uptrend_daily": True}))
    print(f"Sentimen: {hasil['sentimen']} | Reko: {hasil['rekomendasi']} | {hasil['skor_keyakinan']}/10")
    print(f"Alasan: {hasil['alasan_singkat']}")
