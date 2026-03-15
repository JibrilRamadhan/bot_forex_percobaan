"""
bot.py - Forex Trading Bot & AI Screener v2.0
==========================================
Fitur:
- Auto Chart Generation untuk Forex
- Pip Calculation Risk Management (ATR)
- Macro Trend (1H/4H Correlation)
- Final Recommendation (STRONG BUY / BUY / HOLD / SELL / STRONG SELL)
- Peringatan Volatilitas DXY & Berita Red Folder
"""

import asyncio
import logging
import html
import io
import gc
import os
from datetime import datetime, time as dtime

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (wajib untuk server tanpa display)
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import pytz

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler,
    ContextTypes, CallbackQueryHandler,
)
from telegram.constants import ParseMode

import config
from config import validate_config, WIB
from data_fetcher import (
    full_screening, format_ticker, get_clean_code,
    scan_forex_buy, scan_forex_danger,
    get_market_leaders, get_autoscalping_candidates
)
from news_scraper import get_news_for_forex, get_macro_news, get_economic_calendar, is_kill_switch_active
from ai_analyzer import (
    analyze_sentiment, is_signal_approved, get_final_recommendation,
    analyze_autoscalping
)
import db_manager as db

# ----------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("mplfinance").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# EMOJI & LABEL
# ----------------------------------------------------------------
EMOJI = {
    "rocket": "🚀", "chart_up": "📈", "chart_down": "📉", "warning": "⚠️",
    "check": "✅", "cross": "❌", "target": "🎯", "fire": "🔥",
    "bell": "🔔", "news": "📰", "robot": "🤖", "clock": "🕐",
    "money": "💰", "radar": "📡", "star": "⭐", "info": "ℹ️",
    "bullish": "🟢", "bearish": "🔴", "neutral": "🟡",
    "shield": "🛡️", "lightning": "⚡", "chart": "📊", "search": "🔍",
}

SENTIMENT_EMOJI = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}
SENTIMENT_LABEL = {"Bullish": "BULLISH 📈", "Bearish": "BEARISH 📉", "Neutral": "NEUTRAL ➡️"}


def format_number(n: float) -> str:
    if n >= 1_000_000_000_000:
        return f"{n/1_000_000_000_000:.1f}T"
    elif n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}M"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.1f}jt"
    elif n >= 1_000:
        return f"{n/1_000:.1f}rb"
    return f"{n:.0f}"


# ----------------------------------------------------------------
# CHART GENERATOR
# ----------------------------------------------------------------
def generate_chart(df: pd.DataFrame, kode: str, screening_data: dict) -> io.BytesIO | None:
    """
    Generate candlestick chart PNG dengan EMA5, EMA13, Volume, RSI.
    Return: BytesIO buffer yang bisa langsung dikirim via Telegram.
    """
    try:
        col_ef = f"EMA_{config.EMA_FAST}"
        col_es = f"EMA_{config.EMA_SLOW}"
        col_rsi = f"RSI_{config.RSI_PERIOD}"

        # Ambil 50 candle terakhir
        df_chart = df.tail(50).copy()

        # Pastikan index adalah DatetimeIndex
        if not isinstance(df_chart.index, pd.DatetimeIndex):
            df_chart.index = pd.to_datetime(df_chart.index)

        # Rename kolom sesuai yang dibutuhkan mplfinance
        df_chart = df_chart.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })

        # Siapkan addplot
        addplots = []
        if col_ef in df_chart.columns:
            addplots.append(mpf.make_addplot(df_chart[col_ef], color="#00BFFF", width=1.5, label=f"EMA{config.EMA_FAST}"))
        if col_es in df_chart.columns:
            addplots.append(mpf.make_addplot(df_chart[col_es], color="#FF8C00", width=1.5, label=f"EMA{config.EMA_SLOW}"))
        if "BB_upper" in df_chart.columns:
            addplots.append(mpf.make_addplot(df_chart["BB_upper"], color="#8A2BE2", width=0.8, linestyle="--"))
            addplots.append(mpf.make_addplot(df_chart["BB_lower"], color="#8A2BE2", width=0.8, linestyle="--"))
        if col_rsi in df_chart.columns:
            addplots.append(mpf.make_addplot(df_chart[col_rsi], panel=2, color="#9370DB", ylabel="RSI",
                                             ylim=(0, 100)))

        # Style chart dark mode
        mc = mpf.make_marketcolors(up="#00C851", down="#FF4444",
                                    edge="inherit", wick="inherit",
                                    volume={"up": "#00C851", "down": "#FF4444"})
        s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style="nightclouds",
                                facecolor="#0D0D0D", figcolor="#0D0D0D",
                                gridcolor="#2A2A2A", gridstyle=":")

        tech_score = screening_data.get("technical_score", 0)
        harga = screening_data.get("harga_terakhir", 0)
        waktu = datetime.now(WIB).strftime("%d %b %Y %H:%M WIB")
        title = f"{kode} | {harga:.5f} | Score: {tech_score}/100 | {waktu}"

        buf = io.BytesIO()
        
        # v7.5 Horizontal Lines (SL/TP)
        h_lines = []
        h_colors = []
        risk_m = screening_data.get("risk_management", {})
        if risk_m.get("stop_loss"):
            h_lines.append(risk_m["stop_loss"])
            h_colors.append("#FF4444") # Red for SL
        if risk_m.get("target_price"):
            h_lines.append(risk_m["target_price"])
            h_colors.append("#00C851") # Green for TP
        if screening_data.get("harga_terakhir"):
            h_lines.append(screening_data["harga_terakhir"])
            h_colors.append("#33B5E5") # Cyan for Entry
            
        # v7.5 Fix: mplfinance sometimes rejects 'None' for addplot/hlines explicitly passed
        mpf_kwargs = {
            "type": "candle",
            "style": s,
            "title": title,
            "volume": True,
            "panel_ratios": (4, 1, 2) if col_rsi in df_chart.columns else (4, 1),
            "figsize": (12, 8),
            "returnfig": True
        }
        if addplots:
            mpf_kwargs["addplot"] = addplots
        if h_lines:
            mpf_kwargs["hlines"] = dict(hlines=h_lines, colors=h_colors, linestyle="-.", linewidths=1.0)
            
        fig, axes = mpf.plot(df_chart, **mpf_kwargs)
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor="#0D0D0D")

        # Bebaskan semua memori matplotlib segera setelah render
        # Penting untuk server free tier (RAM 500MB) agar tidak OOM
        plt.close(fig)       # Tutup figure spesifik ini
        plt.close("all")     # Tutup semua figure yang mungkin masih terbuka
        gc.collect()         # Paksa garbage collector bersihkan sisa memori

        buf.seek(0)
        logger.info(f"[CHART] ✅ Chart berhasil dibuat untuk {kode}")
        return buf

    except Exception as e:
        logger.error(f"[CHART] Gagal generate chart untuk {kode}: {e}")
        return None


# ----------------------------------------------------------------
# BUILD PESAN SCREENING
# ----------------------------------------------------------------
def build_screening_message(screening_data: dict, sentiment_data: dict, headlines: list[str]) -> str:
    kode = screening_data["kode"]
    nama = html.escape(screening_data.get("nama_perusahaan", kode))
    harga = screening_data["harga_terakhir"]
    perubahan = screening_data["perubahan_pct"]
    kondisi = screening_data["kondisi"]
    pivot = screening_data.get("pivot_points", {})
    risk = screening_data.get("risk_management", {})
    daily = screening_data.get("daily_trend", {})
    tech_score = screening_data.get("technical_score", 0)

    sentimen = sentiment_data.get("sentimen", "Neutral")
    alasan = html.escape(sentiment_data.get("alasan_singkat", ""))
    skor = sentiment_data.get("skor_keyakinan", 0)
    faktor_risiko = html.escape(sentiment_data.get("faktor_risiko", "N/A"))
    dari_cache = sentiment_data.get("dari_cache", False)

    # Final Recommendation
    reko_data = get_final_recommendation(tech_score, sentiment_data)
    reko_label = reko_data["label"]
    final_score = reko_data["final_score"]

    perubahan_emoji = EMOJI["chart_up"] if perubahan >= 0 else EMOJI["chart_down"]
    perubahan_str = f"{perubahan:+.2f}%"
    waktu_wib = datetime.now(WIB).strftime("%d %b %Y, %H:%M WIB")

    rsi_val = kondisi["rsi"]["nilai"]
    vol_ratio = kondisi["volume"]["rasio"]
    ema_fast = kondisi["crossover"]["ema_fast_sekarang"]
    ema_slow = kondisi["crossover"]["ema_slow_sekarang"]

    # Score bar visual
    filled = round(tech_score / 10)
    score_bar = "█" * filled + "░" * (10 - filled)
    score_color = "🟢" if tech_score >= 70 else ("🟡" if tech_score >= 45 else "🔴")

    # EMA status
    ema_up = ema_fast > ema_slow
    ema_status = f"{EMOJI['check']} EMA{config.EMA_FAST} di atas EMA{config.EMA_SLOW}" if ema_up else f"{EMOJI['cross']} EMA{config.EMA_FAST} di bawah EMA{config.EMA_SLOW}"

    # Volume
    vol_status = f"{EMOJI['fire']} SURGE (×{vol_ratio:.1f})" if kondisi["volume"]["status"] else f"{EMOJI['warning']} Normal (×{vol_ratio:.1f})"

    # RSI
    if rsi_val < 30:
        rsi_status = f"🔵 OVERSOLD ({rsi_val:.1f}) — Peluang Rebound"
    elif rsi_val > 70:
        rsi_status = f"🔴 OVERBOUGHT ({rsi_val:.1f}) — Hati-hati"
    else:
        rsi_status = f"🟡 Normal ({rsi_val:.1f})"

    # Bollinger Bands
    bb = kondisi.get("bollinger", {})
    if bb.get("breakout"):
        bb_status = f"{EMOJI['lightning']} <b>BREAKOUT dari Squeeze!</b> — Sinyal A++"
    elif bb.get("squeeze"):
        bb_status = f"⚡ Squeeze — Energi terkumpul, siap meledak"
    else:
        bb_status = f"📊 Normal"

    # Multi-timeframe
    mtf_uptrend = daily.get("uptrend_daily", False)
    pct_vs_ema20d = daily.get("harga_vs_ema20d", 0)
    mtf_status = (f"{EMOJI['chart_up']} UPTREND Makro ({pct_vs_ema20d:+.2f}% vs EMA20D) ✅"
                  if mtf_uptrend else
                  f"{EMOJI['chart_down']} Di bawah EMA20D Makro ({pct_vs_ema20d:+.2f}%) ⚠️ Hati-hati trend balik")

    # Stop Loss & Target (Pips)
    sl = risk.get("stop_loss", 0)
    tp = risk.get("target_price", 0)
    rr = risk.get("risk_reward", 0)
    atr = risk.get("atr", 0)
    sl_pips = risk.get("stop_pips", 0)
    lot = risk.get("recommended_lot", 0.01)
    sl_str = f"{sl:.5f} ({sl_pips:.1f} Pips)" if sl > 0 else "N/A"
    tp_str = f"{tp:.5f}" if tp > 0 else "N/A"
    rr_str = f"1 : {rr:.1f}" if rr > 0 else "N/A"

    # Berita
    berita_str = ""
    if headlines:
        bl = "\n".join(f"  • {html.escape(h[:85])}{'...' if len(h) > 85 else ''}" for h in headlines[:3])
        berita_str = f"\n{EMOJI['news']} <b>BERITA TERBARU</b>\n{bl}\n"

    cache_note = " <i>(cached)</i>" if dari_cache else ""

    msg = f"""
{EMOJI['radar']} <b>INFORMASI PAIR: {kode}</b> | <i>{nama}</i> | <i>by Jibril</i>
━━━━━━━━━━━━━━━━━━━━━━━━

{EMOJI['money']} Harga: <code>{harga:.5f}</code> {perubahan_emoji} {perubahan_str}

━━━━━━━━━━━━━━━━━━━━━━━━
{score_color} <b>TECHNICAL SCORE: {tech_score}/100</b>
<code>[{score_bar}]</code>

━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['chart']} <b>INDIKATOR TEKNIKAL</b>
━━━━━━━━━━━━━━━━━━━━━━━━
📉 {ema_status}
   └ EMA{config.EMA_FAST}: <code>{ema_fast:,.2f}</code> | EMA{config.EMA_SLOW}: <code>{ema_slow:,.2f}</code>

{EMOJI['chart']} Volume: {vol_status}
   └ Saat ini: <code>{format_number(kondisi['volume']['volume_sekarang'])}</code> | Avg: <code>{format_number(kondisi['volume']['volume_sma'])}</code>

💹 RSI: {rsi_status}

{EMOJI['lightning']} Bollinger Bands: {bb_status}

🌐 Multi-Timeframe: {mtf_status}

━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['robot']} <b>ANALISA AI</b>{cache_note}
━━━━━━━━━━━━━━━━━━━━━━━━
{SENTIMENT_EMOJI.get(sentimen, '⚪')} Sentimen: <b>{SENTIMENT_LABEL.get(sentimen, sentimen)}</b> ({skor}/10)
💬 <i>{alasan}</i>
⚠️ Risiko: <i>{faktor_risiko}</i>
{berita_str}
━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['shield']} <b>MANAJEMEN RISIKO</b>
━━━━━━━━━━━━━━━━━━━━━━━━
🛑 Stop Loss: <code>{sl_str}</code>
{EMOJI['target']} Target: <code>{tp_str}</code>
⚖️ Risk/Reward: <b>{rr_str}</b>
📦 Recommended Lot: <b>{lot:.2f}</b> (Risk 1%)

━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['target']} <b>SUPPORT & RESISTANCE</b>
━━━━━━━━━━━━━━━━━━━━━━━━
🔴 R2: <code>{pivot.get('R2', 0):.5f}</code> | 🟠 R1: <code>{pivot.get('R1', 0):.5f}</code>
⚪ PP: <code>{pivot.get('PP', 0):.5f}</code>
🟢 S1: <code>{pivot.get('S1', 0):.5f}</code> | 🔵 S2: <code>{pivot.get('S2', 0):.5f}</code>

━━━━━━━━━━━━━━━━━━━━━━━━
🏁 <b>REKOMENDASI FINAL:</b> {reko_label}
   Score Gabungan: <code>{final_score}/100</code>
━━━━━━━━━━━━━━━━━━━━━━━━

{EMOJI['clock']} {waktu_wib}
{EMOJI['info']} <i>⚠️ Bukan rekomendasi resmi. Selalu DYOR! by J</i>
""".strip()
    return msg


# ----------------------------------------------------------------
# BUILD PESAN ALERT RADAR
# ----------------------------------------------------------------
def build_signal_alert_message(screening_data: dict, sentiment_data: dict) -> str:
    kode = screening_data["kode"]
    nama = html.escape(screening_data.get("nama_perusahaan", kode))
    harga = screening_data["harga_terakhir"]
    perubahan = screening_data["perubahan_pct"]
    kondisi = screening_data["kondisi"]
    pivot = screening_data.get("pivot_points", {})
    risk = screening_data.get("risk_management", {})
    tech_score = screening_data.get("technical_score", 0)
    sentimen = sentiment_data.get("sentimen", "Neutral")
    alasan = html.escape(sentiment_data.get("alasan_singkat", ""))
    skor = sentiment_data.get("skor_keyakinan", 0)

    reko_data = get_final_recommendation(tech_score, sentiment_data)
    reko_label = reko_data["label"]

    perubahan_emoji = EMOJI["chart_up"] if perubahan >= 0 else EMOJI["chart_down"]
    waktu_wib = datetime.now(WIB).strftime("%H:%M WIB")

    rsi_val = kondisi["rsi"]["nilai"]
    vol_ratio = kondisi["volume"]["rasio"]
    bb = kondisi.get("bollinger", {})

    sl = risk.get("stop_loss", 0)
    tp = risk.get("target_price", 0)
    rr = risk.get("risk_reward", 0)

    # BB badge
    bb_badge = ""
    if bb.get("breakout"):
        bb_badge = f"\n{EMOJI['lightning']} <b>BB BREAKOUT dari SQUEEZE!</b> — Sinyal Super Kuat A++"

    msg = f"""
{EMOJI['bell']} <b>SINYAL RADAR TERDETEKSI!</b> {EMOJI['fire']}
━━━━━━━━━━━━━━━━━━━━━━━━

{EMOJI['chart_up']} <b>{kode}</b> | <i>{nama}</i>
{EMOJI['money']} Harga: <code>Rp {harga:,.0f}</code> {perubahan_emoji} {perubahan:+.2f}%
📊 Technical Score: <b>{tech_score}/100</b>{bb_badge}

━━━━━━━━━━━━━━━━━━━━━━━━
🔎 <b>TEKNIKAL</b>
━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['check']} EMA{config.EMA_FAST} crossover EMA{config.EMA_SLOW} ↗️
{EMOJI['fire']} Volume: SURGE ×{vol_ratio:.1f} di atas rata-rata
💹 RSI: {rsi_val:.1f} ({'Aman ✓' if rsi_val < 70 else '⚠️ Overbought'})

━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['robot']} <b>ANALISA AI</b>
━━━━━━━━━━━━━━━━━━━━━━━━
{SENTIMENT_EMOJI.get(sentimen, '⚪')} {SENTIMENT_LABEL.get(sentimen, sentimen)} | Keyakinan: {skor}/10
💬 <i>{alasan}</i>

━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['shield']} <b>MANAJEMEN RISIKO (PIPS)</b>
━━━━━━━━━━━━━━━━━━━━━━━━
🛑 Stop Loss: <code>{sl:.5f}</code> ({risk.get('risiko_pips', 0):.1f} Pips)
{EMOJI['target']} Target: <code>{tp:.5f}</code> ({risk.get('potensi_pips', 0):.1f} Pips)
⚖️ Risk/Reward: <b>1 : {rr:.1f}</b>

━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['target']} S/R | R1: <code>{pivot.get('R1',0):.5f}</code> | PP: <code>{pivot.get('PP',0):.5f}</code> | S1: <code>{pivot.get('S1',0):.5f}</code>

🏁 <b>REKOMENDASI: {reko_label}</b>

{EMOJI['clock']} {waktu_wib} | {EMOJI['info']} <i>DYOR! by J</i>
""".strip()
    return msg


# ----------------------------------------------------------------
# COMMAND HANDLERS
# ----------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    nama = user.first_name if user else "Trader"
    keyboard = [
        [
            InlineKeyboardButton("📡 Watchlist", callback_data="watchlist"),
            InlineKeyboardButton("❓ Panduan", callback_data="help")
        ],
        [
            InlineKeyboardButton("📊 EURUSD", callback_data="screen_EURUSD=X"),
            InlineKeyboardButton("📊 GBPUSD", callback_data="screen_GBPUSD=X"),
        ],
        [
            InlineKeyboardButton("📊 GBPJPY", callback_data="screen_GBPJPY=X"),
            InlineKeyboardButton("🥇 Gold (XAU)", callback_data="screen_GC=F"),
        ],
    ]
    pesan = f"""
{EMOJI['rocket']} <b>Halo, {html.escape(nama)}! Selamat datang di Forex AI v2.0 🔥</b>

{EMOJI['radar']} <b>Forex Daily Scalper & AI Screener v2.0</b>
     <i>by J — Dirancang untuk Trader Profesional</i>

━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['star']} <b>FITUR UNGGULAN</b>
━━━━━━━━━━━━━━━━━━━━━━━━
📊 Auto Chart Candlestick + EMA + RSI
🛡️ Risk Management Otomatis (Lot Size & Pips)
⚡ Volatility & Multi-Timeframe Checks
📅 Kalender Ekonomi Real-Time (Red Folder)
🤖 AI Trading Plan (Llama-3 + Gemini)
🏆 Currency Strength Meter (CSM)

━━━━━━━━━━━━━━━━━━━━━━━━
📋 <b>PERINTAH UTAMA</b>
━━━━━━━━━━━━━━━━━━━━━━━━
/autoscalping — 🤖 AI Trading Plan Setup Terbaik
/autoscalpingforce — ⚡ Paksa AI (High Risk)
/screening [PAIR] — 📊 Analisa teknikal + chart
/heatmap — 🌡️ Live CSM & kekuatan mata uang
/signals — 🎯 Top kandidat BUY hari ini
/calendar — 📅 Kalender Ekonomi & Red Folder
/winrate — 🏆 Win Rate & statistik bot
/danger — ⚠️ Pair berisiko tinggi
/watchlist — 📡 Daftar instrumen pantauan
/help — ❓ Panduan lengkap
""".strip()
    await update.effective_message.reply_text(pesan, parse_mode=ParseMode.HTML,
                                    reply_markup=InlineKeyboardMarkup(keyboard))


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pesan = f"""
{EMOJI['info']} <b>PANDUAN PENGGUNAAN FOREX BOT v2.0</b>
━━━━━━━━━━━━━━━━━━━━━━━━

{EMOJI['chart_up']} <b>Screening & Rekomendasi:</b>
<code>/autoscalping</code> — AI pilihkan 1 setup scalping terbaik!
<code>/autoscalpingforce</code> — Paksa AI jika Anda agresif.
<code>/screening PAIR</code> — Contoh: <code>EURUSD=X</code>, <code>GBPJPY=X</code>, <code>GC=F</code>
<code>/heatmap</code> — Currency Strength Meter Real-Time
<code>/signals</code> — Top sinyal BUY hari ini
<code>/calendar</code> — Kalender ekonomi & jadwal Red Folder
<code>/danger</code> — Hindari pair dengan spike berita

━━━━━━━━━━━━━━━━━━━━━━━━
📈 <b>Tracking Performa:</b>
<code>/winrate</code> — Statistik akurasi bot (WIN/LOSS)

━━━━━━━━━━━━━━━━━━━━━━━━
🛡️ <b>Manajemen Risiko (Otomatis):</b>
Stop Loss dihitung berdasarkan ATR × 1.5
Lot Size dihitung berdasarkan risiko 1% per trade

━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['warning']} <i>Bot ini adalah alat bantu analisa. BUKAN saran investasi resmi. Selalu DYOR! by J</i>
""".strip()
    await update.effective_message.reply_text(pesan, parse_mode=ParseMode.HTML)


async def cmd_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    n = len(config.FOREX_WATCHLIST)
    # Tampilkan 20 instrumen pertama sebagai preview
    preview = " | ".join(f"<code>{k}</code>" for k in config.FOREX_WATCHLIST[:20])
    pesan = f"""
{EMOJI['radar']} <b>WATCHLIST FOREX & COMMODITY</b>
━━━━━━━━━━━━━━━━━━━━━━━━

📡 Radar memantau <b>{n} instrumen</b> (Majors, Crosses, Gold, Oil)

<b>Preview:</b>
{preview}
<i>... dan {max(0, n - 20)} instrumen lainnya</i>

━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['clock']} Sesi: <b>Tokyo / London / New York</b>
{EMOJI['chart_up']} Waktu aktif: <b>24 Jam (Senin-Jumat)</b>
{EMOJI['lightning']} Filter: Crossover + Volume Surge + Daily Trend

💡 <i>Gunakan /signals untuk top BUY hari ini</i>
""".strip()
    await update.effective_message.reply_text(pesan, parse_mode=ParseMode.HTML)


async def cmd_screening(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ── Jika tidak ada argumen → tampilkan pilihan saham populer ──
    if not context.args:
        # Buat grid tombol dari 24 instrumen teratas
        instrumen_populer = config.FOREX_WATCHLIST[:24]  # 24 pair = 6 baris × 4 kolom
        tombol_rows = []
        for i in range(0, len(instrumen_populer), 4):
            row = [
                InlineKeyboardButton(k, callback_data=f"screen_{k}")
                for k in instrumen_populer[i:i+4]
            ]
            tombol_rows.append(row)

        await update.effective_message.reply_text(
            f"{EMOJI['radar']} <b>Pilih pair untuk dianalisa:</b>\n"
            f"<i>Atau ketik: /screening KODE (contoh: /screening EURUSD=X)</i>",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(tombol_rows),
        )
        return

    # Normalisasi Ticker — Mendukung Forex (EURUSD=X), Commodity (GC=F), dll.
    raw = context.args[0].strip().upper()
    # Jika sudah ada suffix yfinance, pakai langsung
    if "=X" in raw or "=F" in raw or raw.endswith("=F"):
        kode_input = raw
    elif len(raw) == 6 and raw.isalpha():
        # Pair forex 6 huruf → tambahkan =X
        kode_input = f"{raw}=X"
    elif raw in ("XAUUSD", "GOLD", "GC"):
        kode_input = "GC=F"
    elif raw in ("XAGUSD", "SILVER"):
        kode_input = "SI=F"
    elif raw in ("OIL", "CRUDEOIL", "CL"):
        kode_input = "CL=F"
    elif raw in ("BZ", "BRENT"):
        kode_input = "BZ=F"
    else:
        # Fallback: pakai raw input langsung
        kode_input = raw

    await update.effective_message.reply_chat_action("typing")

    loading_msg = await update.effective_message.reply_text(
        f"{EMOJI['clock']} <b>[1/4]</b> Mengambil data harga & kalkulasi indikator...",
        parse_mode=ParseMode.HTML)

    try:
        async def _do_screening():
            # Step 1: Technical data
            screening_data = await asyncio.get_event_loop().run_in_executor(
                None, full_screening, kode_input)

            if screening_data is None:
                await loading_msg.edit_text(
                    f"{EMOJI['cross']} Gagal mengambil data <b>{kode_input}</b>.\n"
                    f"Pastikan pair valid (e.g., EURUSD=X).",
                    parse_mode=ParseMode.HTML)
                return

            # Step 2: News
            await loading_msg.edit_text(
                f"{EMOJI['news']} <b>[2/4]</b> Mengumpulkan berita fundamental...\n"
                f"<i>(Hanya butuh 1-3 detik)</i>",
                parse_mode=ParseMode.HTML)
            headlines = await get_news_for_forex(kode_input)

            # Step 3: AI Analysis with technical context
            await loading_msg.edit_text(
                f"{EMOJI['robot']} <b>[3/4]</b> Analisa {len(headlines)} berita & indikator dengan AI...",
                parse_mode=ParseMode.HTML)
            tech_ctx = {
                "technical_score": screening_data.get("technical_score", 0),
                "rsi": screening_data["kondisi"]["rsi"]["nilai"],
                "uptrend_daily": screening_data.get("daily_trend", {}).get("uptrend_daily", False),
                "bb_squeeze": screening_data["kondisi"]["bollinger"]["squeeze"],
                "bb_breakout": screening_data["kondisi"]["bollinger"]["breakout"],
            }
            sentiment_data = await analyze_sentiment(kode_input, headlines, tech_ctx)

            # Step 4: Generate chart with timeout (15s graceful degradation)
            await loading_msg.edit_text(
                f"{EMOJI['chart']} <b>[4/4]</b> Membuat chart candlestick...",
                parse_mode=ParseMode.HTML)
            try:
                chart_buf = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, generate_chart, screening_data["df"], kode_input, screening_data),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"[CHART] Chart timeout untuk {kode_input} — kirim teks saja")
                chart_buf = None

            # Build message
            pesan = build_screening_message(screening_data, sentiment_data, headlines)

            if chart_buf:
                # Telegram limit caption foto = 1024 char.
                # Solusi: kirim chart dengan caption pendek, analisa sebagai pesan teks terpisah.
                reko_data = get_final_recommendation(
                    screening_data.get("technical_score", 0), sentiment_data)
                reko_label = reko_data["label"]
                caption_pendek = (
                    f"📊 <b>{kode_input}</b> | {screening_data['harga_terakhir']:.5f} "
                    f"| Score: {screening_data.get('technical_score', 0)}/100\n"
                    f"🏁 <b>{reko_label}</b>"
                )
                try:
                    await update.effective_message.reply_photo(
                        photo=chart_buf,
                        caption=caption_pendek,
                        parse_mode=ParseMode.HTML)
                    await loading_msg.delete()
                    # Kirim analisa lengkap sebagai pesan teks terpisah
                    await update.effective_message.reply_text(pesan, parse_mode=ParseMode.HTML)
                except Exception as photo_err:
                    logger.warning(f"[BOT] Gagal kirim chart, fallback ke teks: {photo_err}")
                    await loading_msg.edit_text(pesan, parse_mode=ParseMode.HTML)
            else:
                await loading_msg.edit_text(pesan, parse_mode=ParseMode.HTML)

            logger.info(f"[BOT] ✅ Screening selesai untuk {kode_input}")

        await asyncio.wait_for(_do_screening(), timeout=180.0)

    except asyncio.TimeoutError:
        await loading_msg.edit_text(
            f"{EMOJI['warning']} Analisa <b>{kode_input}</b> timeout (>3 menit).\n"
            f"Coba lagi dalam beberapa menit.",
            parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"[BOT] Error screening {kode_input}: {e}")
        await loading_msg.edit_text(
            f"{EMOJI['cross']} Error menganalisa <b>{kode_input}</b>.\nCoba lagi.",
            parse_mode=ParseMode.HTML)


# ----------------------------------------------------------------
# COMMAND: /calendar (v7.0 HOLY GRAIL)
# ----------------------------------------------------------------
async def cmd_calendar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menampilkan jadwal berita ekonomi penting hari ini."""
    msg = await update.effective_message.reply_text(f"{EMOJI['clock']} Mengambil data kalender ekonomi...")
    
    try:
        calendar = await get_economic_calendar()
        if not calendar:
            await msg.edit_text("Tidak ada data kalender ekonomi tersedia.")
            return

        txt = (
            f"📅 <b>KALENDER EKONOMI HARI INI</b>\n"
            f"<i>Fokus pada High Impact (Red Folder)</i>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        )
        
        # Sort based on time
        calendar.sort(key=lambda x: x["time"])
        
        for event in calendar:
            time_str = event["time"].astimezone(WIB).strftime("%H:%M")
            impact_emoji = "🔴" if event["impact"] == "High" else "🟡" if event["impact"] == "Medium" else "⚪"
            
            txt += (
                f"{time_str} | {impact_emoji} <b>{event['country']}</b>\n"
                f"📌 {event['title']}\n"
            )
            if event["forecast"] or event["previous"]:
                txt += f"📊 <i>F: {event['forecast']} | P: {event['previous']}</i>\n"
            txt += "\n"

        txt += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        txt += f"<i>{EMOJI['info']} Hindari trading 30 menit sebelum/sesudah berita High Impact (🔴) untuk menghindari slippage.</i>"
        
        await msg.edit_text(txt, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"[CALENDAR] Error: {e}")
        await msg.edit_text("Gagal mengambil data kalender.")


# ----------------------------------------------------------------
# /SIGNALS
# ----------------------------------------------------------------
async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scan Forex dan tampilkan top kandidat BUY/SELL hari ini."""
    await update.effective_message.reply_chat_action("typing")
    msg = await update.effective_message.reply_text(
        f"{EMOJI['radar']} Scanning <b>{len(config.FOREX_WATCHLIST)} instrumen</b> untuk kandidat BUY..."
        f"\n<i>Proses sekitar 30 detik...</i>",
        parse_mode=ParseMode.HTML)

    try:
        # v7.0 Fetch Calendar first for Kill-Switch
        calendar = await get_economic_calendar()
        
        candidates = await asyncio.get_event_loop().run_in_executor(
            None, scan_forex_buy, config.FOREX_WATCHLIST, calendar)

        if not candidates:
            await msg.edit_text(
                f"{EMOJI['info']} Tidak ada kandidat signal kuat saat ini.\n"
                f"Filter: Volatilitas tinggi + volume surge + AI Sentiment.",
                parse_mode=ParseMode.HTML)
            return

        waktu = datetime.now(WIB).strftime("%d %b %Y, %H:%M WIB")
        baris = []
        for i, c in enumerate(candidates, 1):
            kode = c["kode"]
            harga = c["harga_terakhir"]
            pct = c["perubahan_pct"]
            score = c["technical_score"]
            sl = c["risk_management"]["stop_loss"]
            tp = c["risk_management"]["target_price"]
            vol_ratio = c["kondisi"]["volume"]["rasio"]
            bb_txt = " ⚡SQUEEZE" if c["kondisi"]["bollinger"]["squeeze"] else ""
            filled = round(score / 10)
            bar = "█" * filled + "░" * (10 - filled)
            lot = c["risk_management"]["recommended_lot"]
            baris.append(
                f"<b>{i}. {kode}</b>{bb_txt}\n"
                f"   💰 <code>{harga:.5f}</code> 📈 <b>+{pct:.2f}%</b> | Vol ×{vol_ratio:.1f}\n"
                f"   [{bar}] <b>{score}/100</b>\n"
                f"   🛑 SL: <code>{sl:.5f}</code> | 🎯 TP: <code>{tp:.5f}</code>\n"
                f"   📦 Recommended Lot: <b>{lot:.2f}</b> (Risk 1%)"
            )

        teks = (
            f"{EMOJI['rocket']} <b>TOP KANDIDAT SETUP HARI INI</b>\n"
            f"Dari {len(config.FOREX_WATCHLIST)} instrumen dipantau\n"
            f"Filter: Volume/Volatility Breakout\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            + "\n\n".join(baris)
            + f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{EMOJI['clock']} {waktu}\n"
            f"{EMOJI['info']} <i>Gunakan /screening [KODE] untuk analisa chart lengkap. by J</i>"
        )
        await msg.edit_text(teks, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"[BOT] Error /rekomendasi: {e}")
        await msg.edit_text(f"{EMOJI['cross']} Error saat scan. Coba lagi.", parse_mode=ParseMode.HTML)


# ----------------------------------------------------------------
# /DANGER
# ----------------------------------------------------------------
async def cmd_danger(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scan Forex untuk mendeteksi Drop / Volatilitas bahaya."""
    await update.effective_message.reply_chat_action("typing")
    msg = await update.effective_message.reply_text(
        f"{EMOJI['warning']} Scanning <b>{len(config.FOREX_WATCHLIST)} instrumen</b> untuk deteksi bahaya berita Red Folder..."
        f"\n<i>Proses sekitar 30 detik...</i>",
        parse_mode=ParseMode.HTML)

    try:
        dangerous = await asyncio.get_event_loop().run_in_executor(
            None, scan_forex_danger, config.FOREX_WATCHLIST)

        if not dangerous:
            await msg.edit_text(
                f"{EMOJI['check']} Tidak ada instrumen yang memasuki zona bahaya saat ini.\n"
                f"Pasar Forex relatif stabil! {EMOJI['bullish']}",
                parse_mode=ParseMode.HTML)
            return

        waktu = datetime.now(WIB).strftime("%d %b %Y, %H:%M WIB")
        baris = []
        for i, d in enumerate(dangerous, 1):
            kode = d["kode"]
            harga = d["harga_terakhir"]
            pct = d["perubahan_pct"]
            score = d["technical_score"]
            rsi = d["kondisi"]["rsi"]["nilai"]
            dtype = d.get("danger_type", "DROP")
            badge = "💥 DROP BESAR / FLASH CRASH" if dtype == "DROP" else "🔥 OVERBOUGHT MAKSIMAL"
            vol_ratio = d["kondisi"]["volume"]["rasio"]
            baris.append(
                f"<b>{i}. {kode}</b> — {badge}\n"
                f"   💰 <code>{harga:.5f}</code> 📉 <b>{pct:+.2f}%</b>\n"
                f"   RSI: {rsi:.1f} | Vol ×{vol_ratio:.1f} | Score: {score}/100"
            )

        teks = (
            f"{EMOJI['warning']} <b>RADAR BAHAYA — FLASH VOLATILITY</b>\n"
            f"Dari {len(config.FOREX_WATCHLIST)} instrumen Forex\n"
            f"Kriteria: Turun ekstrem atau oversold/overbought kuat.\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            + "\n\n".join(baris)
            + f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{EMOJI['clock']} {waktu}\n"
            f"{EMOJI['info']} <i>Hindari masuk posisi pada instrumen di atas! by J {EMOJI['shield']}</i>"
        )
        await msg.edit_text(teks, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"[BOT] Error /danger: {e}")
        await msg.edit_text(f"{EMOJI['cross']} Error saat scan. Coba lagi.", parse_mode=ParseMode.HTML)


# ----------------------------------------------------------------
# /HEATMAP (MENGGANTIKAN /MARKET)
# ----------------------------------------------------------------
async def cmd_heatmap(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Tampilkan Ringkasan Volatilitas Mata Uang / Pasangan Mata uang."""
    await update.effective_message.reply_chat_action("typing")
    msg = await update.effective_message.reply_text(
        f"{EMOJI['radar']} Mengumpulkan data Heatmap Market dari <b>{len(config.FOREX_WATCHLIST)} instrumen</b>..."
        f"\n<i>Proses sekitar 5-10 detik...</i>",
        parse_mode=ParseMode.HTML)

    try:
        data = await asyncio.get_event_loop().run_in_executor(
            None, get_market_leaders, config.FOREX_WATCHLIST)

        if not data:
            await msg.edit_text(f"{EMOJI['cross']} Gagal mengambil data market.", parse_mode=ParseMode.HTML)
            return

        waktu = datetime.now(WIB).strftime("%d %b %Y, %H:%M WIB")
        
        # Helper format
        def format_vol(val):
            if val >= 1e6:
                return f"{val/1e6:.1f}M"
            elif val >= 1e3:
                return f"{val/1e3:.0f}K"
            return f"{val:,.0f}"

        txt = f"🌐 <b>FOREX HEATMAP</b>\n"
        txt += f"<i>Update: {waktu}</i>\n"
        txt += f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # 0. Currency Strength Meter (CSM)
        if data.get("csm"):
            txt += f"📊 <b>CURRENCY STRENGTH METER (CSM)</b>\n"
            txt += f"<i>Relative Strength vs Market Average</i>\n"
            for m, val in data["csm"].items():
                emoji = "🟦" if val >= 0.5 else "🟩" if val >= 0.1 else "⬜" if val >= -0.1 else "🟧" if val >= -0.5 else "🟥"
                txt += f"{emoji} <b>{m}</b>: <code>{val:+.2f}%</code>\n"
            txt += "\n"

        # 1. Top Gainers
        txt += f"🚀 <b>MOST BULLISH PAIRS</b>\n"
        if data.get("top_gainer"):
            for i, x in enumerate(data["top_gainer"], 1):
                txt += f"{i}. <b>{x['kode']}</b> :  {x['harga']:.5f} (<b>+{x['change_pct']:.2f}%</b>)\n"
        else:
            txt += "<i>Belum ada data...</i>\n"
        txt += "\n"

        # 2. Live Rebound
        txt += f"🟢 <b>REBOUNDING PAIRS (Sentuh Support)</b>\n"
        if data.get("live_rebound"):
            for i, x in enumerate(data["live_rebound"], 1):
                txt += f"{i}. <b>{x['kode']}</b> :  {x['harga']:.5f} (<b>+{x['change_pct']:.2f}%</b>)\n"
        else:
            txt += "<i>Tidak ada pair yang rebound saat ini...</i>\n"
        txt += "\n"

        # 3. Top Value -> kita jadikan Momentum / Volatility terbesar (Volume Proxy)
        txt += f"💸 <b>ALGORITMA DETEKSI MOMENTUM TINGGI</b>\n"
        if data.get("top_volume"):
            for i, x in enumerate(data["top_volume"], 1):
                txt += f"{i}. <b>{x['kode']}</b> :  Volatility Surge Detected\n"
        else:
            txt += "<i>Belum ada data...</i>\n"
        txt += "\n"

        # 4. Top Volume
        txt += f"📊 <b>TOP VOLUME TRANSAKSI</b>\n"
        if data.get("top_volume"):
            for i, x in enumerate(data["top_volume"], 1):
                txt += f"{i}. <b>{x['kode']}</b> :  {format_vol(x['volume'])}\n"
        else:
            txt += "<i>Belum ada data...</i>\n"

        txt += f"\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
        txt += f"💡 <i>Gunakan /screening [KODE] untuk cek teknikal detail. by J</i>"

        await msg.edit_text(txt, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"[BOT] Error /market: {e}")
        await msg.edit_text(f"{EMOJI['cross']} Error saat mengambil data market.", parse_mode=ParseMode.HTML)


# ----------------------------------------------------------------
# /AUTOSCALPING (v6.0) - AI Trading Plan Forex
# ----------------------------------------------------------------
async def _run_autoscalping(update: Update, context: ContextTypes.DEFAULT_TYPE, force: bool = False) -> None:
    await update.effective_message.reply_chat_action("typing")
    msg_text = f"{EMOJI['radar']} <b>Fase 1/3:</b> Filter Kuantitatif Ketat dari {len(config.FOREX_WATCHLIST)} instrumen..."
    if force:
        msg_text = f"{EMOJI['radar']} <b>Fase 1/3:</b> Filter Kuantitatif (Mode FORCE) dari {len(config.FOREX_WATCHLIST)} instrumen..."
        
    msg = await update.effective_message.reply_text(msg_text, parse_mode=ParseMode.HTML)

    try:
        # FASE 1: Filter Kuantitatif
        # v7.0 Ambil Kalender Ekonomi
        calendar = await get_economic_calendar()
        
        candidates = await asyncio.get_event_loop().run_in_executor(
            None, get_autoscalping_candidates, config.FOREX_WATCHLIST, force, calendar)
            
        if not candidates:
            await msg.edit_text(
                f"🤷‍♂️ <b>Tidak ada kandidat sempurna saat ini.</b>\n"
                f"Pasar tidak sedang memberikan setup scalping yang jelas (Volume sepi/Trend campur). "
                f"<i>Lindungi modalmu!</i>\n\n"
                f"💡 <i>Gunakan /autoscalpingforce jika Anda agresif (High Risk).</i>", 
                parse_mode=ParseMode.HTML)
            return

        cand_str = ", ".join([c['kode'] for c in candidates])
        await msg.edit_text(
            f"{EMOJI['search']} <b>Fase 2/3:</b> Mengumpulkan sentimen Makro Ekonomi...\n"
            f"<i>Lolos filter kuantitatif: {cand_str}</i>", 
            parse_mode=ParseMode.HTML)

        # FASE 2: Macro News
        macro_news = await get_macro_news(3)
        
        # v7.5 Fetch CSM for AI Context
        from data_fetcher import get_market_leaders
        market_stats = await asyncio.get_event_loop().run_in_executor(
            None, get_market_leaders, config.FOREX_WATCHLIST[:15])
        csm_data = market_stats.get("csm")

        await msg.edit_text(
            f"🧠 <b>Fase 3/3:</b> Llama-3 70B sedang meramu Trading Plan...\n"
            f"<i>Basis: CSM + Macro News + Technical</i>", 
            parse_mode=ParseMode.HTML)

        # FASE 3: AI Inference
        ai_plan = await analyze_autoscalping(candidates, macro_news, csm_data)

        if not ai_plan or not isinstance(ai_plan, dict):
            await msg.edit_text(f"{EMOJI['cross']} AI Gagal meramu Trading Plan. Coba lagi.", parse_mode=ParseMode.HTML)
            return
            
        # Parse output JSON dari AI
        market_view = ai_plan.get("market_view", "Kondisi pasar standar.")
        kode = ai_plan.get("pemenang_kode", "N/A").upper()
        nama = ai_plan.get("pemenang_nama", "")
        alasan = ai_plan.get("alasan_menang", "")
        plan = ai_plan.get("trading_plan", {})
        entry = plan.get("entry_area", "")
        t1 = plan.get("target_1", "")
        t2 = plan.get("target_2", "")
        sl = plan.get("stop_loss", "")
        psikologi = ai_plan.get("pesan_psikologi", "")
        
        # Validasi format kode
        kode_ticker = format_ticker(kode)
        kode = get_clean_code(kode_ticker)

        force_badge = " (FORCE MODE ⚠️ HIGH RISK)" if force else ""
        teks = (
            f"⚡ <b>AUTO SCALPING TRADING PLAN{force_badge}</b> ⚡\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌐 <b>Market Hari Ini:</b> {market_view}\n\n"
            
            f"🎯 <b>PAIR TERPILIH: {kode}</b>\n"
            f"<i>{nama}</i>\n"
            f"💡 <b>Strategi:</b> {alasan}\n\n"
            
            f"⚖️ <b>TRADING PLAN KETAT:</b>\n"
            f"🟩 <b>ENTRY AREA:</b> <code>{entry}</code>\n"
            f"🚀 <b>TARGET 1:</b> <code>{t1}</code>\n"
            f"🚀 <b>TARGET 2:</b> <code>{t2}</code>\n"
            f"🛑 <b>CUT LOSS:</b> <code>{sl}</code>\n\n"
            
            f"📦 <b>RECOMMENDED LOT: {plan.get('lot_size', '0.01' if not candidates else candidates[0]['risk_management']['recommended_lot'])}</b> (Risk 1%)\n"
            f"⚠️ <b>Pesan AI:</b> <i>{psikologi}</i>\n"
            f"💡 <i>DYOR! by J</i>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        await msg.edit_text(teks, parse_mode=ParseMode.HTML)
        
        # Log to Database
        cand_dict = {c["kode"]: c for c in candidates}
        try:
            if kode in cand_dict:
                harga_skrg = cand_dict[kode].get("harga_terakhir", 0.0)
                risk_m = cand_dict[kode].get("risk_management", {})
                target_q = risk_m.get("target_price", 0.0)
                sl_q = risk_m.get("stop_loss", 0.0)
                await db.log_signal("AUTOSCALPING", kode, harga_skrg, target_q, sl_q)
        except Exception as db_err:
            logger.error(f"[DB] Error logging autoscalp signal: {db_err}")
        
        # Opsi: Kirim chart sekalian agar user bisa lihat visual
        if kode in cand_dict:
            df = cand_dict[kode]["df"]
            score = cand_dict[kode]["technical_score"]
            photo_io = await asyncio.get_event_loop().run_in_executor(
                None, generate_chart, df, kode_ticker, cand_dict[kode])
            if photo_io:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=photo_io,
                    caption=f"Visual chart untuk {kode} 👆"
                )

    except Exception as e:
        logger.error(f"[BOT] Error /autoscalping: {e}")
        await msg.edit_text(f"{EMOJI['cross']} Terjadi kesalahan internal.", parse_mode=ParseMode.HTML)


async def cmd_autoscalping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scan Forex secara kuantitatif, lalu AI membuat Trading Plan."""
    await _run_autoscalping(update, context, force=False)

async def cmd_autoscalping_force(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scan Forex dan paksa AI mencari setup terbaik meskipun kurang ideal."""
    await _run_autoscalping(update, context, force=True)


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /settings — Live config tanpa restart server.
    Usage:
      /settings              → tampilkan setting saat ini
      /settings equity 2000  → ubah modal trading
      /settings risk 0.5     → ubah risiko per trade (%)
      /settings reset        → kembalikan ke default
    """
    args = context.args or []

    # Tampilkan setting saat ini
    if not args:
        s = await db.get_all_settings()
        txt = (
            f"⚙️ <b>KONFIGURASI BOT SAAT INI</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Modal Trading (Equity): <b>${s.get('equity', '1000.0')}</b>\n"
            f"🛡️ Risiko per Trade: <b>{s.get('risk_pct', '1.0')}%</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>Perintah:</i>\n"
            f"<code>/settings equity 2000</code> — ubah modal\n"
            f"<code>/settings risk 0.5</code> — ubah risiko (%)\n"
            f"<code>/settings reset</code> — kembalikan ke default"
        )
        await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML)
        return

    sub_cmd = args[0].lower()

    # Reset ke default
    if sub_cmd == "reset":
        await db.set_setting("equity", "1000.0")
        await db.set_setting("risk_pct", "1.0")
        await update.effective_message.reply_text(
            f"✅ Setting dikembalikan ke default: Modal $1000.0 | Risiko 1.0%",
            parse_mode=ParseMode.HTML)
        return

    # Update equity
    if sub_cmd == "equity" and len(args) >= 2:
        try:
            val = float(args[1])
            if val <= 0:
                raise ValueError
            await db.set_setting("equity", str(val))
            await update.effective_message.reply_text(
                f"✅ Modal trading diubah → <b>${val:.2f}</b>\n"
                f"Lot size pada sinyal berikutnya akan menyesuaikan.",
                parse_mode=ParseMode.HTML)
        except (ValueError, IndexError):
            await update.effective_message.reply_text(
                f"❌ Nilai tidak valid. Contoh: <code>/settings equity 2000</code>",
                parse_mode=ParseMode.HTML)
        return

    # Update risk %
    if sub_cmd == "risk" and len(args) >= 2:
        try:
            val = float(args[1])
            if not (0.1 <= val <= 5.0):
                raise ValueError("Risiko harus antara 0.1% - 5.0%")
            await db.set_setting("risk_pct", str(val))
            await update.effective_message.reply_text(
                f"✅ Risiko per trade diubah → <b>{val}%</b>\n"
                f"Nilai dollar risiko = ${float(await db.get_setting('equity')) * val / 100:.2f}",
                parse_mode=ParseMode.HTML)
        except ValueError as e:
            await update.effective_message.reply_text(
                f"❌ Input tidak valid: {e}. Contoh: <code>/settings risk 1.5</code>",
                parse_mode=ParseMode.HTML)
        return

    await update.effective_message.reply_text(
        f"❓ Perintah tidak dikenal. Ketik <code>/settings</code> untuk pilihan.",
        parse_mode=ParseMode.HTML)


    query = update.callback_query
    await query.answer()

    if query.data == "watchlist":
        await cmd_watchlist(update, context)
    elif query.data == "help":
        await cmd_help(update, context)
    elif query.data.startswith("screen_"):
        kode = query.data.replace("screen_", "")
        context.args = [kode]
        await cmd_screening(update, context)


# ----------------------------------------------------------------
# RADAR JOB
# ----------------------------------------------------------------
def is_market_open() -> bool:
    now = datetime.now(WIB)
    if now.weekday() >= 5:
        return False
    return dtime(config.MARKET_OPEN_HOUR, 0) <= now.time() <= dtime(config.MARKET_CLOSE_HOUR, 0)


async def radar_scan_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_market_open():
        logger.info("[RADAR] Bursa tutup, scan dilewati.")
        return

    waktu = datetime.now(WIB).strftime("%H:%M WIB")
    n = len(config.FOREX_WATCHLIST)
    logger.info(f"[RADAR] 🔍 Scan {n} pair Forex — {waktu}")

    # ── FASE 1: Bulk Download semua pair sekaligus (1 request) ──
    from data_fetcher import bulk_fetch_ohlcv, quick_scan
    data_map = await asyncio.get_event_loop().run_in_executor(
        None, bulk_fetch_ohlcv, config.FOREX_WATCHLIST)
    logger.info(f"[RADAR] Bulk download selesai: {len(data_map)}/{n} instrumen")

    # ── FASE 2: Quick scan — filter awal yang ringan ──
    kandidat = []
    for kode, df in data_map.items():
        result = quick_scan(kode, df)
        if result and result.get("sinyal_valid"):
            kandidat.append(kode)

    if not kandidat:
        logger.info(f"[RADAR] Tidak ada sinyal dari {len(data_map)} saham.")
        return

    logger.info(f"[RADAR] {len(kandidat)} kandidat lolos quick scan → analisa mendalam...")

    # ── FASE 3: Full screening hanya untuk kandidat (hemat RAM) ──
    sinyal = 0
    for kode in kandidat:
        try:
            data = await asyncio.get_event_loop().run_in_executor(None, full_screening, kode)
            if data is None:
                continue

            # Filter multi-timeframe: hanya alert jika daily uptrend
            if not data.get("daily_trend", {}).get("uptrend_daily", False):
                logger.info(f"[RADAR] {kode}: daily downtrend/sideways → skip")
                continue

            headlines = await get_news_for_forex(kode)
            tech_ctx = {
                "technical_score": data.get("technical_score", 0),
                "rsi": data["kondisi"]["rsi"]["nilai"],
                "uptrend_daily": True,
                "bb_squeeze": data["kondisi"]["bollinger"]["squeeze"],
                "bb_breakout": data["kondisi"]["bollinger"]["breakout"],
            }
            sentiment = await analyze_sentiment(kode, headlines, tech_ctx)

            if not is_signal_approved(sentiment):
                logger.info(f"[RADAR] ⛔ {kode} difilter sentimen Bearish")
                continue

            pesan = build_signal_alert_message(data, sentiment)
            chart_buf = await asyncio.get_event_loop().run_in_executor(
                None, generate_chart, data["df"], kode, data)

            if chart_buf:
                await context.bot.send_photo(
                    chat_id=config.TELEGRAM_CHAT_ID, photo=chart_buf,
                    caption=pesan, parse_mode=ParseMode.HTML)
            else:
                await context.bot.send_message(
                    chat_id=config.TELEGRAM_CHAT_ID, text=pesan,
                    parse_mode=ParseMode.HTML)

            # Log to Database
            try:
                harga = data.get("harga_terakhir", 0.0)
                risk_m = data.get("risk_management", {})
                target = risk_m.get("target_price", 0.0)
                sl = risk_m.get("stop_loss", 0.0)
                await db.log_signal("RADAR_SCAN", kode, harga, target, sl)
            except Exception as db_err:
                logger.error(f"[DB] Error logging radar signal: {db_err}")

            sinyal += 1
            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"[RADAR] Error {kode}: {e}")

    logger.info(f"[RADAR] Selesai. {sinyal} sinyal dari {n} pasang mata uang/komoditas.")


# ----------------------------------------------------------------
# ERROR HANDLER
# ----------------------------------------------------------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle semua error Telegram secara terpusat."""
    from telegram.error import Conflict, NetworkError, TimedOut
    err = context.error

    if isinstance(err, Conflict):
        # Conflict = 2 instance bot jalan bersamaan — bukan crash, cukup log
        logger.warning(
            "[BOT] ⚠️ CONFLICT: Ada 2 instance bot berjalan bersamaan! "
            "Pastikan TIDAK menjalankan 'python bot.py' di PC saat Railway aktif."
        )
    elif isinstance(err, (NetworkError, TimedOut)):
        logger.warning(f"[BOT] Network error (akan retry otomatis): {err}")
    else:
        logger.error(f"[BOT] Unhandled error: {err}", exc_info=err)


# ----------------------------------------------------------------
# SETUP
# ----------------------------------------------------------------
async def post_init(application: Application) -> None:
    await db.init_db()
    logger.info("[BOT] ✅ Terhubung ke Telegram.")
    commands = [
        BotCommand("start", "Menu utama"),
        BotCommand("autoscalping", "AI Trading Plan Otomatis (v2.0)"),
        BotCommand("autoscalpingforce", "Paksa AI mencari setup scalping (High Risk)"),
        BotCommand("screening", "Analisa + Chart (contoh: /screening EURUSD=X)"),
        BotCommand("heatmap", "Data CSM & volatilitas market"),
        BotCommand("signals", "Top pair kandidat BUY"),
        BotCommand("danger", "Peringatan Flash Volatility"),
        BotCommand("calendar", "Jadwal berita ekonomi hari ini"),
        BotCommand("winrate", "Statistik Win Rate bot"),
        BotCommand("settings", "Konfigurasi equity & risiko (live)"),
        BotCommand("watchlist", "Pantauan instrumen"),
        BotCommand("help", "Panduan penggunaan v2.0"),
    ]
    await application.bot.set_my_commands(commands)

# ----------------------------------------------------------------
# TRADE TRACKER JOB & WINRATE (v7.5)
# ----------------------------------------------------------------
async def trade_tracker_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Job background untuk mengecek resolusi sinyal (WIN/LOSS)"""
    from data_fetcher import format_ticker
    import yfinance as yf
    
    open_signals = await db.get_open_signals()
    if not open_signals:
        return
        
    logger.info(f"[TRACKER] Mengecek {len(open_signals)} sinyal OPEN...")
    
    # Bungkus yf.download ke fungsi sync agar bisa di-run in executor
    def fetch_1m_data(ticker):
        return yf.download(ticker, period="1d", interval="1m", progress=False)
    
    for s in open_signals:
        try:
            sid = s["id"]
            kode = s["kode"]
            target = s["target_1"]
            sl = s["stop_loss"]
            entry = s["harga_masuk"]
            
            ticker = format_ticker(kode)
            
            # Gunakan Run in Executor agar tidak memblokir Bot
            data = await asyncio.get_event_loop().run_in_executor(None, fetch_1m_data, ticker)
            if data.empty: continue
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Cari harga Tertinggi (High) dan Terendah (Low) hari ini
            highest_price = float(data["High"].max())
            lowest_price = float(data["Low"].min())
            
            # Resolusi Sinyal: Cek apakah High menyentuh Target ATAU Low menyentuh SL
            # Asumsi ini adalah posisi BUY (Long)
            if highest_price >= target:
                pl = ((target - entry) / entry) * 100
                await db.update_signal_status(sid, "WIN", pl)
                logger.info(f"[TRACKER] 🏆 {kode} hit TARGET! status set to WIN.")
            elif lowest_price <= sl:
                pl = ((sl - entry) / entry) * 100
                await db.update_signal_status(sid, "LOSS", pl)
                logger.info(f"[TRACKER] 🛑 {kode} hit SL! status set to LOSS.")
                
        except Exception as e:
            logger.error(f"[TRACKER] Error tracking {s['kode']}: {e}")

async def cmd_winrate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Tampilkan statistik performa AI Bot"""
    stats = await db.get_signal_stats()
    
    txt = f"{EMOJI['chart']} <b>STATISTIK PERFORMA BOT AI</b>\n"
    txt += f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    txt += f"📊 Total Sinyal Resolved: <b>{stats['total']}</b>\n"
    txt += f"✅ Take Profit (WIN): <b>{stats['wins']}</b>\n"
    txt += f"❌ Stop Loss (LOSS): <b>{stats['losses']}</b>\n"
    txt += f"🔥 <b>Win Rate: {stats['win_rate']}%</b>\n\n"
    
    if stats['win_rate'] > 60:
        txt += f"💡 <i>Bot dalam performa tinggi! Tetap jaga manajemen risiko.</i>"
    elif stats['total'] < 5:
        txt += f"💡 <i>Data belum cukup untuk statistik akurat. Terus pantau!</i>"
    else:
        txt += f"💡 <i>Pasar sedang dinamis. Review strategi di Sesi London/NY.</i>"
    
    txt += f"\n💡 <i>DYOR! by J</i>"
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML)



def main() -> None:
    validate_config()
    logger.info("[BOT] 🚀 Forex Radar Bot & AI Screener v2.0 starting...")
    logger.info(f"[BOT] Memantau {len(config.FOREX_WATCHLIST)} instrumen")

    application = (
        ApplicationBuilder()
        .token(config.TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.add_error_handler(error_handler)

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("watchlist", cmd_watchlist))
    application.add_handler(CommandHandler("autoscalping", cmd_autoscalping))
    application.add_handler(CommandHandler("autoscalpingforce", cmd_autoscalping_force))
    application.add_handler(CommandHandler("screening", cmd_screening))
    application.add_handler(CommandHandler("heatmap", cmd_heatmap))
    application.add_handler(CommandHandler("signals", cmd_signals))
    application.add_handler(CommandHandler("danger", cmd_danger))
    application.add_handler(CommandHandler("calendar", cmd_calendar))
    application.add_handler(CommandHandler("winrate", cmd_winrate))
    application.add_handler(CommandHandler("settings", cmd_settings))
    application.add_handler(CallbackQueryHandler(handle_callback))

    job_queue = application.job_queue
    if job_queue:
        # Job 1: Radar Scan (Existing)
        job_queue.run_repeating(
            callback=radar_scan_job,
            interval=config.RADAR_INTERVAL_MINUTES * 60,
            first=15,
            name="radar_scan",
        )
        # Job 2: Trade Tracking (v7.5) - Cek setiap 15 menit
        job_queue.run_repeating(
            callback=trade_tracker_job,
            interval=15 * 60,
            first=30,
            name="trade_tracking",
        )
        logger.info(f"[BOT] ⏰ Radar & Trade Tracker Aktif.")
    else:
        logger.warning("[BOT] ⚠️ JobQueue tidak tersedia!")

    logger.info("[BOT] ✅ Bot siap!")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
