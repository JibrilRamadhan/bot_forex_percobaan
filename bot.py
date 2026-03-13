"""
bot.py - IHSG Radar Bot & AI Screener v2.0
==========================================
Fitur Baru v2.0:
- Auto Chart Generation (candlestick PNG dikirim ke Telegram)
- ATR Stop Loss + Target Price
- Bollinger Bands Squeeze alert
- Multi-Timeframe confirmation
- Final Recommendation (STRONG BUY / BUY / HOLD / SELL / STRONG SELL)
- Technical Score 0-100
"""

import asyncio
import logging
import html
import io
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
from data_fetcher import full_screening, format_ticker, get_clean_code
from news_scraper import get_news_for_stock
from ai_analyzer import analyze_sentiment, is_signal_approved, get_final_recommendation

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
    "shield": "🛡️", "lightning": "⚡", "chart": "📊",
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
        title = f"{kode} | Rp {harga:,.0f} | Score: {tech_score}/100 | {waktu}"

        buf = io.BytesIO()
        fig, axes = mpf.plot(
            df_chart,
            type="candle",
            style=s,
            title=title,
            volume=True,
            addplot=addplots if addplots else None,
            panel_ratios=(4, 1, 2) if col_rsi in df_chart.columns else (4, 1),
            figsize=(12, 8),
            returnfig=True,
        )
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor="#0D0D0D")
        plt.close(fig)
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
    mtf_status = (f"{EMOJI['chart_up']} UPTREND Harian ({pct_vs_ema20d:+.1f}% vs EMA20D) ✅ Konfirmasi Valid"
                  if mtf_uptrend else
                  f"{EMOJI['chart_down']} Di bawah EMA20D ({pct_vs_ema20d:+.1f}%) ⚠️ Waspadai False Breakout")

    # Stop Loss & Target
    sl = risk.get("stop_loss", 0)
    tp = risk.get("target_price", 0)
    rr = risk.get("risk_reward", 0)
    atr = risk.get("atr", 0)
    sl_str = f"Rp {sl:,.0f} (1.5× ATR={atr:.0f})" if sl > 0 else "N/A"
    tp_str = f"Rp {tp:,.0f} (2.0× ATR)" if tp > 0 else "N/A"
    rr_str = f"1 : {rr:.1f}" if rr > 0 else "N/A"

    # Berita
    berita_str = ""
    if headlines:
        bl = "\n".join(f"  • {html.escape(h[:85])}{'...' if len(h) > 85 else ''}" for h in headlines[:3])
        berita_str = f"\n{EMOJI['news']} <b>BERITA TERBARU</b>\n{bl}\n"

    cache_note = " <i>(cached)</i>" if dari_cache else ""

    msg = f"""
{EMOJI['radar']} <b>SCREENING: {kode}</b> | <i>{nama}</i>
━━━━━━━━━━━━━━━━━━━━━━━━

{EMOJI['money']} Harga: <code>Rp {harga:,.0f}</code> {perubahan_emoji} {perubahan_str}

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

━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['target']} <b>SUPPORT & RESISTANCE</b>
━━━━━━━━━━━━━━━━━━━━━━━━
🔴 R2: <code>Rp {pivot.get('R2', 0):,.0f}</code> | 🟠 R1: <code>Rp {pivot.get('R1', 0):,.0f}</code>
⚪ PP: <code>Rp {pivot.get('PP', 0):,.0f}</code>
🟢 S1: <code>Rp {pivot.get('S1', 0):,.0f}</code> | 🔵 S2: <code>Rp {pivot.get('S2', 0):,.0f}</code>

━━━━━━━━━━━━━━━━━━━━━━━━
🏁 <b>REKOMENDASI FINAL:</b> {reko_label}
   Score Gabungan: <code>{final_score}/100</code>
━━━━━━━━━━━━━━━━━━━━━━━━

{EMOJI['clock']} {waktu_wib}
{EMOJI['info']} <i>⚠️ Bukan rekomendasi resmi. Selalu DYOR!</i>
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
{EMOJI['shield']} <b>MANAJEMEN RISIKO</b>
━━━━━━━━━━━━━━━━━━━━━━━━
🛑 Stop Loss: <code>Rp {sl:,.0f}</code> (1.5× ATR)
{EMOJI['target']} Target: <code>Rp {tp:,.0f}</code>
⚖️ Risk/Reward: <b>1 : {rr:.1f}</b>

━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['target']} S/R | R1: <code>Rp {pivot.get('R1',0):,.0f}</code> | PP: <code>Rp {pivot.get('PP',0):,.0f}</code> | S1: <code>Rp {pivot.get('S1',0):,.0f}</code>

🏁 <b>REKOMENDASI: {reko_label}</b>

{EMOJI['clock']} {waktu_wib} | {EMOJI['info']} <i>DYOR!</i>
""".strip()
    return msg


# ----------------------------------------------------------------
# COMMAND HANDLERS
# ----------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    nama = user.first_name if user else "Trader"
    keyboard = [
        [InlineKeyboardButton("📡 Watchlist", callback_data="watchlist"),
         InlineKeyboardButton("❓ Bantuan", callback_data="help")],
        [InlineKeyboardButton("📊 Contoh Screening BBCA", callback_data="screen_BBCA")],
    ]
    pesan = f"""
{EMOJI['rocket']} <b>Selamat datang, {html.escape(nama)}!</b>

{EMOJI['radar']} <b>IHSG Radar Bot & AI Screener v2.0</b>

━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['star']} <b>FITUR BARU v2.0</b>
━━━━━━━━━━━━━━━━━━━━━━━━
📊 Auto Chart Candlestick (PNG ke Telegram)
🛡️ ATR Stop Loss + Target Price otomatis
⚡ Bollinger Bands Squeeze Detection
🌐 Multi-Timeframe (filter 80% sinyal palsu)
🏁 Rekomendasi: STRONG BUY / BUY / HOLD / SELL

━━━━━━━━━━━━━━━━━━━━━━━━
📋 <b>PERINTAH</b>
━━━━━━━━━━━━━━━━━━━━━━━━
/screening [KODE] — Analisa + Chart saham
/watchlist — Daftar saham radar
/help — Panduan lengkap
""".strip()
    await update.message.reply_text(pesan, parse_mode=ParseMode.HTML,
                                    reply_markup=InlineKeyboardMarkup(keyboard))


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pesan = f"""
{EMOJI['info']} <b>PANDUAN PENGGUNAAN v2.0</b>
━━━━━━━━━━━━━━━━━━━━━━━━

{EMOJI['chart_up']} <b>Screening:</b>
<code>/screening KODE</code> — misal: <code>/screening BBCA</code>
Bot kirim analisa lengkap + chart PNG otomatis!

━━━━━━━━━━━━━━━━━━━━━━━━
📊 <b>Technical Score 0-100:</b>
🟢 70-100: Strong signal
🟡 45-69: Wait and watch  
🔴 0-44: Hindari / Cut Loss

━━━━━━━━━━━━━━━━━━━━━━━━
🛡️ <b>ATR Stop Loss:</b>
Stop loss dihitung otomatis: <code>Harga - (1.5 × ATR)</code>
Jangan hold di bawah stop loss!

━━━━━━━━━━━━━━━━━━━━━━━━
⚡ <b>Bollinger Bands Squeeze:</b>
Squeeze = saham sedang "ngumpet" energi
Breakout + Volume Surge = sinyal A++ 🔥

━━━━━━━━━━━━━━━━━━━━━━━━
🌐 <b>Multi-Timeframe:</b>
Sinyal 15m valid HANYA jika trend harian juga naik
(harga > EMA20 Daily). Filter sinyal palsu!

━━━━━━━━━━━━━━━━━━━━━━━━
{EMOJI['warning']} <i>Bot ini adalah alat bantu. BUKAN rekomendasi resmi. DYOR!</i>
""".strip()
    await update.message.reply_text(pesan, parse_mode=ParseMode.HTML)


async def cmd_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    wl = "\n".join(f"  {i+1}. <code>{k}</code>" for i, k in enumerate(config.WATCHLIST))
    pesan = f"""
{EMOJI['radar']} <b>WATCHLIST RADAR</b>
━━━━━━━━━━━━━━━━━━━━━━━━
{wl}

{EMOJI['clock']} Interval: <b>{config.RADAR_INTERVAL_MINUTES} menit</b>
{EMOJI['chart_up']} Jam aktif: <b>09:00 – 16:00 WIB</b>
""".strip()
    await update.message.reply_text(pesan, parse_mode=ParseMode.HTML)


async def cmd_screening(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            f"{EMOJI['warning']} Sertakan kode saham.\nContoh: <code>/screening INET</code>",
            parse_mode=ParseMode.HTML)
        return

    kode_input = context.args[0].strip().upper().replace(".JK", "")
    await update.message.reply_chat_action("typing")

    loading_msg = await update.message.reply_text(
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
                    f"Pastikan kode saham BEI valid.",
                    parse_mode=ParseMode.HTML)
                return

            # Step 2: News
            await loading_msg.edit_text(
                f"{EMOJI['news']} <b>[2/4]</b> Mengumpulkan berita dari 4+ sumber...\n"
                f"<i>(30-90 detik untuk akurasi tinggi)</i>",
                parse_mode=ParseMode.HTML)
            headlines = await asyncio.get_event_loop().run_in_executor(
                None, get_news_for_stock, kode_input)

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
            sentiment_data = await asyncio.get_event_loop().run_in_executor(
                None, analyze_sentiment, kode_input, headlines, tech_ctx)

            # Step 4: Generate chart
            await loading_msg.edit_text(
                f"{EMOJI['chart']} <b>[4/4]</b> Membuat chart candlestick...",
                parse_mode=ParseMode.HTML)
            chart_buf = await asyncio.get_event_loop().run_in_executor(
                None, generate_chart, screening_data["df"], kode_input, screening_data)

            # Build message
            pesan = build_screening_message(screening_data, sentiment_data, headlines)

            if chart_buf:
                # Hapus pesan loading, kirim chart + analisa
                await loading_msg.delete()
                await update.message.reply_photo(
                    photo=chart_buf,
                    caption=pesan,
                    parse_mode=ParseMode.HTML)
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
# CALLBACK QUERY
# ----------------------------------------------------------------
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == "watchlist":
        await cmd_watchlist(update._replace(message=query.message), context)
    elif query.data == "help":
        await cmd_help(update._replace(message=query.message), context)
    elif query.data.startswith("screen_"):
        kode = query.data.replace("screen_", "")
        context.args = [kode]
        await cmd_screening(update._replace(message=query.message), context)


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
    logger.info(f"[RADAR] 🔍 Scan {len(config.WATCHLIST)} saham — {waktu}")
    sinyal = 0

    for kode in config.WATCHLIST:
        try:
            data = await asyncio.get_event_loop().run_in_executor(None, full_screening, kode)
            if data is None:
                continue
            if not data.get("sinyal_valid", False):
                continue

            # Cek multi-timeframe: hanya alert jika tren harian mendukung
            if not data.get("daily_trend", {}).get("uptrend_daily", False):
                logger.info(f"[RADAR] {kode}: Sinyal 15m ada, tapi Daily downtrend → skip")
                continue

            logger.info(f"[RADAR] 🔔 Sinyal valid {kode}! Cek berita...")
            headlines = await asyncio.get_event_loop().run_in_executor(None, get_news_for_stock, kode)
            tech_ctx = {
                "technical_score": data.get("technical_score", 0),
                "rsi": data["kondisi"]["rsi"]["nilai"],
                "uptrend_daily": True,
                "bb_squeeze": data["kondisi"]["bollinger"]["squeeze"],
                "bb_breakout": data["kondisi"]["bollinger"]["breakout"],
            }
            sentiment = await asyncio.get_event_loop().run_in_executor(
                None, analyze_sentiment, kode, headlines, tech_ctx)

            if not is_signal_approved(sentiment):
                logger.info(f"[RADAR] ⛔ {kode} difilter (sentimen Bearish)")
                continue

            pesan = build_signal_alert_message(data, sentiment)

            # Kirim alert + chart
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

            sinyal += 1
            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"[RADAR] Error {kode}: {e}")

    logger.info(f"[RADAR] Selesai. {sinyal} sinyal dikirim dari {len(config.WATCHLIST)} saham.")


# ----------------------------------------------------------------
# SETUP
# ----------------------------------------------------------------
async def post_init(application: Application) -> None:
    logger.info("[BOT] ✅ Terhubung ke Telegram.")
    commands = [
        BotCommand("start", "Menu utama"),
        BotCommand("screening", "Analisa + Chart saham (contoh: /screening INET)"),
        BotCommand("watchlist", "Daftar saham radar"),
        BotCommand("help", "Panduan penggunaan"),
    ]
    await application.bot.set_my_commands(commands)


def main() -> None:
    validate_config()
    logger.info("[BOT] 🚀 IHSG Radar Bot & AI Screener v2.0 starting...")
    logger.info(f"[BOT] Watchlist: {', '.join(config.WATCHLIST)}")

    application = (
        ApplicationBuilder()
        .token(config.TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("watchlist", cmd_watchlist))
    application.add_handler(CommandHandler("screening", cmd_screening))
    application.add_handler(CallbackQueryHandler(handle_callback))

    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(
            callback=radar_scan_job,
            interval=config.RADAR_INTERVAL_MINUTES * 60,
            first=15,
            name="radar_scan",
        )
        logger.info(f"[BOT] ⏰ Radar setiap {config.RADAR_INTERVAL_MINUTES} menit.")
    else:
        logger.warning("[BOT] ⚠️ JobQueue tidak tersedia!")

    logger.info("[BOT] ✅ Bot siap!")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
