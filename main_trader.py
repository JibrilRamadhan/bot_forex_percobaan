"""
main_trader.py - Fully Automated Trading Engine (v8.0 MT5)
==========================================================
Sistem 'Jantung' (Main Loop) dari Arsitektur MT5 Triple Screen + AI.
Skrip ini berjalan murni di background Windows, menarik data pasar 
secara berkala (1 menit), meminta analisa ke AI Llama-3/Gemini, 
dan langsung mengeksekusi market order ketika terkonfirmasi STRONG BUY / STRONG SELL.
"""

import asyncio
import logging
import time
from typing import Dict, Any

import MetaTrader5 as mt5
import pandas as pd
from mt5_engine import init_mt5, execute_trade, close_position
from data_fetcher import analyze_triple_screen, get_pip_multiplier
from ai_analyzer import analyze_mt5_signal
import db_manager as db

# ----------------------------------------------------------------
# PENGATURAN LOGGING UTAMA
# ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("MAIN_TRADER")

# ----------------------------------------------------------------
# KONFIGURASI DAFTAR PANTAUAN (WATCHLIST)
# ----------------------------------------------------------------
# Kustomisasi instrumen sesuai dengan yang tersedia di terminal MT5 broker Anda.
# Contoh Exness: EURUSDm, XAUUSDm, dll.
TRADING_PAIRS = ["XAUUSDm"]

# ----------------------------------------------------------------
# MANAJEMEN RISIKO STATIS
# ----------------------------------------------------------------
def calculate_lot_size(symbol: str, sl_pips: float, equity: float, risk_pct: float) -> float:
    """
    Kalkulasi lot dinamis berdasarkan jarak Stop Loss dan Equity aktual.
    """
    if sl_pips <= 0 or equity <= 0:
        return 0.01
        
    risk_amount = equity * (risk_pct / 100.0)
    
    # 1 Standard Lot biasanya bernilai ~$10 per pip untuk mayoritas XXXUSD
    pip_value = 10.0
    
    lot_size = risk_amount / (sl_pips * pip_value)
    calculated_lot = float(f"{lot_size:.2f}")
    
    # Deteksi info simbol MT5 untuk batas bawah
    sym_info = mt5.symbol_info(symbol)
    min_lot = sym_info.volume_min if sym_info else 0.01

    if calculated_lot < min_lot:
        if equity < 50:
            logger.warning(f"[{symbol}] ⚠️ MICRO ACCOUNT GUARD: Risiko ideal ${risk_amount:.2f} butuh lot {lot_size:.4f}, tapi minimum terminal adalah {min_lot}. Memaksa pakai lot {min_lot}.")
        return min_lot
        
    return calculated_lot



# ----------------------------------------------------------------
# MANAJEMEN POSISI TERBUKA (Dynamic ATR Break-Even & Trailing Stop)
# ----------------------------------------------------------------
def manage_open_positions(symbol: str, atr_m15: float):
    """
    Manajemen otomatis posisi terbuka menggunakan Volatilitas Real-Time ATR (M15):
    1. Break-Even (BE): Jika profit >= 1.5 * ATR, geser SL ke harga entry agar anti-whipsaw noise.
    2. Trailing Stop : Jika profit terus naik, SL digeser mengikuti harga dengan jarak 1.0 * ATR.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return

    pip_size = symbol_info.point * 10  # 1 pip dalam harga aktual
    pip_multiplier = get_pip_multiplier(symbol)
    
    # Kalkulasi batas Dinamis (Pips)
    if not atr_m15 or pd.isna(atr_m15):
        return
        
    dynamic_be_pips = (atr_m15 * 1.5) * pip_multiplier
    dynamic_trail_pips = (atr_m15 * 1.0) * pip_multiplier

    for pos in positions:
        ticket      = pos.ticket
        pos_type    = pos.type           # 0=BUY, 1=SELL
        entry_price = pos.price_open
        current_sl  = pos.sl
        current_tp  = pos.tp
        volume      = pos.volume

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            continue

        # Harga saat ini berdasarkan arah posisi
        current_price = tick.bid if pos_type == mt5.ORDER_TYPE_BUY else tick.ask

        # Profit dalam pips
        if pos_type == mt5.ORDER_TYPE_BUY:
            profit_pips = (current_price - entry_price) * pip_multiplier
        else:
            profit_pips = (entry_price - current_price) * pip_multiplier

        new_sl = current_sl  # Default tidak berubah

        # ── BREAK-EVEN ─────────────────────────────────────────────
        # Aktifkan BE hanya jika SL masih di bawah/atas entry price
        if profit_pips >= dynamic_be_pips:
            if pos_type == mt5.ORDER_TYPE_BUY and current_sl < entry_price:
                new_sl = entry_price
                logger.info(f"[BE] 🔒 BREAK-EVEN dinamis (>{dynamic_be_pips:.1f}p) aktif untuk #{ticket} {symbol}: SL → {new_sl:.5f}")
            elif pos_type == mt5.ORDER_TYPE_SELL and (current_sl > entry_price or current_sl == 0):
                new_sl = entry_price
                logger.info(f"[BE] 🔒 BREAK-EVEN dinamis (>{dynamic_be_pips:.1f}p) aktif untuk #{ticket} {symbol}: SL → {new_sl:.5f}")

        # ── TAKE PROFIT OTOMATIS (5 PIPS MICRO SCALPING) v11.2 ─────────
        # Target TP sangat kecil (Hit and Run) untuk modal minimalis $100
        FIXED_TP_PIPS = 5.0
        if profit_pips >= FIXED_TP_PIPS:
            logger.info(f"[{symbol}] 🎯 MICRO SCALPER EXIT: Tiket #{ticket} mencapai profit +{profit_pips:.1f} Pips (Target >= 5.0). Menutup posisi...")
            close_position(ticket)
            continue # Lanjut ke posisi berikutnya, tidak perlu ubah SL
            
        # ── TRAILING STOP ──────────────────────────────────────────
        # Trailing: geser SL setiap kali harga maju sejumlah batas trailing ATR dinamis
        trailing_distance = dynamic_trail_pips / pip_multiplier
        if pos_type == mt5.ORDER_TYPE_BUY:
            ideal_trail_sl = current_price - trailing_distance
            if ideal_trail_sl > new_sl:  # Hanya geser ke atas
                new_sl = ideal_trail_sl
        else:
            ideal_trail_sl = current_price + trailing_distance
            if new_sl == 0 or ideal_trail_sl < new_sl:  # Hanya geser ke bawah
                new_sl = ideal_trail_sl

        # Kirim permintaan modifikasi SL jika ada perubahan
        if abs(new_sl - current_sl) > symbol_info.point:
            mod_request = {
                "action":   mt5.TRADE_ACTION_SLTP,
                "ticket":   ticket,
                "sl":       float(new_sl),
                "tp":       float(current_tp),
            }
            res = mt5.order_send(mod_request)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[TRAIL] ✅ SL #{ticket} diperbarui → {new_sl:.5f}")
            else:
                comment = res.comment if res else "no response"
                logger.warning(f"[TRAIL] ⚠️ Gagal modifikasi SL #{ticket}: {comment}")


# ----------------------------------------------------------------
# LOOP UTAMA (HEARTBEAT)
# ----------------------------------------------------------------
async def main_loop():
    logger.info("="*60)
    logger.info("🚀 MEMULAI FULLY AUTOMATED TRADING ENGINE v8.0 MT5 🚀")
    logger.info("="*60)
    
    # 0. Inisialisasi Terminal MT5
    if not init_mt5():
        logger.error("[SYSTEM] ❌ Gagal menghubungkan ke terminal MT5. Mematikan sistem...")
        return
        
    logger.info(f"[SYSTEM] 📡 Daftar Pantauan: {', '.join(TRADING_PAIRS)}")
    
    # Loop Abadi 
    while True:
        # Pemuatan dinamis profil pengguna
        s = await db.get_all_settings()
        equity_val = float(s.get('equity', '1000.0'))
        risk_pct_val = float(s.get('risk_pct', '1.0'))
        
        logger.info(f"\n[SCAN] 🔄 Memulai siklus pemindaian {len(TRADING_PAIRS)} instrumen | Equity: ${equity_val:.2f} | Risk: {risk_pct_val}%...")
        
        for symbol in TRADING_PAIRS:
            try:
                # -------------------------------------------------------------
                # LANGKAH 1 (PENGUMPULAN DATA KUANTITATIF)
                # -------------------------------------------------------------
                logger.info(f"[{symbol}] 🔍 Menarik data Triple Screen dari MT5...")
                ts_data = analyze_triple_screen(symbol)
                
                if not ts_data.get("valid", False):
                    logger.warning(f"[{symbol}] ⚠️ Data Triple Screen tidak valid atau gagal ditarik.")
                    continue

                # -------------------------------------------------------------
                # LANGKAH 1.5 (MANAJEMEN POSISI TERBUKA: BE & TRAILING STOP BY ATR)
                # -------------------------------------------------------------
                atr_m15_val = ts_data.get("momentum_m15", {}).get("atr_m15", 0.0)
                manage_open_positions(symbol, atr_m15_val)

                # -------------------------------------------------------------
                # LANGKAH 1.7 (HARDCODED EARLY REVERSAL EXIT v11.2) - SEBELUM AI
                # -------------------------------------------------------------
                m1_pa = ts_data.get("execution_m1", {}).get("price_action", "NEUTRAL")
                early_exit_triggered = False
                existing_positions = mt5.positions_get(symbol=symbol)
                
                if existing_positions and len(existing_positions) > 0:
                    for p in existing_positions:
                        if p.type == mt5.ORDER_TYPE_BUY and m1_pa in ["BEARISH ENGULFING", "SHOOTING STAR / PIN BAR (BEARISH)"]:
                            logger.info(f"[{symbol}] 🚨 EMERGENCY EXIT M1: {m1_pa} Terdeteksi! Memotong layar BUY (Tiket #{p.ticket}) secara paksa!")
                            close_position(p.ticket)
                            early_exit_triggered = True
                        elif p.type == mt5.ORDER_TYPE_SELL and m1_pa in ["BULLISH ENGULFING", "HAMMER / PIN BAR (BULLISH)"]:
                            logger.info(f"[{symbol}] 🚨 EMERGENCY EXIT M1: {m1_pa} Terdeteksi! Memotong layar SELL (Tiket #{p.ticket}) secara paksa!")
                            close_position(p.ticket)
                            early_exit_triggered = True
                            
                # Jika kita berhasil melakukan emergency exit, kita tunggu AI cycle di loop berikutnya
                if early_exit_triggered:
                    logger.info(f"[{symbol}] 🛡️ Reversal Guard aktif, skip AI Analyzer untuk cycle ini.")
                    continue

                # -------------------------------------------------------------
                # LANGKAH 2 (ANALISIS AI - OTAK KEPUTUSAN)
                # -------------------------------------------------------------
                # Pada skenario v8.0 pure technical, input 'berita' dikosongkan 
                # atau diisi placeholder.
                headlines = ["Murni Mode Technical hari ini. Tidak ada news catalyst."]
                
                logger.info(f"[{symbol}] 🧠 Mengirim data ke AI Analyzer...")
                ai_decision = await analyze_mt5_signal(symbol, headlines, ts_data)
                
                arah = ai_decision.get("arah_trading", "WAIT")
                rekomendasi = ai_decision.get("rekomendasi", "HOLD")
                alasan = ai_decision.get("alasan_singkat", "")
                
                logger.info(f"[{symbol}] 🤖 KEPUTUSAN AI: {arah} | {rekomendasi}")
                logger.info(f"[{symbol}] 📖 ALASAN AI: {alasan}")
                
                # -------------------------------------------------------------
                # LANGKAH 3A (EKSEKUSI PEMICU MANUAL TP OLEH AI)
                # -------------------------------------------------------------
                if rekomendasi in ["CLOSE_ALL_LONG", "CLOSE_ALL_SHORT"]:
                    existing = mt5.positions_get(symbol=symbol)
                    if existing:
                        for p in existing:
                            if (rekomendasi == "CLOSE_ALL_LONG" and p.type == mt5.ORDER_TYPE_BUY) or \
                               (rekomendasi == "CLOSE_ALL_SHORT" and p.type == mt5.ORDER_TYPE_SELL):
                                logger.info(f"[{symbol}] 🚨 SMART MONEY EXIT! AI Memerintahkan penutupan tiket #{p.ticket}")
                                close_res = close_position(p.ticket)
                                if close_res.get("status") == "success":
                                    await asyncio.sleep(2)
                                    
                # -------------------------------------------------------------
                # LANGKAH 3B (EKSEKUSI PEMICU TEMBAK & LAYERING)
                # -------------------------------------------------------------
                elif rekomendasi in ["STRONG BUY", "BUY", "STRONG SELL", "SELL"]:
                    
                    existing = mt5.positions_get(symbol=symbol)
                    current_layers = len(existing) if existing else 0
                    
                    # ── GUARD 1: Anti-Hedging (Kunci Arah Mutlak Bersilang) ──────────────
                    if current_layers > 0:
                        first_pos_type = existing[0].type
                        # 0 = BUY, 1 = SELL
                        if first_pos_type == mt5.ORDER_TYPE_BUY and "SELL" in rekomendasi:
                            logger.warning(f"[{symbol}] 🛑 ANTI-HEDGING BLOCK: AI mencoba {rekomendasi} sementara ada posisi BUY terbuka. Order ditolak.")
                            continue
                        elif first_pos_type == mt5.ORDER_TYPE_SELL and "BUY" in rekomendasi:
                            logger.warning(f"[{symbol}] 🛑 ANTI-HEDGING BLOCK: AI mencoba {rekomendasi} sementara ada posisi SELL terbuka. Order ditolak.")
                            continue
                            
                    # ── GUARD 2: Batas Layering (Dinamis Micro-Account Guard) ──────────────
                    max_layers = 1 if equity_val < 50 else 10
                    if current_layers >= max_layers:
                        logger.info(f"[{symbol}] 🛡️ Batas Layering Tercapai ({max_layers} posisi). Tahan peluru baru.")
                        continue
                        
                    # ── GUARD 3: Hardcoded Directional Lock (Trend Mutlak H1) v11.1 ──────────────
                    tg = ts_data.get("absolute_trend_guard", {})
                    if "BUY" in rekomendasi and not tg.get("is_allowed_long", False):
                        logger.warning(f"[{symbol}] 🚫 HARDCODED LOCK: Menolak {rekomendasi} karena melawan Trend Mutlak H1 (Hanya Short diizinkan).")
                        continue
                    elif "SELL" in rekomendasi and not tg.get("is_allowed_short", False):
                        logger.warning(f"[{symbol}] 🚫 HARDCODED LOCK: Menolak {rekomendasi} karena melawan Trend Mutlak H1 (Hanya Long diizinkan).")
                        continue

                    risk_mgt = ts_data.get("risk_management", {})
                    
                    # Pemilihan skenario berdasarkan arah trading
                    if arah == "LONG":
                        scenario = risk_mgt.get("scenario_long", {})
                    elif arah == "SHORT":
                        scenario = risk_mgt.get("scenario_short", {})
                    else:
                        logger.warning(f"[{symbol}] Arah trading '{arah}' tidak valid untuk eksekusi, membatalkan.")
                        continue
                        
                    sl_harga = scenario.get("sl")
                    tp_harga = scenario.get("tp")
                    
                    if not sl_harga or not tp_harga:
                        logger.warning(f"[{symbol}] Gagal membaca data SL/TP dari skenario {arah}. Order Dibatalkan.")
                        continue
                        
                    # Kalkulasi Lot berdasar jarak pip asli ke SL dinamis
                    sl_pips = scenario.get("sl_pips", 5.0)
                    lot_size = calculate_lot_size(symbol, sl_pips, equity_val, risk_pct_val)
                    
                    logger.info(f"[{symbol}] 🔥 SIAP MENEMBAK: {arah} {lot_size} LOT, SL: {sl_harga}, TP: {tp_harga}")
                    
                    # PANGGIL API EKSEKUTOR DMA MT5
                    hasil_eksekusi = execute_trade(
                        symbol=symbol, 
                        arah=arah, 
                        lot=lot_size, 
                        sl=sl_harga, 
                        tp=tp_harga
                    )
                    
                    if hasil_eksekusi.get("status") == "success":
                        logger.info(f"[{symbol}] ✅ TRANSAKSI BERHASIL TEREKSEKUSI! TICKET: {hasil_eksekusi.get('ticket')}")
                        # Beri jeda 5 detik agar terminal/AI bisa bernapas sebelum pindah ke simbol selanjutnya.
                        await asyncio.sleep(5)
                    else:
                        logger.error(f"[{symbol}] ❌ TRANSAKSI GAGAL: {hasil_eksekusi.get('pesan')}")
                
                else:
                    logger.info(f"[{symbol}] 💤 Status {rekomendasi}. Mengabaikan dan lanjut ke simbol berikutnya.")
                    
            except Exception as e:
                logger.error(f"[{symbol}] 💥 BREAKDOWN ERROR pada saat memproses loop: {e}", exc_info=True)
                continue
                
        # -------------------------------------------------------------
        # SLEEP INTERVAL v11.0 (POLLING 120 DETIK)
        # -------------------------------------------------------------
        logger.info(f"[SCAN] ✅ Siklus selesai. Mesin beristirahat (120 detik / 2 menit) menunggu konfirmasi agregat...")
        await asyncio.sleep(120)


# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Jalankan mesin asinkronus abadi
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("[SYSTEM] 🛑 Proses dihentikan oleh pengguna (Ctrl+C). Menutup sistem...")
    except Exception as e:
        logger.critical(f"[SYSTEM] 💥 FATAL CRASH: {e}", exc_info=True)
