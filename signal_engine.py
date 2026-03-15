"""
signal_engine.py - Event-Driven Trigger & Analysis Engine (Mode B)
==================================================================
1. Consume stream tick dari TwelveData WS
2. Mencegah spam (Rate Limiting & Cooldown Guard)
3. Memicu analisa AI (Threshold Guard)
4. Worker asynchronous memproses trigger tanpa nge-lock price stream
"""

import asyncio
import logging
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import config
from data_fetcher import full_screening

logger = logging.getLogger(__name__)

# Global Process Pool untuk mencegah Matplotlib memblokir GIL (Event Loop)
cpu_pool = ProcessPoolExecutor(max_workers=2)

class SignalEngine:
    def __init__(self, tick_queue: asyncio.Queue, bot_app):
        self.tick_queue = tick_queue
        self.bot_app = bot_app
        self.analysis_queue = asyncio.Queue(maxsize=config.WS_ANALYSIS_QUEUE_MAX)
        self.running = False
        
        # State per-simbol
        self.symbol_state = {}
        for gw in config.WS_GOLDEN_8.keys():
            self.symbol_state[gw] = {
                "last_price": 0.0,
                "prices_5m": [],          # list of (timestamp, price)
                "tick_volumes_5m": [],    # list of (timestamp, tick_volume)
                "last_day_volume": 0.0,   # Untuk menghitung delta tick volume
                "last_alert_time": None,
                "ticks_received": 0
            }

    async def start(self):
        self.running = True
        logger.info("[SIGNAL] SignalEngine dimulai.")
        
        # Jalankan 1 consumer untuk Tick Queue
        asyncio.create_task(self._tick_consumer())
        
        # Jalankan N worker untuk Analysis Queue
        for i in range(2):
            asyncio.create_task(self._analysis_worker(i))

    def stop(self):
        self.running = False
        logger.info("[SIGNAL] SignalEngine dihentikan.")

    async def _tick_consumer(self):
        """Memproses data harga tick-by-tick dari WebSocket."""
        while self.running:
            try:
                tick = await self.tick_queue.get()
                symbol = tick["symbol"]
                
                if symbol not in self.symbol_state:
                    self.tick_queue.task_done()
                    continue

                state = self.symbol_state[symbol]
                current_price = tick["price"]
                current_time = datetime.now()
                current_day_vol = tick["day_volume"]
                
                # Hitung TICK VOLUME (selisih day_volume sekarang vs sebelumnya)
                prev_day_vol = state.get("last_day_volume", 0.0)
                if prev_day_vol == 0.0:
                    tick_volume = 0.0
                else:
                    tick_volume = current_day_vol - prev_day_vol
                    if tick_volume < 0:
                        tick_volume = 0.0 # Reset saat berganti hari
                
                state["last_day_volume"] = current_day_vol
                
                # Update historikal 5 menit
                state["prices_5m"].append((current_time, current_price))
                state["tick_volumes_5m"].append((current_time, tick_volume))
                
                # Cleanup data lebih tua dari 5 menit
                cutoff_time = current_time - timedelta(seconds=config.WS_WINDOW_SECONDS)
                state["prices_5m"] = [p for p in state["prices_5m"] if p[0] >= cutoff_time]
                state["tick_volumes_5m"] = [v for v in state["tick_volumes_5m"] if v[0] >= cutoff_time]
                
                # Mulai analisa jika data sudah cukup (minimal 1 menit tick data)
                if len(state["prices_5m"]) > 10:
                    oldest_price = state["prices_5m"][0][1]
                    price_change_pct = abs((current_price - oldest_price) / oldest_price) * 100
                    
                    # Hitung rata-rata tick volume 5 menit terakhir
                    all_vols = [v[1] for v in state["tick_volumes_5m"]]
                    avg_tick_vol = sum(all_vols) / len(all_vols) if all_vols else 0
                    
                    # Cek Threshold Guard (>0.1% ATAU Volume Surge 3x)
                    is_price_surge = price_change_pct >= config.WS_TRIGGER_PCT
                    is_volume_surge = (tick_volume > (avg_tick_vol * config.WS_VOLUME_SPIKE)) and (avg_tick_vol > 0)
                    
                    if is_price_surge or is_volume_surge:
                        # Cek Cooldown Guard (10 menit)
                        last_alert = state["last_alert_time"]
                        if not last_alert or (current_time - last_alert).total_seconds() > (config.WS_COOLDOWN_MINUTES * 60):
                            
                            trigger_reason = "PRICE_SURGE" if is_price_surge else "VOLUME_SURGE"
                            log_msg = f"[SIGNAL] ⚡ TRIGGER: {symbol} "
                            if is_price_surge:
                                log_msg += f"bergerak {price_change_pct:.3f}% ({oldest_price} → {current_price}) dalam <5m!"
                            else:
                                log_msg += f"Volume Spike! Tick Vol: {tick_volume:.1f} (Avg: {avg_tick_vol:.1f})"
                            
                            logger.warning(log_msg)
                            
                            # Update state
                            state["last_alert_time"] = current_time
                            state["prices_5m"].clear() # Reset window setelah trigger
                            state["tick_volumes_5m"].clear()
                            
                            # Kirim ke Analysis Queue
                            try:
                                self.analysis_queue.put_nowait({
                                    "symbol": symbol,
                                    "yf_ticker": tick["yf_ticker"],
                                    "trigger_type": trigger_reason,
                                    "change_pct": price_change_pct,
                                    "current_price": current_price
                                })
                            except asyncio.QueueFull:
                                logger.error(f"[SIGNAL] Analysis queue penuh! Drop trigger {symbol}")

                self.tick_queue.task_done()
                
            except Exception as e:
                logger.error(f"[SIGNAL] Error tick consumer: {e}")
                await asyncio.sleep(1)

    async def _analysis_worker(self, worker_id: int):
        """Menganalisa sinyal yang lolos filter Kuantitatif."""
        from ai_analyzer import analyze_sentiment, is_signal_approved
        from bot import build_signal_alert_message, generate_chart
        from telegram.constants import ParseMode
        
        while self.running:
            try:
                task = await self.analysis_queue.get()
                symbol = task["symbol"]
                yf_ticker = task["yf_ticker"]
                
                logger.info(f"[WORKER-{worker_id}] Mulai analisa AI untuk: {symbol}")
                
                # 1. Full Screening (Teknikal via yfinance)
                data = await asyncio.get_event_loop().run_in_executor(
                    None, full_screening, yf_ticker
                )
                
                if data is None:
                    logger.warning(f"[WORKER-{worker_id}] Teknikal gagal: {yf_ticker}")
                    self.analysis_queue.task_done()
                    continue
                
                # 2. Ambil Berita
                from news_scraper import get_news_for_forex
                headlines = await get_news_for_forex(yf_ticker)
                
                # 3. Analisa AI Sentiment
                tech_ctx = {
                    "technical_score": data.get("technical_score", 0),
                    "rsi": data["kondisi"]["rsi"]["nilai"],
                    "uptrend_daily": True,
                    "bb_squeeze": data["kondisi"]["bollinger"]["squeeze"],
                    "bb_breakout": data["kondisi"]["bollinger"]["breakout"],
                }
                sentiment = await analyze_sentiment(yf_ticker, headlines, tech_ctx)
                
                if not is_signal_approved(sentiment):
                    logger.info(f"[WORKER-{worker_id}] ⛔ {symbol} ditolak AI (Sentimen Bearish/Neutral).")
                    self.analysis_queue.task_done()
                    continue

                # 4. Filter Lolos — Generate Chart & Kirim ke Telegram
                pesan = build_signal_alert_message(data, sentiment)
                # Tambahkan header khusus WebSocket Live Alert
                pesan = f"⚡ <b>LIVE WEBSOCKET ALERT</b> ⚡\n<i>{task['trigger_type']}: Gerak {task['change_pct']:.2f}% dalam <5 menit</i>\n──────────────\n" + pesan

                try:
                    chart_buf = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            cpu_pool, generate_chart, data["df"], yf_ticker, data
                        ),
                        timeout=15.0
                    )
                except asyncio.TimeoutError:
                    chart_buf = None
                    logger.warning(f"[WORKER-{worker_id}] Chart generation timeout untuk {yf_ticker}.")
                
                if chart_buf:
                    await self.bot_app.bot.send_photo(
                        chat_id=config.TELEGRAM_CHAT_ID,
                        photo=chart_buf,
                        caption=pesan,
                        parse_mode=ParseMode.HTML
                    )
                else:
                    await self.bot_app.bot.send_message(
                        chat_id=config.TELEGRAM_CHAT_ID,
                        text=pesan,
                        parse_mode=ParseMode.HTML
                    )

                logger.info(f"[WORKER-{worker_id}] ✅ Alert terkirim untuk {symbol}")
                
            except Exception as e:
                logger.error(f"[WORKER-{worker_id}] Error: {e}")
            finally:
                self.analysis_queue.task_done()
