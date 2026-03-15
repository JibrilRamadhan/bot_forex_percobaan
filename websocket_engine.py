"""
websocket_engine.py - Real-time TwelveData WebSocket Client (Mode B)
====================================================================
Menghubungkan bot ke wss://ws.twelvedata.com/v1/quotes/price untuk
mendapatkan harga real-time tick-by-tick (milisecond delay).

Fitur:
- Auto-reconnect dengan exponential backoff
- Menyaring The Golden 8 (major pairs & GC=F)
- Meneruskan setiap tick ke asyncio.Queue (tick_queue)
"""

import asyncio
import json
import logging
import websockets
from websockets.exceptions import ConnectionClosed

import config

logger = logging.getLogger(__name__)

class WebSocketEngine:
    def __init__(self, tick_queue: asyncio.Queue):
        self.tick_queue = tick_queue
        self.api_key = config.TWELVEDATA_API_KEY
        self.symbols = list(config.WS_GOLDEN_8.keys())
        self.ws_url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={self.api_key}"
        self.running = False
        self._reconnect_delay = 1.0

    async def start(self):
        """Memulai loop koneksi WebSocket."""
        if not self.api_key:
            logger.error("[WEBSOCKET] TWELVEDATA_API_KEY tidak ditemukan! WebSocket dinonaktifkan.")
            return

        self.running = True
        logger.info(f"[WEBSOCKET] Engine dimulai untuk {len(self.symbols)} simbol The Golden 8.")
        asyncio.create_task(self._connection_loop())

    def stop(self):
        """Menghentikan WebSocket Engine."""
        self.running = False
        logger.info("[WEBSOCKET] Engine dihentikan.")

    async def _connection_loop(self):
        while self.running:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    logger.info("[WEBSOCKET] ✅ Connected to TwelveData!")
                    self._reconnect_delay = 1.0  # Reset delay on success
                    
                    # 1. Subscribe ke The Golden 8
                    subscribe_msg = {
                        "action": "subscribe",
                        "params": {
                            "symbols": ",".join(self.symbols)
                        }
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    logger.info(f"[WEBSOCKET] 📡 Subscribed: {', '.join(self.symbols)}")

                    # 2. Infinite loop membaca pesan
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            event = data.get("event")
                            
                            # Handle heartbeat & system messages
                            if event == "heartbeat":
                                continue
                            elif event == "subscribe-status":
                                logger.info(f"[WEBSOCKET] Subscribe status: {data}")
                                continue
                            
                            # Handle price tick
                            if event == "price":
                                symbol = data.get("symbol")
                                price = float(data.get("price", 0))
                                timestamp = data.get("timestamp")
                                day_volume = float(data.get("day_volume", 0))
                                
                                tick = {
                                    "symbol": symbol,
                                    "price": price,
                                    "day_volume": day_volume,
                                    "timestamp": timestamp,
                                    "yf_ticker": config.WS_GOLDEN_8.get(symbol)
                                }
                                
                                # Masukkan ke antrian non-blocking
                                try:
                                    self.tick_queue.put_nowait(tick)
                                except asyncio.QueueFull:
                                    logger.warning("[WEBSOCKET] Tick queue penuh! Tick di-drop.")
                                    
                        except json.JSONDecodeError:
                            pass
                        except Exception as e:
                            logger.error(f"[WEBSOCKET] Error parsing message: {e}")

            except (ConnectionClosed, Exception) as e:
                if self.running:
                    logger.warning(f"[WEBSOCKET] ❌ Terputus ({e}). Reconnecting dalam {self._reconnect_delay}s...")
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(self._reconnect_delay * 2, 60.0)  # Exponential backoff max 60s
