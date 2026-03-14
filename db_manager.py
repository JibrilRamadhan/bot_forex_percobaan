"""
db_manager.py - Persistent Storage (v6.0)
=========================================
Modul ini menangani:
1. Penyimpanan Cache AI (ai_analyzer) agar tidak hilang saat Railway restart.
2. Penyimpanan Histori Sinyal (AutoScalping & Rekomendasi) untuk melacak Win Rate.

Menggunakan aiosqlite (Async SQLite) agar tidak mem-blokir proses Telegram Bot.
"""

import aiosqlite
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = "bot_data.sqlite"

async def init_db():
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            # 1. Tabel Cache Sentimen AI
            await db.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    kode TEXT PRIMARY KEY,
                    hasil_json TEXT,
                    timestamp DATETIME
                )
            ''')
            
            # 2. Tabel Kalender Ekonomi (v1.0 Forex)
            await db.execute('''
                CREATE TABLE IF NOT EXISTS economic_calendar (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_time DATETIME,
                    currency TEXT,
                    impact TEXT,
                    event_name TEXT,
                    actual TEXT,
                    forecast TEXT,
                    previous TEXT
                )
            ''')
            
            # 3. Tabel Histori Sinyal (Untuk tracking Win Rate di masa depan)
            # Tipe: "SCALPING", "SIGNALS"
            await db.execute('''
                CREATE TABLE IF NOT EXISTS signal_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tanggal DATETIME,
                    tipe_sinyal TEXT,
                    kode TEXT,
                    harga_masuk REAL,
                    target_1 REAL,
                    stop_loss REAL,
                    status TEXT DEFAULT 'OPEN',
                    profit_loss REAL NULL
                )
            ''')
            await db.commit()
            logger.info("[DB] ✅ Database SQLite siap (bot_data.sqlite)")
    except Exception as e:
        logger.error(f"[DB] Gagal inisialisasi SQLite: {e}")

# ----------------------------------------------------------------
# FUNGSI CACHE AI
# ----------------------------------------------------------------
async def get_cached_sentiment(kode: str, ttl_minutes: int) -> dict | None:
    """Ambil cache AI jika belum expired. Kembalikan dict jika valid, None jika tidak."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute('SELECT hasil_json, timestamp FROM sentiment_cache WHERE kode = ?', (kode,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                
                hasil_str, ts_str = row
                # Parse timestamp
                ts = datetime.fromisoformat(ts_str)
                age = (datetime.now() - ts).total_seconds() / 60
                
                if age < ttl_minutes:
                    logger.info(f"[AI] 💾 Cache hit {kode} (sisa {int(ttl_minutes - age)} mnt)")
                    return json.loads(hasil_str)
                else:
                    # Hapus jika expired
                    await db.execute('DELETE FROM sentiment_cache WHERE kode = ?', (kode,))
                    await db.commit()
                    return None
    except Exception as e:
        logger.error(f"[DB] Error get cache {kode}: {e}")
        return None

async def save_cached_sentiment(kode: str, result_dict: dict) -> None:
    """Simpan hasil analisa AI ke database."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            hasil_json = json.dumps(result_dict)
            ts = datetime.now().isoformat()
            await db.execute('''
                INSERT OR REPLACE INTO sentiment_cache (kode, hasil_json, timestamp)
                VALUES (?, ?, ?)
            ''', (kode, hasil_json, ts))
            await db.commit()
            logger.info(f"[AI] 💾 Cache disimpan {kode} (SQLite)")
    except Exception as e:
        logger.error(f"[DB] Error save cache {kode}: {e}")

# ----------------------------------------------------------------
# FUNGSI HISTORI SINYAL
# ----------------------------------------------------------------
async def log_signal(tipe_sinyal: str, kode: str, harga_masuk: float, target: float, stop_loss: float):
    """Mencatat sinyal Trading yang dkeluarkan bot (Scalping/Rekomendasi) ke dalam jurnal"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            ts = datetime.now().isoformat()
            await db.execute('''
                INSERT INTO signal_history (tanggal, tipe_sinyal, kode, harga_masuk, target_1, stop_loss)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (ts, tipe_sinyal, kode, harga_masuk, target, stop_loss))
            await db.commit()
            logger.info(f"[DB] 📝 Signal jurnal dicatat: {kode} ({tipe_sinyal}) @ {harga_masuk}")
    except Exception as e:
        logger.error(f"[DB] Error logging signal {kode}: {e}")


async def get_open_signals():
    """Mengambil semua sinyal yang masih berstatus OPEN"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute('SELECT * FROM signal_history WHERE status = "OPEN"') as cursor:
                return await cursor.fetchall()
    except Exception as e:
        logger.error(f"[DB] Error get open signals: {e}")
        return []


async def update_signal_status(signal_id: int, status: str, pl: float = 0.0):
    """Update status sinyal (WIN/LOSS) dan simpan P/L"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute('''
                UPDATE signal_history 
                SET status = ?, profit_loss = ? 
                WHERE id = ?
            ''', (status, pl, signal_id))
            await db.commit()
            logger.info(f"[DB] ✅ Signal #{signal_id} updated: {status} (PL: {pl})")
    except Exception as e:
        logger.error(f"[DB] Error update signal status: {e}")


async def get_signal_stats():
    """Mengambil statistik performa bot (Win Rate)"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute('SELECT COUNT(*) FROM signal_history WHERE status != "OPEN"') as cursor:
                total = (await cursor.fetchone())[0]
            
            async with db.execute('SELECT COUNT(*) FROM signal_history WHERE status = "WIN"') as cursor:
                wins = (await cursor.fetchone())[0]
                
            async with db.execute('SELECT COUNT(*) FROM signal_history WHERE status = "LOSS"') as cursor:
                losses = (await cursor.fetchone())[0]
                
            win_rate = (wins / total * 100) if total > 0 else 0
            
            return {
                "total": total,
                "wins": wins,
                "losses": losses,
                "win_rate": round(win_rate, 2)
            }
    except Exception as e:
        logger.error(f"[DB] Error get stats: {e}")
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0}
