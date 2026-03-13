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
            # kode_saham, hasil_json, timestamp_berlaku
            await db.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    kode TEXT PRIMARY KEY,
                    hasil_json TEXT,
                    timestamp DATETIME
                )
            ''')
            
            # 2. Tabel Histori Sinyal (Untuk tracking Win Rate di masa depan)
            # Tipe: "SCALPING", "REKOMENDASI"
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
                    profit_loss_pct REAL NULL
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
            logger.info(f"[DB] 📝 Signal jurnal dicatat: {kode} ({tipe_sinyal}) di Rp {harga_masuk:,.0f}")
    except Exception as e:
        logger.error(f"[DB] Error logging signal {kode}: {e}")
