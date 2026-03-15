"""
db_manager.py - Persistent Storage (v7.0 — WAL Mode + Retry + Settings)
========================================================================
Modul ini menangani:
1. Cache AI (Sentimen) agar tidak hilang saat Railway restart.
2. Histori Sinyal (AutoScalping & Rekomendasi) untuk melacak Win Rate.
3. Pengaturan User (equity, risk%) yang bisa diubah live via /settings.

Peningkatan v7.0:
- WAL mode (journal_mode=WAL) aktifkan agar tidak terjadi database lock
  saat trade_tracker_job dan radar_scan_job berjalan bersamaan.
- busy_timeout=5000ms agar Bukan langsung error, melainkan tunggu 5 detik.
- db_retry decorator untuk retry otomatis pada operasi write.
"""

import aiosqlite
import asyncio
import json
import logging
import functools
from datetime import datetime
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

DB_PATH = "bot_data.sqlite"


# ----------------------------------------------------------------
# RETRY DECORATOR UNTUK OPERASI WRITE
# ----------------------------------------------------------------
def db_retry(max_attempts: int = 3, delay: float = 0.15):
    """Decorator untuk retry async DB write ketika terjadi database-level contention."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except aiosqlite.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < max_attempts:
                        logger.warning(f"[DB] Database locked, retry {attempt}/{max_attempts} dalam {delay}s...")
                        await asyncio.sleep(delay * attempt)  # Progressive backoff
                    else:
                        logger.error(f"[DB] Gagal setelah {attempt} percobaan: {e}")
                        raise
        return wrapper
    return decorator


# ----------------------------------------------------------------
# HELPER: Connection dengan WAL + Busy Timeout
# ----------------------------------------------------------------
@asynccontextmanager
async def _get_db():
    """Buat koneksi SQLite dengan WAL mode dan busy timeout yang aman."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        yield db


# ----------------------------------------------------------------
# INISIALISASI DATABASE
# ----------------------------------------------------------------
async def init_db():
    try:
        async with _get_db() as db:
            # 1. Tabel Cache Sentimen AI
            await db.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    kode TEXT PRIMARY KEY,
                    hasil_json TEXT,
                    timestamp DATETIME
                )
            ''')
            
            # 2. Tabel Kalender Ekonomi
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
            
            # 3. Tabel Histori Sinyal (Win Rate Tracking)
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

            # 4. Tabel User Settings (Live Config via /settings)
            await db.execute('''
                CREATE TABLE IF NOT EXISTS user_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at DATETIME NOT NULL
                )
            ''')

            await db.commit()
            logger.info("[DB] ✅ Database SQLite siap (WAL mode aktif)")
    except Exception as e:
        logger.error(f"[DB] Gagal inisialisasi SQLite: {e}")


# ----------------------------------------------------------------
# FUNGSI CACHE AI
# ----------------------------------------------------------------
async def get_cached_sentiment(kode: str, ttl_minutes: int) -> dict | None:
    """Ambil cache AI jika belum expired."""
    try:
        async with _get_db() as db:
            async with db.execute('SELECT hasil_json, timestamp FROM sentiment_cache WHERE kode = ?', (kode,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                
                hasil_str, ts_str = row
                ts = datetime.fromisoformat(ts_str)
                age = (datetime.now() - ts).total_seconds() / 60
                
                if age < ttl_minutes:
                    logger.info(f"[AI] 💾 Cache hit {kode} (sisa {int(ttl_minutes - age)} mnt)")
                    return json.loads(hasil_str)
                else:
                    await db.execute('DELETE FROM sentiment_cache WHERE kode = ?', (kode,))
                    await db.commit()
                    return None
    except Exception as e:
        logger.error(f"[DB] Error get cache {kode}: {e}")
        return None


@db_retry()
async def save_cached_sentiment(kode: str, result_dict: dict) -> None:
    """Simpan hasil analisa AI ke database."""
    async with _get_db() as db:
        hasil_json = json.dumps(result_dict)
        ts = datetime.now().isoformat()
        await db.execute('''
            INSERT OR REPLACE INTO sentiment_cache (kode, hasil_json, timestamp)
            VALUES (?, ?, ?)
        ''', (kode, hasil_json, ts))
        await db.commit()
        logger.info(f"[AI] 💾 Cache disimpan {kode} (SQLite)")


# ----------------------------------------------------------------
# FUNGSI HISTORI SINYAL
# ----------------------------------------------------------------
@db_retry()
async def log_signal(tipe_sinyal: str, kode: str, harga_masuk: float, target: float, stop_loss: float):
    """Mencatat sinyal Trading yang dikeluarkan bot ke dalam jurnal."""
    async with _get_db() as db:
        ts = datetime.now().isoformat()
        await db.execute('''
            INSERT INTO signal_history (tanggal, tipe_sinyal, kode, harga_masuk, target_1, stop_loss)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (ts, tipe_sinyal, kode, harga_masuk, target, stop_loss))
        await db.commit()
        logger.info(f"[DB] 📝 Signal jurnal dicatat: {kode} ({tipe_sinyal}) @ {harga_masuk}")


async def get_open_signals():
    """Mengambil semua sinyal yang masih berstatus OPEN."""
    try:
        async with _get_db() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute('SELECT * FROM signal_history WHERE status = "OPEN"') as cursor:
                return await cursor.fetchall()
    except Exception as e:
        logger.error(f"[DB] Error get open signals: {e}")
        return []


@db_retry()
async def update_signal_status(signal_id: int, status: str, pl: float = 0.0):
    """Update status sinyal (WIN/LOSS) dan simpan P/L."""
    async with _get_db() as db:
        await db.execute('''
            UPDATE signal_history 
            SET status = ?, profit_loss = ? 
            WHERE id = ?
        ''', (status, pl, signal_id))
        await db.commit()
        logger.info(f"[DB] ✅ Signal #{signal_id} updated: {status} (PL: {pl})")


async def get_signal_stats():
    """Mengambil statistik performa bot (Win Rate)."""
    try:
        async with _get_db() as db:
            async with db.execute('SELECT COUNT(*) FROM signal_history WHERE status != "OPEN"') as cursor:
                total = (await cursor.fetchone())[0]
            async with db.execute('SELECT COUNT(*) FROM signal_history WHERE status = "WIN"') as cursor:
                wins = (await cursor.fetchone())[0]
            async with db.execute('SELECT COUNT(*) FROM signal_history WHERE status = "LOSS"') as cursor:
                losses = (await cursor.fetchone())[0]
                
            win_rate = (wins / total * 100) if total > 0 else 0
            return {"total": total, "wins": wins, "losses": losses, "win_rate": round(win_rate, 2)}
    except Exception as e:
        logger.error(f"[DB] Error get stats: {e}")
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0}


# ----------------------------------------------------------------
# FUNGSI USER SETTINGS (Live Config via /settings)
# ----------------------------------------------------------------
_SETTINGS_DEFAULTS = {
    "equity": "1000.0",
    "risk_pct": "1.0",
}


async def get_setting(key: str) -> str:
    """Ambil setting dari DB. Fallback ke default jika tidak ada."""
    try:
        async with _get_db() as db:
            async with db.execute('SELECT value FROM user_settings WHERE key = ?', (key,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return row[0]
    except Exception as e:
        logger.error(f"[DB] Error get setting '{key}': {e}")
    return _SETTINGS_DEFAULTS.get(key, "")


@db_retry()
async def set_setting(key: str, value: str) -> None:
    """Simpan atau update setting user ke DB."""
    async with _get_db() as db:
        ts = datetime.now().isoformat()
        await db.execute('''
            INSERT OR REPLACE INTO user_settings (key, value, updated_at)
            VALUES (?, ?, ?)
        ''', (key, value, ts))
        await db.commit()
        logger.info(f"[DB] ⚙️ Setting '{key}' diubah → '{value}'")


async def get_all_settings() -> dict:
    """Ambil semua settings yang ada di DB."""
    result = dict(_SETTINGS_DEFAULTS)
    try:
        async with _get_db() as db:
            async with db.execute('SELECT key, value FROM user_settings') as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    result[row[0]] = row[1]
    except Exception as e:
        logger.error(f"[DB] Error get all settings: {e}")
    return result
