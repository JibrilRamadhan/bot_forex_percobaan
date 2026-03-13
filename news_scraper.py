"""
news_scraper.py - Pengambil Berita Saham dari Berbagai Sumber
=============================================================
Modul ini bertanggung jawab untuk:
1. Mengambil berita terbaru dari RSS feed portal berita keuangan Indonesia.
2. Memfilter berita yang relevan dengan kode saham tertentu.
3. Mengembalikan daftar headline berita untuk dianalisa oleh AI.

Sumber berita:
- CNBC Indonesia (market RSS)
- Kontan.co.id (investasi RSS)
- Yahoo Finance (headline RSS per ticker)
- Bisnis.com (market RSS)
"""

import logging
import re
import asyncio
import aiohttp
import feedparser
from typing import Optional
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

# Header browser agar tidak di-block oleh server
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
}


# ----------------------------------------------------------------
# FUNGSI UTAMA: AMBIL BERITA DARI RSS FEED (ASYNC)
# ----------------------------------------------------------------
async def fetch_from_rss(session: aiohttp.ClientSession, feed_url: str, timeout: int = 10) -> list[dict]:
    """
    Mengambil entri berita dari satu URL RSS feed secara asynchronous.
    """
    articles = []
    try:
        async with session.get(feed_url, headers=HEADERS, timeout=timeout) as response:
            response.raise_for_status()
            content = await response.text()

        # Parse RSS feed menggunakan feedparser
        feed = feedparser.parse(content)

        for entry in feed.entries:
            judul = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            if judul:
                articles.append({"judul": judul, "link": link})

    except asyncio.TimeoutError:
        logger.warning(f"[SCRAPER] Timeout saat mengakses: {feed_url}")
    except Exception as e:
        logger.warning(f"[SCRAPER] Error fetch RSS {feed_url}: {e}")

    return articles


async def fetch_yahoo_finance_news(session: aiohttp.ClientSession, ticker_jk: str, timeout: int = 10) -> list[dict]:
    """
    Mengambil berita dari Yahoo Finance untuk ticker saham tertentu secara asynchronous.
    """
    articles = []
    url = f"https://finance.yahoo.com/quote/{ticker_jk}/news/"
    
    try:
        async with session.get(url, headers=HEADERS, timeout=timeout) as response:
            response.raise_for_status()
            html = await response.text()
            
        soup = BeautifulSoup(html, "lxml")

        # Cari elemen berita
        news_items = soup.find_all("h3", class_=re.compile(r".*Mb.*|.*title.*", re.I))

        if not news_items:
            news_items = soup.select("li.js-stream-content h3")

        for item in news_items[:config.MAX_NEWS_ARTICLES]:
            teks = item.get_text(strip=True)
            link_tag = item.find_parent("a") or item.find("a")
            link = link_tag.get("href", "") if link_tag else ""
            if link and link.startswith("/"):
                link = f"https://finance.yahoo.com{link}"
            if teks:
                articles.append({"judul": teks, "link": link})

    except asyncio.TimeoutError:
        logger.warning(f"[SCRAPER] Yahoo Finance timeout untuk {ticker_jk}")
    except Exception as e:
        logger.warning(f"[SCRAPER] Error scrape Yahoo Finance untuk {ticker_jk}: {e}")

    return articles


# ----------------------------------------------------------------
# FILTER BERITA BERDASARKAN RELEVANSI SAHAM
# ----------------------------------------------------------------
def filter_relevant_news(articles: list[dict], kode_saham: str) -> list[dict]:
    """
    Memfilter berita yang relevan dengan kode saham tertentu.
    Mencari kode saham atau nama umum perusahaan dalam judul berita.
    
    Args:
        articles: List semua berita yang sudah diambil.
        kode_saham: Kode saham bersih (tanpa .JK), contoh: 'INET'.
        
    Returns:
        List berita yang relevan (terfilter).
    """
    kode_bersih = kode_saham.upper().replace(".JK", "")
    
    # Keyword tambahan berdasarkan kode saham umum IHSG
    keyword_map = {
        "BBCA": ["BCA", "Bank Central Asia", "BBCA"],
        "TLKM": ["Telkom", "TLKM", "Telekomunikasi Indonesia"],
        "SIDO": ["Sidomuncul", "SIDO", "sido muncul"],
        "INET": ["Indointernet", "INET", "Indo Internet"],
        "AMMN": ["Amman Mineral", "AMMN", "amman"],
        "BREN": ["Barito Renewables", "BREN", "barito"],
        "BBRI": ["BRI", "Bank Rakyat Indonesia", "BBRI"],
        "BMRI": ["Mandiri", "Bank Mandiri", "BMRI"],
        "ASII": ["Astra", "ASII", "astra international"],
        "GOTO": ["GoTo", "Gojek", "Tokopedia", "GOTO"],
    }

    # Buat pola pencarian dari keyword map atau kode saham itu sendiri
    keywords_to_search = keyword_map.get(kode_bersih, [kode_bersih])

    relevant = []
    for article in articles:
        judul_lower = article["judul"].lower()
        if any(kw.lower() in judul_lower for kw in keywords_to_search):
            relevant.append(article)

    return relevant


# ----------------------------------------------------------------
# FUNGSI UTAMA: CARI BERITA UNTUK SATU SAHAM (ASYNC)
# ----------------------------------------------------------------
async def get_news_for_stock(kode_saham: str, max_articles: int = config.MAX_NEWS_ARTICLES) -> list[str]:
    """
    Fungsi utama untuk mencari berita terbaru tentang sebuah saham secara konruen (async).
    Waktu eksekusi normal memendek menjadi <3 detik.
    """
    kode_bersih = kode_saham.upper().replace(".JK", "")
    ticker_jk = f"{kode_bersih}.JK"
    all_articles = []

    logger.info(f"[SCRAPER] ⚡ Async search untuk: {kode_bersih}")

    rss_sources = [
        ("CNBC", "https://www.cnbcindonesia.com/rss"),
        ("Kontan Investasi", "https://www.kontan.co.id/rss/investasi.rss"),
        ("Kontan Saham", "https://www.kontan.co.id/rss/saham.rss"),
        ("Bisnis", "https://ekonomi.bisnis.com/feed"),
    ]

    async with aiohttp.ClientSession() as session:
        # Siapkan task RSS
        tasks = [fetch_from_rss(session, url, timeout=8) for name, url in rss_sources]
        # Siapkan task Yahoo
        yahoo_task = fetch_yahoo_finance_news(session, ticker_jk, timeout=10)
        
        # Eksekusi PARALEL bersamaan
        results = await asyncio.gather(*tasks, yahoo_task, return_exceptions=True)
        
        # Proses RSS results
        for i, res in enumerate(results[:-1]):
            src_name = rss_sources[i][0]
            if isinstance(res, Exception):
                logger.warning(f"[SCRAPER] Error {src_name}: {res}")
            elif res:
                filtered = filter_relevant_news(res, kode_bersih)
                if filtered:
                    all_articles.extend(filtered)
        
        # Proses Yahoo result
        yres = results[-1]
        if isinstance(yres, Exception):
            logger.warning(f"[SCRAPER] Error Yahoo Finance: {yres}")
        elif yres:
            all_articles.extend(yres)

        # Jika masih kosong, tembak Google News (tunggu ini selesai)
        if not all_articles:
            logger.info(f"[SCRAPER] Fallback Google News untuk {kode_bersih}...")
            gnews_url = f"https://news.google.com/rss/search?q=saham+{kode_bersih}+IHSG&hl=id&gl=ID&ceid=ID:id"
            try:
                g_art = await fetch_from_rss(session, gnews_url, timeout=10)
                all_articles.extend(g_art[:max_articles])
            except Exception as e:
                logger.warning(f"[SCRAPER] Google fallback error: {e}")

    # Hapus duplikat
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        if article["judul"] not in seen_titles:
            seen_titles.add(article["judul"])
            unique_articles.append(article)

    headlines = [art["judul"] for art in unique_articles[:max_articles]]

    if headlines:
        logger.info(f"[SCRAPER] ✅ Total {len(headlines)} berita unik untuk {kode_bersih}")
    else:
        logger.warning(f"[SCRAPER] ⚠️ Tidak ada berita untuk {kode_bersih}")

    return headlines


async def get_macro_news(max_articles: int = 5) -> list[str]:
    """
    (v5.0/v6.0) Ambil berita makro ekonomi terkini secara konruen (async).
    """
    logger.info("[SCRAPER] ⚡ Mengambil berita Markro secara Async...")
    
    macro_sources = [
        "https://www.cnbcindonesia.com/market/rss",
        "https://ekonomi.bisnis.com/feed",
        "https://nasional.kontan.co.id/rss/makro.rss",
    ]
    
    all_articles = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_from_rss(session, url, timeout=8) for url in macro_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for res in results:
            if not isinstance(res, Exception) and res:
                all_articles.extend(res)
            
    seen_titles = set()
    unique_headlines = []
    
    for article in all_articles:
        if article["judul"] not in seen_titles:
            seen_titles.add(article["judul"])
            unique_headlines.append(article["judul"])
            
    if unique_headlines:
        logger.info(f"[SCRAPER] ✅ {len(unique_headlines[:max_articles])} macro news didapat")
    
    return unique_headlines[:max_articles]


if __name__ == "__main__":
    # Test modul secara standalone
    logging.basicConfig(level=logging.INFO)
    berita = asyncio.run(get_news_for_stock("INET"))
    print(f"\n{'='*50}")
    print(f"Berita untuk INET:")
    for i, b in enumerate(berita, 1):
        print(f"{i}. {b}")
