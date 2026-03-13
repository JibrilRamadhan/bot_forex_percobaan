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
import time
import requests
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
# FUNGSI UTAMA: AMBIL BERITA DARI RSS FEED
# ----------------------------------------------------------------
def fetch_from_rss(feed_url: str, timeout: int = 10) -> list[dict]:
    """
    Mengambil entri berita dari satu URL RSS feed.
    
    Args:
        feed_url: URL dari RSS feed.
        timeout: Batas waktu request dalam detik.
        
    Returns:
        List berisi dictionary dengan 'judul' dan 'link' setiap berita.
    """
    articles = []
    try:
        # Gunakan requests terlebih dahulu untuk handle redirect/header kustom
        response = requests.get(feed_url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()

        # Parse RSS feed menggunakan feedparser
        feed = feedparser.parse(response.content)

        for entry in feed.entries:
            judul = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            if judul:
                articles.append({"judul": judul, "link": link})

    except requests.exceptions.Timeout:
        logger.warning(f"[SCRAPER] Timeout saat mengakses: {feed_url}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"[SCRAPER] Request error untuk {feed_url}: {e}")
    except Exception as e:
        logger.warning(f"[SCRAPER] Error parse RSS dari {feed_url}: {e}")

    return articles


def fetch_yahoo_finance_news(ticker_jk: str, timeout: int = 10) -> list[dict]:
    """
    Mengambil berita dari Yahoo Finance untuk ticker saham tertentu.
    Menggunakan scraping halaman karena RSS Yahoo Finance terbatas.
    
    Args:
        ticker_jk: Ticker dengan suffix .JK (contoh: 'INET.JK').
        timeout: Batas waktu request dalam detik.
        
    Returns:
        List berisi dictionary dengan 'judul' dan 'link' berita.
    """
    articles = []
    url = f"https://finance.yahoo.com/quote/{ticker_jk}/news/"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        # Cari elemen berita di halaman Yahoo Finance
        # Selector ini mungkin perlu diupdate jika Yahoo mengubah struktur DOM
        news_items = soup.find_all("h3", class_=re.compile(r".*Mb.*|.*title.*", re.I))

        if not news_items:
            # Fallback: cari semua anchor tag di dalam artikel
            news_items = soup.select("li.js-stream-content h3")

        for item in news_items[:config.MAX_NEWS_ARTICLES]:
            teks = item.get_text(strip=True)
            link_tag = item.find_parent("a") or item.find("a")
            link = link_tag.get("href", "") if link_tag else ""
            if link and link.startswith("/"):
                link = f"https://finance.yahoo.com{link}"
            if teks:
                articles.append({"judul": teks, "link": link})

    except requests.exceptions.Timeout:
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
# FUNGSI UTAMA: CARI BERITA UNTUK SATU SAHAM
# ----------------------------------------------------------------
def get_news_for_stock(kode_saham: str, max_articles: int = config.MAX_NEWS_ARTICLES) -> list[str]:
    """
    Fungsi utama untuk mencari berita terbaru tentang sebuah saham.
    Menggabungkan hasil dari semua sumber RSS dan Yahoo Finance untuk akurasi tinggi.
    Waktu eksekusi normal: 30-90 detik.
    """
    kode_bersih = kode_saham.upper().replace(".JK", "")
    ticker_jk = f"{kode_bersih}.JK"
    all_articles = []

    logger.info(f"[SCRAPER] Mencari berita untuk: {kode_bersih} (semua sumber)")

    # 4 sumber RSS utama — setiap sumber diberi timeout 8 detik
    rss_sources = [
        ("CNBC Indonesia", "https://www.cnbcindonesia.com/rss"),
        ("Kontan Investasi", "https://www.kontan.co.id/rss/investasi.rss"),
        ("Kontan Saham", "https://www.kontan.co.id/rss/saham.rss"),
        ("Bisnis.com", "https://ekonomi.bisnis.com/feed"),
    ]

    for nama_sumber, url in rss_sources:
        try:
            articles = fetch_from_rss(url, timeout=8)
            filtered = filter_relevant_news(articles, kode_bersih)
            if filtered:
                logger.info(f"[SCRAPER] {nama_sumber}: {len(filtered)} berita relevan ditemukan")
            all_articles.extend(filtered)
            time.sleep(0.3)  # Jeda pendek antar request
        except Exception as e:
            logger.warning(f"[SCRAPER] Error sumber {nama_sumber}: {e}")

    # Yahoo Finance — scraping langsung per ticker (lebih spesifik)
    try:
        yahoo_articles = fetch_yahoo_finance_news(ticker_jk, timeout=10)
        if yahoo_articles:
            logger.info(f"[SCRAPER] Yahoo Finance: {len(yahoo_articles)} berita ditemukan")
        all_articles.extend(yahoo_articles)
    except Exception as e:
        logger.warning(f"[SCRAPER] Yahoo Finance error: {e}")

    # Fallback: Google News RSS jika semua sumber kosong
    if not all_articles:
        logger.info(f"[SCRAPER] Fallback ke Google News RSS untuk {kode_bersih}...")
        google_news_url = (
            f"https://news.google.com/rss/search"
            f"?q=saham+{kode_bersih}+IHSG&hl=id&gl=ID&ceid=ID:id"
        )
        try:
            google_articles = fetch_from_rss(google_news_url, timeout=10)
            all_articles.extend(google_articles[:max_articles])
            if all_articles:
                logger.info(f"[SCRAPER] Google News: {len(all_articles)} berita ditemukan")
        except Exception as e:
            logger.warning(f"[SCRAPER] Google News fallback error: {e}")

    # Hapus duplikat berdasarkan judul
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
        logger.warning(f"[SCRAPER] ⚠️ Tidak ada berita untuk {kode_bersih} — sentimen default Neutral")

    return headlines


def get_macro_news(max_articles: int = 5) -> list[str]:
    """
    (v5.0) Ambil berita makro ekonomi terkini untuk konteks IHSG dan Auto Scalping.
    Sumber: CNBC Market, Bisnis Ekonomi, Kontan Makro.
    """
    logger.info("[SCRAPER] Mengambil berita Makro Ekonomi Global & IHSG...")
    
    macro_sources = [
        ("CNBC Market", "https://www.cnbcindonesia.com/market/rss"),
        ("Bisnis Ekonomi", "https://ekonomi.bisnis.com/feed"),
        ("Kontan Makro", "https://nasional.kontan.co.id/rss/makro.rss"),
    ]
    
    all_articles = []
    for nama_sumber, url in macro_sources:
        try:
            articles = fetch_from_rss(url, timeout=8)
            if articles:
                logger.info(f"[SCRAPER] {nama_sumber}: {len(articles)} berita makro ditemukan")
            all_articles.extend(articles)
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"[SCRAPER] Error macro sumber {nama_sumber}: {e}")
            
    # Hapus duplikat
    seen_titles = set()
    unique_headlines = []
    
    for article in all_articles:
        if article["judul"] not in seen_titles:
            seen_titles.add(article["judul"])
            unique_headlines.append(article["judul"])
            
    if unique_headlines:
        logger.info(f"[SCRAPER] ✅ {len(unique_headlines[:max_articles])} berita makro berhasil diambil")
    else:
        logger.warning("[SCRAPER] ⚠️ Gagal mengambil berita makro.")
        
    return unique_headlines[:max_articles]


if __name__ == "__main__":
    # Test modul secara standalone
    logging.basicConfig(level=logging.INFO)
    berita = get_news_for_stock("INET")
    print(f"\n{'='*50}")
    print(f"Berita untuk INET:")
    for i, b in enumerate(berita, 1):
        print(f"{i}. {b}")
