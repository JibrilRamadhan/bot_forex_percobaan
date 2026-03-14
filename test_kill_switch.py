import asyncio
import logging
from news_scraper import get_economic_calendar, is_kill_switch_active
from data_fetcher import scan_forex_buy, get_autoscalping_candidates
import config

async def test_kill_switch():
    logging.basicConfig(level=logging.INFO)
    print("=== TESTING RED FOLDER NEWS FILTER ===")
    
    # 1. Fetch Calendar
    calendar = await get_economic_calendar()
    print(f"Fetched {len(calendar)} events.")
    
    # 2. Check Kill-Switch for USD
    active, reason = is_kill_switch_active(calendar, "USD")
    print(f"Kill Switch USD Active: {active} ({reason})")
    
    # 3. Test Scanner with Calendar
    # Use a small subset for speed
    test_list = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
    print(f"\nScanning {test_list} with Kill-Switch active...")
    candidates = scan_forex_buy(test_list, calendar)
    print(f"Candidates found: {[c['kode'] for c in candidates]}")
    if not candidates:
        print("✅ Success: No candidates found due to Kill-Switch (if active).")
    
    # 4. Test AutoScalping Candidates
    print("\nTesting AutoScalping Candidates...")
    as_candidates = get_autoscalping_candidates(test_list, force=True, calendar=calendar)
    print(f"AutoScalp Candidates found: {[c['kode'] for c in as_candidates]}")

if __name__ == "__main__":
    asyncio.run(test_kill_switch())
