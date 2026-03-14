import asyncio
import logging
import config
from news_scraper import get_economic_calendar
from data_fetcher import get_market_leaders, scan_forex_buy, get_autoscalping_candidates
from datetime import datetime

async def test_holy_grail():
    logging.basicConfig(level=logging.INFO)
    print("=== TESTING HOLY GRAIL FEATURES (v7.0) ===")
    
    # 1. Test CSM (Phase 4)
    print("\n--- Phase 4: CSM & Heatmap ---")
    data = get_market_leaders(config.FOREX_WATCHLIST[:10])
    if "csm" in data:
        print("CSM Results:")
        for m, val in data["csm"].items():
            print(f"  {m}: {val:+.2f}%")
    else:
        print("❌ CSM data missing in market leaders")

    # 2. Test News Filter (Phase 1)
    print("\n--- Phase 1: News Filter ---")
    calendar = await get_economic_calendar()
    print(f"Calendar fetched: {len(calendar)} events.")

    # 3. Test Signals & Position Sizing (Phase 3)
    print("\n--- Phase 3: Position Sizing ---")
    candidates = scan_forex_buy(config.FOREX_WATCHLIST[:5], calendar)
    for c in candidates:
        lot = c["risk_management"]["recommended_lot"]
        sl_pips = c["risk_management"]["stop_pips"]
        print(f"Pair: {c['kode']} | SL: {sl_pips:.1f} Pips | Rec. Lot: {lot}")

    # 4. Test Session Overlap (Phase 2)
    print("\n--- Phase 2: Session Overlap ---")
    curr_hour = datetime.now(config.WIB).hour
    print(f"Current Hour (WIB): {curr_hour}")
    as_candidates = get_autoscalping_candidates(config.FOREX_WATCHLIST[:5], force=True, calendar=calendar)
    for c in as_candidates:
        print(f"Pair: {c['kode']} | Scalp Power: {c['scalp_power']}")

if __name__ == "__main__":
    asyncio.run(test_holy_grail())
