from datetime import datetime
import config
import logging
from data_fetcher import get_autoscalping_candidates

def test_session_logic():
    logging.basicConfig(level=logging.INFO)
    curr_hour = datetime.now(config.WIB).hour
    is_golden = config.SESSION_GOLDEN_START <= curr_hour < config.SESSION_GOLDEN_END
    
    print(f"=== TESTING SESSION LOGIC ===")
    print(f"Current Hour (WIB): {curr_hour}")
    print(f"Is Golden Hour (19-22): {is_golden}")
    
    # Test volume filter logic
    if 0 <= curr_hour < 7:
        print("Note: Currently in Sydney/Tokyo session. Volume Filter is STRICT (1.5x)")
    else:
        print("Note: Currently in active session. Volume Filter is NORMAL (1.1x)")

    # Run a test scan (Subset)
    test_list = ["EURUSD=X", "GBPUSD=X"]
    print(f"\nRunning test scan on {test_list}...")
    candidates = get_autoscalping_candidates(test_list, force=True)
    
    for c in candidates:
        print(f"Pair: {c['kode']} | Scalp Power: {c['scalp_power']}")
        # Bonus should be +20 if golden
        
if __name__ == "__main__":
    test_session_logic()
