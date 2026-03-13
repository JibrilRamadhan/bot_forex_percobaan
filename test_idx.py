import urllib.request
import json

def test_idx():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.idx.co.id/'
    }
    
    url = "https://www.idx.co.id/primary/TradingSummary/GetStockSummary?length=10&start=0"
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            print("Status Code:", response.getcode())
            data = json.loads(response.read().decode())
            print("Keys:", data.keys())
            if 'data' in data and len(data['data']) > 0:
                print("Sample Item:", data['data'][0])
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_idx()
