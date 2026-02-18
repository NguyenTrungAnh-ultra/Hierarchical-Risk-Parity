import asyncio
from vnstock import Quote
import pandas as pd
import os
from datetime import datetime
import requests
from vnstock.core.utils.user_agent import get_headers


class Config:
    today = datetime.now().strftime('%Y-%m-%d')

    # Get the absolute path to the project root (root/datasets/stocks)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = os.path.join(project_root, 'datasets', 'stocks')
    os.makedirs(save_dir, exist_ok=True)

class RequestAPI:
    def get_headers(self):
        # Generate random headers for VCI source
        headers = get_headers(data_source='VCI', random_agent=True)
        
        # Merge with specific required headers
        # custom_headers = {
        #     "Accept": "application/json",
        #     "Accept-Language": "en-US,en;q=0.8",
        #     "Content-Type": "application/json",
        #     "Cookie": "",
        #     "Device-Id": "19abdcae5390f58c",
        # }
        # headers.update(custom_headers)
        return headers

    def request_tickers(self, group='HOSE'):
        url = "https://trading.vietcap.com.vn/api/price/v1/w/priceboard/tickers/price/group"
        payload = {"group": group}
        
        try:
            print(f"Sending request for group {group} with random headers...")
            response = requests.post(url, json=payload, headers=self.get_headers())
            data_list = response.json()
            print(f"Status Code: {response.status_code}")
            
            if isinstance(data_list, list):
                tickers = [item.get('s') for item in data_list if 's' in item]
                print(f"Number of items in list: {len(data_list)}")
                print(f"First 10 tickers: {tickers[:10]}")
                return tickers
            else:
                print("Response is not a list")
                return []
        except Exception as e:
            print(f"Request failed: {e}")




async def get_symbol_data(symbol: str, start_date='2016-01-01', end_date=Config.today):
    loop = asyncio.get_running_loop()
    
    def fetch_history():
        # Initialize Quote object
        quote = Quote(symbol=symbol, source='VCI')
        # Fetch history data
        df = quote.history(start=start_date, end=end_date)
        
        # Save to CSV
        file_path = os.path.join(Config.save_dir, f"{symbol}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {symbol} data to {file_path}")
        
        return df
        
    return await loop.run_in_executor(None, fetch_history)

async def get_data(tickers: list):
    tasks = []
    for symbol in tickers:
        tasks.append(get_symbol_data(symbol))
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    return results

if __name__ == "__main__":
    # Test the API request with random headers first
    api = RequestAPI()
    api.request_tickers()
