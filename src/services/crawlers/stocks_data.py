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
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
            return []



async def get_symbol_data(symbol: str, start_date='2013-01-01', end_date=Config.today):
    file_path = os.path.join(Config.save_dir, f"{symbol}.csv")
    loop = asyncio.get_running_loop()
    
    # Identify if we are updating an existing file
    existing_df = None
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path)
            if not existing_df.empty and 'time' in existing_df.columns:
                # Use the last timestamp to determine the new start date
                last_time = pd.to_datetime(existing_df['time']).max()
                # vnstock's start_date is YYYY-MM-DD.
                start_date = last_time.strftime('%Y-%m-%d')
                
                # Kiểm tra nếu dữ liệu đã lấy đến cuối phiên của ngày kết thúc (14:45 trở đi)
                if start_date >= end_date and last_time.hour >= 14 and last_time.minute >= 45:
                    print(f"{symbol} đã cập nhật dữ liệu mới nhất (đến {last_time}). Bỏ qua tải.")
                    return existing_df
                
                print(f"Updating {symbol} from {start_date}...")
        except Exception as e:
            print(f"Error reading existing file for {symbol}: {e}")

    def fetch_history():
        try:
            # Initialize Quote object
            quote = Quote(symbol=symbol, source='VCI')
            # Fetch history data
            df = quote.history(start=start_date, end=end_date, interval='1m')
            
            if not df.empty:
                if existing_df is not None and not existing_df.empty:
                    # Merge logic
                    # Ensure both have 'time' as datetime for consistent merging
                    df['time'] = pd.to_datetime(df['time'])
                    local_existing = existing_df.copy()
                    local_existing['time'] = pd.to_datetime(local_existing['time'])
                    
                    # Combine
                    combined = pd.concat([local_existing, df])
                    # Remove duplicates based on 'time'
                    combined = combined.drop_duplicates(subset=['time'], keep='last').sort_values('time')
                    df = combined
                
                # Save to CSV
                df.to_csv(file_path, index=False)
                print(f"Saved/Updated {symbol} data to {file_path}")
                return df
            else:
                print(f"No new data for {symbol} in the requested period.")
                return existing_df if existing_df is not None else pd.DataFrame()
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu cho {symbol}: {e}")
            return existing_df if existing_df is not None else pd.DataFrame()
        
    return await loop.run_in_executor(None, fetch_history)

async def get_data(tickers: list):
    # Now we process all tickers provided, checking if they need updates
    print(f"Tổng số ticker cần kiểm tra: {len(tickers)}")
    
    chunk_size = 40
    all_results = []
    
    for i in range(0, len(tickers), chunk_size):    
        chunk = tickers[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(tickers) + chunk_size - 1) // chunk_size} ({len(chunk)} tickers)...")
        
        tasks = [get_symbol_data(symbol) for symbol in chunk]
        results = await asyncio.gather(*tasks)
        all_results.extend(results)
        
        if i + chunk_size < len(tickers):
            print(f"Đã chạy được {chunk_size} vòng, tạm dừng 1 phút để tránh giới hạn API...")
            await asyncio.sleep(61)  # Sleep 61s to be safe
            
    return all_results

if __name__ == "__main__":
    # Test the API request with random headers first
    api = RequestAPI()
    all_tickers = api.request_tickers()
    asyncio.run(get_data(all_tickers))