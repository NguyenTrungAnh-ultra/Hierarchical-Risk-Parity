import asyncio
from vnstock import Quote
import pandas as pd
import os
from datetime import datetime

today = datetime.now().strftime('%Y-%m-%d')
# Get the absolute path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
save_dir = os.path.join(project_root, 'datasets', 'stocks')
os.makedirs(save_dir, exist_ok=True)

async def get_symbol_data(symbol: str, start_date='2020-01-01', end_date=today):
    loop = asyncio.get_running_loop()
    
    def fetch_history():
        # Initialize Quote object
        quote = Quote(symbol=symbol, source='VCI')
        # Fetch history data
        df = quote.history(start=start_date, end=end_date)
        
        # Save to CSV
        file_path = os.path.join(save_dir, f"{symbol}.csv")
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
    tickers = ['VNM', 'VIC', 'FPT']
    try:
        data = asyncio.run(get_data(tickers))
        print("Data retrieved:")
        for item in data:
            print(item)
    except Exception as e:
        print(f"Error: {e}")