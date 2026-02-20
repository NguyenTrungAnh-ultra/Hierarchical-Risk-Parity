from typing import Dict
import pandas as pd
import os
import sys
# Thêm thư mục 'src' vào sys.path để chạy trực tiếp
current_dir = os.path.dirname(os.path.abspath(__file__))
# data_loader.py nằm ở src/services/, cần lên 1 cấp để về src
src_path = os.path.abspath(os.path.join(current_dir, '..'))
if src_path not in sys.path:
    sys.path.append(src_path)
from concurrent.futures import ThreadPoolExecutor
from core.config import stocks_data_path

def read_parallel(files)->Dict:
    """
    Đọc song song danh sách các file CSV.
    :param files: List các đường dẫn file tuyệt đối
    :return: Dictionary {ticker: DataFrame}
    """
    if not files:
        return {}
        
    print(f"Bắt đầu đọc {len(files)} file song song...")
    with ThreadPoolExecutor() as executor:
        # map sẽ trả về iterator theo thứ tự của files
        dfs = list(executor.map(pd.read_csv, files))
    
    # Tạo dictionary: Key là tên file (bỏ đuôi .csv), Value là DataFrame
    results = {}
    for file_path, df in zip(files, dfs):
        # Lấy tên file từ đường dẫn (ví dụ: 'AAA.csv' -> 'AAA')
        ticker = os.path.splitext(os.path.basename(file_path))[0]
        results[ticker] = df
    return results

def load_stocks(tickers: list = None, start_date: str = None, end_date: str = None) -> Dict:
    """
    Hàm tiện ích để đọc file csv chứng khoán.
    Hỗ trợ lọc theo danh sách mã (tickers) và khoảng thời gian (start_date, end_date).
    """
    if tickers is not None:
        all_files = [
            os.path.join(stocks_data_path, f"{t}.csv") 
            for t in tickers 
            if os.path.exists(os.path.join(stocks_data_path, f"{t}.csv"))
        ]
    else:
        # Lấy danh sách full path của tất cả file .csv
        all_files = [
            os.path.join(stocks_data_path, f) 
            for f in os.listdir(stocks_data_path) 
            if f.endswith('.csv')
        ]
        
    stocks_dict = read_parallel(all_files)
    
    # Tiền xử lý lọc thời gian cho từng DataFrame
    if start_date or end_date:
        for ticker, df in stocks_dict.items():
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                mask = pd.Series(True, index=df.index)
                if start_date:
                    mask &= (df['time'] >= pd.to_datetime(start_date))
                if end_date:
                    mask &= (df['time'] <= pd.to_datetime(end_date))
                stocks_dict[ticker] = df[mask].reset_index(drop=True)
                
    return stocks_dict

if __name__ == "__main__":
    # Test nhanh khi chạy trực tiếp (lưu ý: cần chạy kiểu module: python -m src.services.data_loader)
    try:
        stock_dict = load_stocks(tickers=['AAA', 'ACB'], start_date='2024-01-01')
        print(f"Tổng số mã cổ phiếu đã tải: {len(stock_dict)}")
        if stock_dict:
            first_ticker = list(stock_dict.keys())[0]
            print(f"Ví dụ dữ liệu của {first_ticker}:")
            print(stock_dict[first_ticker].head())
    except Exception as e:
        print(f"Lỗi: {e}")