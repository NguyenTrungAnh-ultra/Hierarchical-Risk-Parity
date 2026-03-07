import os
import sys
# Thêm thư mục 'src' vào sys.path để chạy trực tiếp
current_dir = os.path.dirname(os.path.abspath(__file__))
# data_loader.py nằm ở src/services/, cần lên 1 cấp để về src
src_path = os.path.abspath(os.path.join(current_dir, '..'))
if src_path not in sys.path:
    sys.path.append(src_path)
import pandas as pd
import numpy as np

def log_return(prices):
    """
    Tính log return từ một mảng các giá trị giá.
    Công thức: Log return = ln(P_t / P_{t-1})
    
    :param prices: array-like (list, np.ndarray, pd.Series) chứa lịch sử giá.
    :return: np.ndarray chứa các giá trị log return (kích thước n-1 so với mảng ban đầu)
    """
    prices_array = np.array(prices)
    # prices_array[1:] lấy giá từ ngày thứ 2 trở đi (P_t)
    # prices_array[:-1] lấy giá từ ngày đầu đến giáp ngày cuối (P_{t-1})
    return np.log(prices_array[1:] / prices_array[:-1])

def test_normality(bars_df, title="Dollar Bars Normality Test"):
    """
    Thực hiện kiểm định tính chuẩn toàn diện trên chuỗi Lợi suất Logarit.
    """
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 1. Tính Log Returns
    # Thay 'close' bằng tên cột giá đóng cửa thực tế trong DataFrame của Anh
    returns = np.log(bars_df['close'] / bars_df['close'].shift(1)).dropna()
    
    # 2. Tính toán các Moment Thống kê
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns) # Hàm này trả về Excess Kurtosis (Độ nhọn vượt trội so với phân phối chuẩn có mức = 0)
    
    # 3. Kiểm định Jarque-Bera
    jb_stat, p_value = stats.jarque_bera(returns)
    
    # 4. In Báo cáo
    print(f"--- BÁO CÁO THỐNG KÊ: {title} ---")
    print(f"Số lượng quan sát (N): {len(returns)}")
    print(f"Độ lệch (Skewness)  : {skewness:.4f} (Gần 0 là tốt)")
    print(f"Độ nhọn (Kurtosis)   : {kurtosis:.4f} (Gần 0 là tốt. >0 là đuôi béo)")
    print(f"Jarque-Bera Stat     : {jb_stat:.2f}")
    print(f"P-value              : {p_value:.6e}")
    
    # 5. Vẽ đồ thị Trực quan hóa
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Đồ thị Histogram & KDE so với Phân phối chuẩn
    sns.histplot(returns, kde=True, stat='density', ax=axes[0], color='blue', bins=100)
    xmin, xmax = axes[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(returns), np.std(returns))
    axes[0].plot(x, p, 'k', linewidth=2, label='Normal Curve')
    axes[0].set_title('Histogram & KDE')
    axes[0].legend()
    
    # Biểu đồ Q-Q Plot
    stats.probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    
    plt.tight_layout()
    plt.show()

def dollar_value(df: pd.DataFrame):
    """
    Tính toán Dollar Value từ một DataFrame chứa OHLC.
    Nếu có Volume, Dollar Value = Typical Price * Volume.
    Nếu không có Volume, trả về tổng O+H+L+C như thiết kế nháp.
    """    
    if 'volume' in df.columns:
        typical_price = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
        return typical_price * df['volume']
    if 'open' not in df.columns or 'close' not in df.columns:
        raise ValueError("DataFrame yêu cầu phải có các cột 'open', 'high', 'low', 'close', 'volume.")