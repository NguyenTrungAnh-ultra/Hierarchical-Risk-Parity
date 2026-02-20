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
