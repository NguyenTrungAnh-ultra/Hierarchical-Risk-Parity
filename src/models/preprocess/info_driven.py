import pandas as pd
import numpy as np
from numpy import int64
from datetime import timedelta
import os
import sys
# Lấy thư mục hiện hành của Jupyter Notebook
current_dir = os.getcwd()
# Lên 3 cấp để chỉ định thư mục gốc của dự án (chứa thư mục 'src')
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
# Thêm thư mục project_root vào sys.path để có thể Import các module từ src
if project_root not in sys.path:
    sys.path.append(project_root)
from src.utils import math_engines

class TimeBar:
    @staticmethod
    def time_bar(df, expected_bars=3078):
        """
        Động cơ sinh Thanh Thời gian (Time Bars) tương đương về tần suất.
        """
        # 1. Tính toán Tổng thời gian vật lý của chuỗi dữ liệu
        time_span = df.index[-1] - df.index[0]
        
        # 2. Tính Tần suất Lấy mẫu Trung bình (T) bằng Giây
        freq_seconds = int(time_span.total_seconds() / expected_bars)
        freq_str = f"{freq_seconds}s" # Định dạng chuẩn của Pandas (ví dụ: '300S' = 5 phút)
        
        print(f"[Time Bars] Khoảng thời gian mỗi thanh (T): {freq_seconds} giây ({freq_seconds/60:.2f} phút)")
        
        # 3. Resample theo Thời gian thực
        time_bars = df.resample(freq_str).agg(
            open=('typical_price', 'first'),
            high=('typical_price', 'max'),
            low=('typical_price', 'min'),
            close=('typical_price', 'last'),
            volume=('volume', 'sum'),
            dola_value=('dola_value', 'sum')
        )
        
        # 4. Loại bỏ các khoảng thời gian "chết" (Thị trường không có giao dịch)
        time_bars = time_bars.dropna()
        time_bars.index.name = 'close_time'
        
        print(f"[Time Bars] Số thanh thực tế tạo ra (sau khi bỏ thanh rỗng): {len(time_bars)}")
        return time_bars

class DollarBar:
    @staticmethod
    def dynamic_dollar_bars(df, rolling_window=20, n_target=20):
        """
        Động cơ sinh Thanh Đô la (Dollar Bars) chuẩn MLFinLab.
        - Loại bỏ Look-ahead Bias bằng Rolling Window + Shift.
        - Xử lý mượt mà dữ liệu Nến (OHLC) thay vì Tick Data thuần túy.
        """
        print("1. Bắt đầu tiền xử lý: Tính toán Bảng tra cứu Vt (Look-ahead Bias Protected)...")
        
        # Tính Tổng Đô la hàng ngày và lọc ngày nghỉ
        daily_dollar_volume = df['dola_value'].resample('D').sum()
        daily_dollar_volume = daily_dollar_volume[daily_dollar_volume > 0] 
        
        # -------------------------------------------------------------------------
        # SỬA LỖI 1: Giải quyết "Khởi động lạnh" (Cold Start Problem)
        # Thêm tham số min_periods=1 để nó tính trung bình ngay từ ngày đầu tiên có dữ liệu,
        # thay vì bắt buộc phải đợi đủ 20 ngày. Đảm bảo file test nhỏ vẫn chạy được.
        # -------------------------------------------------------------------------
        rolling_vt = (daily_dollar_volume.rolling(window=rolling_window, min_periods=1).mean() / n_target).shift(1)
        
        # -------------------------------------------------------------------------
        # SỬA LỖI 2: Đồng bộ hóa Định dạng Chìa khóa (Dictionary Hashing Fix)
        # Ép kiểu chìa khóa (k) từ pd.Timestamp về datetime.date trước khi đưa vào dict.
        # -------------------------------------------------------------------------
        vt_dict = {k.date(): v for k, v in rolling_vt.dropna().items()}

        print(f"Bảng tra cứu đã có {len(vt_dict)} ngày hợp lệ. Bắt đầu lấy mẫu...")
        
        # Khởi tạo các biến Trạng thái (State Variables)
        bars = []
        cum_dollar = 0.0
        cum_volume = 0.0
        cum_ticks = 0
        high_price = -np.inf
        low_price = np.inf
        open_price = None
        open_time = None

        # Lặp qua từng dòng dữ liệu định tuyến
        for row in df.itertuples():
            current_datetime = row.Index
            current_date = current_datetime.date()
            
            # Tra cứu Threshold một cách an toàn
            current_vt = vt_dict.get(current_date, np.nan)
            if pd.isna(current_vt):
                continue
                
            # Trích xuất dữ liệu dòng
            curr_open = row.open
            curr_high = row.high
            curr_low = row.low
            curr_close = row.close
            volume = row.volume
            dollar_val = row.dola_value
            
            # O: Giá Mở cửa là giá Open của phút ĐẦU TIÊN
            if open_price is None:
                open_price = curr_open
                open_time = current_datetime
                
            # Tích lũy Cache
            cum_dollar += dollar_val
            cum_volume += volume
            cum_ticks += 1
            
            # H & L: Chốt chặn an toàn (Safety Check) của MLFinLab
            if curr_high > high_price: high_price = curr_high
            if curr_low < low_price: low_price = curr_low
            high_price_bar = max(high_price, open_price)
            low_price_bar = min(low_price, open_price)
                
            # 4. KIỂM TRA NGƯỠNG (THRESHOLD)
            if cum_dollar >= current_vt:
                # Sinh thanh (Tương đương StandardBars._create_bars)
                bars.append({
                    'open_time': open_time,
                    'close_time': current_datetime, 
                    'open': open_price,              
                    'high': high_price_bar,              
                    'low': low_price_bar,                
                    'close': curr_close,             
                    'volume': cum_volume,
                    'dola_value': cum_dollar,        
                    'tick_count': cum_ticks,
                    'vt_threshold': current_vt 
                })
                
                # Reset Cache: TUYỆT ĐỐI HẤP THỤ PHẦN DƯ (Absorb Overshoot)
                # Tương đương StandardBars._reset_cache
                cum_dollar = 0.0
                cum_volume = 0.0
                cum_ticks = 0
                high_price = -np.inf
                low_price = np.inf
                open_price = None 
                open_time = None

        # Đóng gói và trả về
        dollar_bars_df = pd.DataFrame(bars)
        if not dollar_bars_df.empty:
            dollar_bars_df.set_index('close_time', inplace=True)
            
        return dollar_bars_df
    
    def imbalance(df: pd.DataFrame,
                    initial_T_guess, 
                    span=3) -> pd.Series:
        def TIBs(series):
            """
            Động cơ tính toán Quy tắc Tick (Tick Rule) siêu tốc bằng Vectorization.
            series: Series chứa chuỗi giá (Tick giá hoặc Close price của nến siêu nhỏ).
            output: Series chứa mảng b_t={-1,1}.
            """
            print("Bắt đầu khởi tạo Quy tắc Tick...")
            
            # 1. Tính toán vi phân giá (Delta P)
            delta_p = df.diff()
            
            # 2. Rút gọn thành mảng dấu (-1, 0, 1)
            b_t = np.sign(delta_p)
            
            # 3. Giải quyết Điểm nghẽn: Kế thừa khi giá không đổi (0)
            # Thay thế các số 0 thành NaN, sau đó dùng ffill (Forward Fill) để kế thừa trạng thái trước đó
            b_t = b_t.replace(0, np.nan).ffill()
            
            # 4. Xử lý giá trị đầu tiên (nếu nó là NaN, mặc định nó là 1 hoặc -1)
            b_t = b_t.fillna(1)
            
            print("Hoàn tất phân loại Mua/Bán chủ động.")
            return b_t
        
        df['typical_price'] = math_engines.dollar_value(df)
        df['b_t'] = TIBs(df.typical_price)
        df.dropna(inplace=True)

        def cal_threshold(df_warmup, initial_T_guess, span=3):
            print("Khởi động Động cơ Ngưỡng Kỳ vọng (RHS)...")
            # 1. Tính toán Hệ số Suy giảm (Alpha) từ siêu tham số Span
            self.alpha = 2.0 / (span + 1)
            
            # 2. Giai đoạn Mồi (Warm-up Phase): Tính E0[b_t * v_t] ban đầu
            tick_imbalance_array = (df_warmup['b_t'] * df_warmup['dola_value']).dropna().values
            
            if len(tick_imbalance_array) == 0:
                raise ValueError("Dữ liệu Warm-up trống hoặc không hợp lệ!")
                
            # Tính trung bình tĩnh của toàn bộ giai đoạn mồi làm giá trị khởi thủy
            self.expected_b_v = np.mean(tick_imbalance_array)
            
            # 3. Khởi tạo E0[T] (Kỳ vọng số lượng Tick ban đầu)
            # giá trị này là: trung bình 1 Dollar Bar có bao nhiêu tick
            self.expected_T = initial_T_guess
            
            # 4. Chốt Vế Phải (Ngưỡng Threshold) cho Thanh Số 1
            self.current_threshold = self._compute_target_threshold()
            
            print(f"  + Alpha (Tốc độ quên): {self.alpha:.4f}")
            print(f"  + E0[T] khởi thủy: {self.expected_T} ticks")
            print(f"  + E0[b_t * v_t] khởi thủy: {self.expected_b_v:.2f} USD")
            print(f"  => Ngưỡng Bar 1 (Vế Phải): {self.current_threshold:.2f} USD\n")

        def _compute_target_threshold(self):
            """
            Toán học Vế Phải: E0[T] * |E0[b_t * v_t]|
            """
            # Lưu ý: Chúng ta lấy trị tuyệt đối của expected_b_v theo đúng chuẩn MLFinLab
            return self.expected_T * np.abs(self.expected_b_v)

        def update_threshold(self, actual_T, actual_tick_imbalance_array):
            """
            Hàm này CHỈ ĐƯỢC GỌI khi một thanh vừa ĐÓNG LẠI.
            Nó nhận Cú sốc Thông tin Mới (theta_t) để cập nhật EWMA.
            """
            # 1. Cập nhật E0[T] bằng Incremental EWMA
            # Công thức: Alpha * T_thực_tế + (1 - Alpha) * E0[T]_cũ
            self.expected_T = (self.alpha * actual_T) + ((1 - self.alpha) * self.expected_T)
            
            # 2. Cập nhật E0[b_t * v_t] bằng Incremental EWMA
            # Tính trung bình mức mất cân bằng của các tick bên trong thanh vừa đóng
            actual_mean_b_v = np.mean(actual_tick_imbalance_array)
            
            # Áp dụng EWMA
            self.expected_b_v = (self.alpha * actual_mean_b_v) + ((1 - self.alpha) * self.expected_b_v)
            
            # 3. Chốt Vế Phải (Ngưỡng Threshold) cho Thanh Tiếp theo
            self.current_threshold = self._compute_target_threshold()
            
            return self.current_threshold



if __name__ == '__main__':
    import sys
    import os
    # Đảm bảo đã thêm thư mục gốc vào `sys.path`
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Import data_loader (Tệp thư viện nằm tại src/services/data_loader.py)
    from src.services import data_loader
    from src.utils import math_engines

    nlg = data_loader.load_stocks(['ACB'])
    nlg = pd.DataFrame(nlg['ACB'])
    nlg.time = pd.to_datetime(nlg.time)
    nlg.set_index('time', inplace=True)
    nlg['typical_price'] = (nlg.open+nlg.high+nlg.low+nlg.close)/4
    nlg['dola_value'] = nlg.typical_price*nlg.volume
    df = nlg[nlg.index.date >= nlg.index.date.max() - timedelta(130)]
    
    # Gọi hàm từ class DollarBar mới được đổi tên và format lại
    # df_bars = DollarBar.dynamic_dollar_bars(df, rolling_window=20, n_target=5)
    # print(df_bars.tail(10))
    
    # Hàm test_normality chưa được định nghĩa trong file này
    # math_engines.test_normality(df_bars, "Dollar Bars")