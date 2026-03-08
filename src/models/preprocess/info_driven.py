import pandas as pd
import numpy as np
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

# Handle import gracefully for test executions
try:
    from src.utils import math_engines
except ImportError:
    pass

class TimeBar:
    @staticmethod
    def time_bar(df: pd.DataFrame, expected_bars: int = 3078) -> pd.DataFrame:
        """
        Động cơ sinh Thanh Thời gian (Time Bars) tương đương về tần suất.
        """
        # 1. Tính toán Tổng thời gian vật lý của chuỗi dữ liệu
        time_span = df.index[-1] - df.index[0]
        
        # 2. Tính Tần suất Lấy mẫu Trung bình (T) bằng Giây
        freq_seconds = int(time_span.total_seconds() / expected_bars)
        freq_str = f"{freq_seconds}s"
        
        print(f"[Time Bars] Khoảng thời gian mỗi thanh (T): {freq_seconds} giây ({freq_seconds/60:.2f} phút)")
        
        col_dollar = 'dollar_value' if 'dollar_value' in df.columns else 'dola_value'
        if col_dollar not in df.columns:
            df[col_dollar] = df['close'] * df['volume']
            
        # 3. Resample theo Thời gian thực
        time_bars = df.resample(freq_str).agg(
            open=('typical_price', 'first'),
            high=('typical_price', 'max'),
            low=('typical_price', 'min'),
            close=('typical_price', 'last'),
            volume=('volume', 'sum'),
            dollar_value=(col_dollar, 'sum')
        )
        
        # 4. Loại bỏ các khoảng thời gian "chết"
        time_bars = time_bars.dropna()
        time_bars.index.name = 'close_time'
        
        print(f"[Time Bars] Số thanh thực tế tạo ra (sau khi bỏ thanh rỗng): {len(time_bars)}")
        return time_bars


class DollarBar:
    @staticmethod
    def dynamic_dollar_bars(df: pd.DataFrame, rolling_window: int = 20, n_target: int = 20) -> pd.DataFrame:
        """
        Động cơ sinh Thanh Đô la (Dollar Bars) chuẩn MLFinLab.
        - Tối ưu hóa mảng C-contiguous (Numpy vectorization).
        - Absorb Overshoot (Hấp thụ vượt ngưỡng).
        """
        print("[Dynamic Dollar Bars] Bắt đầu khởi tạo hệ thống...")
        df = df.copy()
        
        col_dollar = 'dollar_value' if 'dollar_value' in df.columns else 'dola_value'
        if col_dollar not in df.columns:
            df[col_dollar] = df['typical_price'] * df['volume']
            
        daily_dollar_volume = df[col_dollar].resample('D').sum()
        daily_dollar_volume = daily_dollar_volume[daily_dollar_volume > 0] 
        
        # SỬA LỖI 1: Giải quyết Cold Start Problem với min_periods=1
        rolling_vt = (daily_dollar_volume.rolling(window=rolling_window, min_periods=1).mean() / n_target).shift(1)
        
        # SỬA LỖI 2: Đồng bộ hóa kiểu Date Hash
        vt_dict = {k.date(): v for k, v in rolling_vt.dropna().items()}
        print(f"[Dynamic Dollar Bars] Bảng tra cứu Vt đã có {len(vt_dict)} ngày hợp lệ.")
        
        # TỐI ƯU HÓA 5: ARRAY VECTORIZATION 
        df_dates = pd.Series(df.index.date, index=df.index)
        thresholds = df_dates.map(vt_dict).values
        
        valid_idx = ~pd.isna(thresholds)
        
        times = df.index.values[valid_idx]
        opens = df['open'].values[valid_idx]
        highs = df['high'].values[valid_idx]
        lows = df['low'].values[valid_idx]
        closes = df['close'].values[valid_idx]
        volumes = df['volume'].values[valid_idx]
        dollar_values = df[col_dollar].values[valid_idx]
        threshold_vals = thresholds[valid_idx]

        bars = []
        cum_dollar = 0.0
        cum_volume = 0.0
        cum_ticks = 0
        high_price = -np.inf
        low_price = np.inf
        open_price = None
        open_time = None

        for i in range(len(times)):
            current_datetime = times[i]
            current_vt = threshold_vals[i]
                
            curr_open = opens[i]
            curr_high = highs[i]
            curr_low = lows[i]
            curr_close = closes[i]
            volume = volumes[i]
            dollar_val = dollar_values[i]
            
            if open_price is None:
                open_price = curr_open
                open_time = current_datetime
                
            cum_dollar += dollar_val
            cum_volume += volume
            cum_ticks += 1
            
            if curr_high > high_price: high_price = curr_high
            if curr_low < low_price: low_price = curr_low
                
            if cum_dollar >= current_vt:
                bars.append({
                    'open_time': open_time,
                    'close_time': current_datetime, 
                    'open': open_price,              
                    'high': max(high_price, open_price),              
                    'low': min(low_price, open_price),                
                    'close': curr_close,             
                    'volume': cum_volume,
                    'dollar_value': cum_dollar,        
                    'tick_count': cum_ticks,
                    'vt_threshold': current_vt 
                })
                
                # TUYỆT ĐỐI HẤP THỤ PHẦN DƯ
                cum_dollar = 0.0
                cum_volume = 0.0
                cum_ticks = 0
                high_price = -np.inf
                low_price = np.inf
                open_price = None 
                open_time = None

        dollar_bars_df = pd.DataFrame(bars)
        if not dollar_bars_df.empty:
            dollar_bars_df.set_index('close_time', inplace=True)
            
        return dollar_bars_df
    
    @staticmethod
    def imbalance(df: pd.DataFrame, initial_T_guess: int = 100, span: int = 100) -> pd.DataFrame:
        """
        Động cơ sinh Imbalance Bars chuẩn MLFinLab.
        Khuyến cáo: nên để span=initial_T_guess
        Lưu ý: kiểm tra floor
        """
        def TIBs(series: pd.Series):
            print("[Imbalance Bars] Khởi tạo Quy tắc Tick...")
            delta_p = series.diff()
            b_t = np.sign(delta_p)
            b_t = b_t.replace(0, np.nan).ffill() 
            b_t = b_t.fillna(1)
            print("[Imbalance Bars] Phân loại Mua/Bán thành công.")
            return b_t

        df = df.copy()
        
        col_dollar = 'dollar_value' if 'dollar_value' in df.columns else 'dola_value'
        if col_dollar not in df.columns:
            if 'typical_price' not in df.columns:
                df['typical_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
            df[col_dollar] = df['typical_price'] * df['volume']

        # SỬA LỖI 1: Chuẩn MLFinLab. Dùng giá (Close) để xác định Tick Direction
        df['b_t'] = TIBs(df['close'])
        
        # Lọc bỏ row khuyết ban đầu
        df.dropna(subset=['b_t', col_dollar], inplace=True)
        
        # SỬA LỖI 3: Mồi dữ liệu (Warm-up Phase)
        df_warmup = df.head(initial_T_guess * 5)
        if len(df_warmup) == 0:
            raise ValueError("[Imbalance Bars] Data rỗng, không đủ mồi ngưỡng!")

        class ImbalanceThresholdEngine:
            def __init__(self, df_warmup, initial_T_guess, span=3):
                print("[RHS Engine] Warming up...")
                self.alpha = 2.0 / (span + 1)
                
                # Khởi tạo EWMA cho b_v bằng chuỗi quá khứ thay vì np.mean() cục bộ
                tick_imbalance_array = (df_warmup['b_t'] * df_warmup[col_dollar]).dropna().values
                if len(tick_imbalance_array) > 0:
                    self.expected_b_v = tick_imbalance_array[0]
                    for x in tick_imbalance_array[1:]:
                        self.expected_b_v = (self.alpha * x) + ((1 - self.alpha) * self.expected_b_v)
                else:
                    self.expected_b_v = 0.0
                
                self.expected_T = initial_T_guess
                
                # LƯU LẠI MỨC FLOOR
                self.initial_threshold = self.expected_T * np.abs(self.expected_b_v)
                self.current_threshold = self.initial_threshold
                # Sàn = x0.1 hoặc một con số fixed ***********************************************************
                self.floor = max(self.initial_threshold * 0.1, 100.0) 
                
            def update_threshold(self, actual_T, actual_tick_imbalance_array): # KHI UPDATE
                self.expected_T = (self.alpha * actual_T) + ((1 - self.alpha) * self.expected_T)
                
                # Cập nhật EWMA LUYẾN TUYẾN tick-by-tick bảo toàn decay
                for x in actual_tick_imbalance_array:
                    self.expected_b_v = (self.alpha * x) + ((1 - self.alpha) * self.expected_b_v)
                
                # ÁP DỤNG NGƯỠNG FLOOR để chống Death spiral
                raw_threshold = self.expected_T * np.abs(self.expected_b_v)
                self.current_threshold = max(raw_threshold, self.floor)
                
                return self.current_threshold

        engine = ImbalanceThresholdEngine(df_warmup, initial_T_guess=initial_T_guess, span=span)

        # TỐI ƯU HÓA 5: SỬ DỤNG ARRAY THAY VÌ ITERTUPLES 
        times = df.index.values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        dollar_values = df[col_dollar].values
        b_t_values = df['b_t'].values
        
        theta = 0.0  
        current_bar_imbalances = []  
        bars = []
        
        cum_volume = 0.0
        cum_ticks = 0
        open_price = None
        open_time = None
        high_price = -np.inf
        low_price = np.inf

        for i in range(len(df_warmup), len(times)):
            current_datetime = times[i]
            
            price = closes[i] 
            volume = volumes[i]
            dollar_val = dollar_values[i]
            b_t = b_t_values[i] 
            
            tick_imbalance = b_t * dollar_val
            theta += tick_imbalance
            current_bar_imbalances.append(tick_imbalance)
            
            if open_price is None:
                open_price = opens[i]
                open_time = current_datetime
                
            if highs[i] > high_price: high_price = highs[i]
            if lows[i] < low_price: low_price = lows[i]
                
            cum_volume += volume
            cum_ticks += 1
            
            if abs(theta) >= engine.current_threshold:
                bars.append({
                    'open_time': open_time,
                    'close_time': current_datetime,
                    'open': open_price,
                    'high': max(high_price, open_price),
                    'low': min(low_price, open_price),
                    'close': price,
                    'volume': cum_volume,
                    'dollar_imbalance': theta,
                    'tick_count': cum_ticks,
                    'threshold_used': engine.current_threshold
                })
                
                engine.update_threshold(
                    actual_T=len(current_bar_imbalances), 
                    actual_tick_imbalance_array=current_bar_imbalances
                )
                
                theta = 0.0
                current_bar_imbalances = []
                cum_volume = 0.0
                cum_ticks = 0
                open_price = None
                open_time = None
                high_price = -np.inf
                low_price = np.inf
                
        # SỬA LỖI 4: Trả về DataFrame đàng hoàng thay vì List hay Dict. Bỏ lại Orphan Bars (chuẩn MLFinlab)
        imbalance_bars_df = pd.DataFrame(bars)
        if not imbalance_bars_df.empty:
            imbalance_bars_df.set_index('close_time', inplace=True)
            
        print(f"[Imbalance Bars] Hoàn tất. Mẫu: {len(imbalance_bars_df)} thanh.")
        return imbalance_bars_df

if __name__ == '__main__':
    print("=" * 60)
    print("TESTING MLFINLAB BAR ENGINES WITH REAL DATA (FPT)")
    print("=" * 60)
    
    # 1. Load Real Data from datasets/stocks
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    fpt_path = os.path.join(project_root, 'datasets', 'stocks', 'FPT.csv')
    
    try:
        df_test = pd.read_csv(fpt_path)
        df_test['time'] = pd.to_datetime(df_test['time'])
        df_test.set_index('time', inplace=True)
        df_test.sort_index(inplace=True)
        
        # Calculate typical price and dollar value
        df_test['typical_price'] = (df_test['open'] + df_test['high'] + df_test['low'] + df_test['close']) / 4.0
        df_test['dollar_value'] = df_test['typical_price'] * df_test['volume']
        
        # For testing, let's take the last 6 months to make it run fast but meaningful
        # (adjust this timedelta if you want to test on full history)
        from datetime import timedelta
        df_test = df_test[df_test.index.date >= df_test.index.date.max() - timedelta(days=180)]
        print(f"[*] Đã tải {len(df_test)} dòng dữ liệu dòng dữ liệu FPT 1-phút (6 tháng qua).")
        
        # 2. Test Dynamic Dollar Bars
        print("\n" + "-" * 60)
        print("TEST 1: DYNAMIC DOLLAR BARS")
        print("-" * 60)
        df_bars = DollarBar.dynamic_dollar_bars(
            df_test, 
            rolling_window=20, 
            n_target=10  # Target 10 bars per day
        )
        print(f"[*] Kết quả: Tạo thành công {len(df_bars)} Dynamic Dollar Bars.")
        print(df_bars[['open', 'high', 'low', 'close', 'volume', 'dollar_value', 'tick_count']].head(5))
        print("...")
        print(df_bars[['open', 'high', 'low', 'close', 'volume', 'dollar_value', 'tick_count']].tail(3))

        # 3. Test Imbalance Bars
        print("\n" + "-" * 60)
        print("TEST 2: IMBALANCE BARS (MLFINLAB)")
        print("-" * 60)
        imb_bars = DollarBar.imbalance(
            df_test, 
            initial_T_guess=100, 
            span=100
        )
        print(f"[*] Kết quả: Tạo thành công {len(imb_bars)} Imbalance Bars.")
        print(imb_bars[['open', 'high', 'low', 'close', 'volume', 'dollar_imbalance', 'tick_count', 'threshold_used']].head(5))
        print("...")
        print(imb_bars[['open', 'high', 'low', 'close', 'volume', 'dollar_imbalance', 'tick_count', 'threshold_used']].tail(3))
        
    except FileNotFoundError:
        print(f"[!] Không tìm thấy file tại: {fpt_path}")
    except Exception as e:
        print(f"[!] Lỗi khi chạy test FPT: {e}")