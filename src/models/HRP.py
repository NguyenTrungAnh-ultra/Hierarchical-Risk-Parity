import os
import sys

# Thêm thư mục 'src' vào sys.path để chạy trực tiếp
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

from core.config import project_root, stocks_data_path
from services.data_loader import load_stocks
from core.math_engines import log_return

class HRP:
    def __init__(self):
        pass

    def get_distance_matrix(self, corr):
        # 1. Khoảng cách cơ bản dựa trên tương quan
        dist_corr = np.sqrt(0.5 * (1 - corr))
        
        # 2. Khoảng cách của khoảng cách (Distance of Distance)
        dist_of_dist = np.zeros(corr.shape)
        n_assets = dist_corr.shape[0] 
        
        for i in range(n_assets):
            for j in range(n_assets):
                dist_of_dist[i, j] = np.sqrt(np.sum((dist_corr.iloc[:, i] - dist_corr.iloc[:, j])**2))
                
        return pd.DataFrame(dist_of_dist, index=corr.index, columns=corr.columns)

    def get_linkage(self, dist_of_dist):
        from scipy.spatial.distance import squareform
        dist_values = dist_of_dist.values
        # Đảm bảo ma trận đối xứng và đường chéo bằng 0 cho squareform
        dist_values = (dist_values + dist_values.T) / 2
        np.fill_diagonal(dist_values, 0)
        dist_array = squareform(dist_values)
        
        # Sử dụng 'single' linkage theo thuật toán bài báo
        link = sch.linkage(dist_array, method='single')
        return link

    def get_quasi_diag(self, link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3] 
        
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = (df0.values - num_items).astype(int)
            sort_ix.loc[i] = link[j, 0] 
            df0_new = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0_new]) 
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
            
        return sort_ix.tolist()

    def get_cluster_var(self, cov, c_items):
        cov_slice = cov.iloc[c_items, c_items]
        
        # Tính trọng số nghịch đảo phương sai (Inverse-Variance) nội bộ
        # Cộng thêm 1e-8 để tránh lỗi chia cho 0 với tài sản không có variance
        ivp = 1. / (np.diag(cov_slice) + 1e-8)
        ivp /= ivp.sum()
        w = ivp.reshape(-1, 1)
        
        c_var = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
        return c_var

    def get_rec_bipart(self, cov, sort_ix):
        w = pd.Series(1.0, index=sort_ix) 
        c_items = [sort_ix] 
        
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            
            for i in range(0, len(c_items), 2): 
                c_items0 = c_items[i]   
                c_items1 = c_items[i+1] 
                
                c_var0 = self.get_cluster_var(cov, c_items0)
                c_var1 = self.get_cluster_var(cov, c_items1)
                
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                
                w.loc[c_items0] *= alpha
                w.loc[c_items1] *= 1 - alpha
                
        return w
        
    def plot_visualizations(self, corr, link, sort_ix):
        # --- Cửa sổ 1: Sơ đồ cây (Dendrogram) ---
        plt.figure(figsize=(10, 6))
        plt.title("Tree Clustering Dendrogram")
        sch.dendrogram(link, labels=corr.index.tolist(), leaf_rotation=90)
        min_merge_dist = link[0, 2]
        plt.ylim(bottom=max(0, min_merge_dist - 0.02))
        plt.xlabel("Tài sản (Assets)")
        plt.ylabel("Khoảng cách")
        plt.tight_layout()
        
        # --- Cửa sổ 2: So sánh Heatmap Correlation (Trước và Sau Quasi-Diagonalization) ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 1. Heatmap: Trước khi sắp xếp (Gốc)
        im1 = axes[0].imshow(corr.values, cmap='coolwarm', vmin=-0.5, vmax=0.5, aspect='auto')
        axes[0].set_title("Correlation Heatmap: Gốc (Trước Quasi-Diag)")
        axes[0].set_xticks(np.arange(len(corr.columns)))
        axes[0].set_yticks(np.arange(len(corr.index)))
        axes[0].set_xticklabels(corr.columns, rotation=90, fontsize=9)
        axes[0].set_yticklabels(corr.index, fontsize=9)
        fig.colorbar(im1, ax=axes[0])
        
        # 2. Heatmap: Sau khi sắp xếp (Quasi-Diagonalization)
        corr_sorted = corr.iloc[sort_ix, sort_ix]
        im2 = axes[1].imshow(corr_sorted.values, cmap='coolwarm', vmin=-0.5, vmax=0.5, aspect='auto')
        axes[1].set_title("Correlation Heatmap: Sau khi sắp xếp (Quasi-Diag)")
        axes[1].set_xticks(np.arange(len(corr_sorted.columns)))
        axes[1].set_yticks(np.arange(len(corr_sorted.index)))
        axes[1].set_xticklabels(corr_sorted.columns, rotation=90, fontsize=9)
        axes[1].set_yticklabels(corr_sorted.index, fontsize=9)
        fig.colorbar(im2, ax=axes[1])

        plt.tight_layout()

    def allocate(self, returns_df, visualize=False):
        cov = returns_df.cov()
        corr = returns_df.corr()
        
        # 1. Tree Clustering
        dist_matrix = self.get_distance_matrix(corr)
        link = self.get_linkage(dist_matrix)
            
        # 2. Quasi-Diagonalization
        sort_ix = self.get_quasi_diag(link)
        
        if visualize:
            self.plot_visualizations(corr, link, sort_ix)
        
        # 3. Recursive Bisection
        weights = self.get_rec_bipart(cov, sort_ix)
        
        # Ánh xạ index kết quả về index chữ (tên các assets ban đầu)
        weights.index = returns_df.columns[weights.index] 
        weights = weights.loc[returns_df.columns]
        
        return weights

if __name__ == "__main__":
    print("Đang tải dữ liệu cổ phiếu...")
    dict_all_stocks = load_stocks(start_date='2025-01-01')
    print(f"Đã tải {len(dict_all_stocks)} mã cổ phiếu.")
    
    # Chọn ra max 15 mã để test (có thể chọn toàn bộ nều tài nguyên cho phép)
    tickers_to_test = list(dict_all_stocks.keys())[:50]
    print(f"\nSử dụng {len(tickers_to_test)} mã để test HRP: {tickers_to_test}")
    
    prices_dict = {}
    for ticker in tickers_to_test:
        df = dict_all_stocks[ticker]
        if 'close' in df.columns and 'time' in df.columns:
            # Gán múi thời gian làm index để dễ dàng lọc ngày tháng
            prices_dict[ticker] = df.set_index('time')['close']
            
    # Xóa dòng có NaN để bảo đảm chuỗi dữ liệu khớp nhau
    prices_df = pd.DataFrame(prices_dict).dropna()
    
    # Chuyển đổi index sang dạng chuẩn Datetime
    prices_df.index = pd.to_datetime(prices_df.index)
    
    # LỌC DỮ LIỆU: Chỉ lấy dữ liệu từ ngày 1 tháng 1 năm 2024 trở đi
    prices_df = prices_df[prices_df.index >= '2025-01-01']
    
    # Lọc bỏ các dòng có giá trị <= 0 để tránh lỗi chia cho 0 hoặc log của số âm
    prices_df = prices_df[(prices_df > 0).all(axis=1)]
    
    print("\nTính Log Returns với hàm từ core.math_engines...")
    returns_dict = {}
    for col in prices_df.columns:
        returns_dict[col] = log_return(prices_df[col].values)
    
    returns_df = pd.DataFrame(returns_dict)
    
    # Loại bỏ inf và NaN nếu có trong quá trình log return
    returns_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    returns_df.dropna(inplace=True)
    
    print("\nKhởi tạo mô hình HRP và phân bổ danh mục (có bật visualize=True)...")
    model = HRP()
    weights = model.allocate(returns_df, visualize=True)
    
    print("\n>>> KẾT QUẢ TRỌNG SỐ PHÂN BỔ HRP <<<")
    print(weights.to_frame(name="Weight"))
    print("\nTổng trọng số (Cần xấp xỉ 1.0):", weights.sum())
    
    # Hiện biểu đồ thụt lùi xuống cuối cùng để in text ra xong mới vẽ
    plt.show()