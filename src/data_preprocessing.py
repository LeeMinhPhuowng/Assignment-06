"""
Module: data_preprocessing.py
Mục đích: Tiền xử lý, làm sạch, phân tích thống kê và tạo cửa sổ trượt (Sliding Window)
cho 2 bộ dữ liệu:
1. Amazon Stock Price (Chứng khoán: data/amazon_stock_price.csv)
2. Gold Price (Giao dịch giá vàng theo thời gian: data/gold_price.csv)

Tuân thủ nguyên tắc chống rò rỉ dữ liệu (No Data Leakage):
- Phân chia Train (70%) / Validation (15%) / Test (15%) theo đúng trật tự thời gian.
- Chuẩn hóa Min-Max chỉ fit trên tập Train, áp dụng thụ động cho Val và Test.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class TimeSeriesDataPipeline:
    def __init__(self, dataset_name="stock", window_size=30, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
        self.dataset_name = dataset_name.lower()
        self.window_size = window_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        self.scaler_min = None
        self.scaler_max = None
        self.feature_names = []
        self.target_name = ""
        self.df = None
        
    def load_data(self, data_path=None):
        """Nạp dữ liệu và chuẩn hóa đặc trưng chuỗi thời gian"""
        if self.dataset_name == "stock":
            path = data_path or "data/amazon_stock_price.csv"
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            self.target_name = 'Close'
            self.feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
            
        elif self.dataset_name == "gold":
            path = data_path or "data/gold_price.csv"
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            # Loại bỏ các dòng thiếu giá (ở giai đoạn đầu 1968) và sắp xếp thời gian
            df = df.dropna(subset=['price']).sort_values('date').reset_index(drop=True)
            
            # Tạo các đặc trưng bổ trợ từ chuỗi thời gian giá vàng
            df['MA_7'] = df['price'].rolling(window=7, min_periods=1).mean()
            df['MA_30'] = df['price'].rolling(window=30, min_periods=1).mean()
            df['Return'] = df['price'].pct_change().fillna(0.0)
            df['Volatility_7'] = df['Return'].rolling(window=7, min_periods=1).std().fillna(0.0)
            
            self.target_name = 'price'
            self.feature_names = ['price', 'MA_7', 'MA_30', 'Return', 'Volatility_7']
            
        else:
            raise ValueError(f"Tập dữ liệu không hợp lệ: {self.dataset_name}. Chỉ hỗ trợ 'stock' hoặc 'gold'.")
            
        self.df = df
        return self.df
    
    def get_descriptive_stats(self):
        """Tính toán thống kê mô tả chuyên sâu"""
        if self.df is None:
            self.load_data()
        numeric_df = self.df[self.feature_names]
        stats = numeric_df.describe().T
        stats['skew'] = numeric_df.skew()
        stats['kurtosis'] = numeric_df.kurtosis()
        stats['missing'] = numeric_df.isnull().sum()
        return stats
    
    def temporal_split(self):
        """Phân tách dữ liệu theo thời gian nghiêm ngặt chống rò rỉ dữ liệu"""
        if self.df is None:
            self.load_data()
            
        values = self.df[self.feature_names].values.astype(np.float32)
        target_idx = self.feature_names.index(self.target_name)
        
        n_total = len(values)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        train_raw = values[:n_train]
        val_raw = values[n_train:n_train + n_val]
        test_raw = values[n_train + n_val:]
        
        # Fit Min-Max scaler CHỈ trên tập Train
        self.scaler_min = train_raw.min(axis=0)
        self.scaler_max = train_raw.max(axis=0)
        scale_denom = np.where(self.scaler_max - self.scaler_min == 0, 1.0, self.scaler_max - self.scaler_min)
        
        train_scaled = (train_raw - self.scaler_min) / scale_denom
        val_scaled = (val_raw - self.scaler_min) / scale_denom
        test_scaled = (test_raw - self.scaler_min) / scale_denom
        
        # Tạo chuỗi sliding windows
        X_train, y_train = self._create_sliding_windows(train_scaled, target_idx)
        
        val_context = np.vstack([train_scaled[-self.window_size:], val_scaled])
        X_val, y_val = self._create_sliding_windows(val_context, target_idx)
        
        test_context = np.vstack([val_scaled[-self.window_size:], test_scaled])
        X_test, y_test = self._create_sliding_windows(test_context, target_idx)
        
        split_info = {
            "n_total": n_total,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "num_features": len(self.feature_names),
            "window_size": self.window_size,
            "target_idx": target_idx
        }
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), split_info
    
    def _create_sliding_windows(self, series_data, target_idx):
        X_list, y_list = [], []
        n_samples = len(series_data) - self.window_size
        for i in range(n_samples):
            X_list.append(series_data[i : i + self.window_size])
            y_list.append(series_data[i + self.window_size, target_idx])
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
    
    def inverse_transform_target(self, y_scaled):
        target_idx = self.feature_names.index(self.target_name)
        min_val = self.scaler_min[target_idx]
        max_val = self.scaler_max[target_idx]
        return y_scaled * (max_val - min_val) + min_val

    def save_scaler(self, json_path):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        target_idx = self.feature_names.index(self.target_name)
        data = {
            "dataset": self.dataset_name,
            "features": self.feature_names,
            "target": self.target_name,
            "target_idx": target_idx,
            "min": self.scaler_min.tolist(),
            "max": self.scaler_max.tolist()
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    print("=== KIỂM THỬ TIỀN XỬ LÝ CHO AMAZON STOCK VÀ GOLD PRICE ===")
    
    # 1. Amazon Stock
    s_pipe = TimeSeriesDataPipeline(dataset_name="stock", window_size=30)
    s_pipe.load_data()
    print("\n[1] Thống kê mô tả Amazon Stock:")
    print(s_pipe.get_descriptive_stats()[['mean', 'std', 'min', '50%', 'max', 'skew']])
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), info = s_pipe.temporal_split()
    print("Thông tin split Stock:", info)
    
    # 2. Gold Price
    g_pipe = TimeSeriesDataPipeline(dataset_name="gold", window_size=30)
    g_pipe.load_data()
    print("\n[2] Thống kê mô tả Gold Price:")
    print(g_pipe.get_descriptive_stats()[['mean', 'std', 'min', '50%', 'max', 'skew']])
    (X_tr_g, y_tr_g), (X_va_g, y_va_g), (X_te_g, y_te_g), info_g = g_pipe.temporal_split()
    print("Thông tin split Gold:", info_g)
