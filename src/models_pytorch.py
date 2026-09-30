"""
Module: models_pytorch.py
Mục đích: Xây dựng, huấn luyện và đánh giá mô hình mạng nơ-ron hồi quy với PyTorch.
Hỗ trợ các biến thể:
- Vanilla RNN (SimpleRNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
Bao gồm:
- Custom PyTorch Dataset & DataLoader
- Vòng lặp huấn luyện tường minh (Explicit Training Loop) với AdamW và Gradient Clipping
- Tính toán các chỉ số hồi quy chuỗi thời gian: RMSE, MAE, MAPE, R2
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class TimeSeriesDataset(Dataset):
    """Lớp Dataset đóng gói tensor chuỗi thời gian cho PyTorch DataLoader"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PyTorchRNNModel(nn.Module):
    """
    Kiến trúc mạng hồi quy tổng quát hỗ trợ RNN, LSTM và GRU.
    Cấu trúc:
        Input: (batch_size, seq_len, input_dim)
        -> Recurrent Backbone (RNN / LSTM / GRU, multi-layer, dropout)
        -> Trích xuất hidden state tại bước thời gian cuối cùng h_T
        -> Fully Connected Layer (Linear Projection)
        -> Output: (batch_size, 1)
    """
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2, rnn_type="LSTM"):
        super(PyTorchRNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.upper()
        
        # Chọn backbone hồi quy tương ứng
        drop_rate = dropout if num_layers > 1 else 0.0
        if self.rnn_type == "RNN":
            self.recurrent = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity='tanh',
                dropout=drop_rate
            )
        elif self.rnn_type == "LSTM":
            self.recurrent = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=drop_rate
            )
        elif self.rnn_type == "GRU":
            self.recurrent = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=drop_rate
            )
        else:
            raise ValueError(f"Kiến trúc không được hỗ trợ: {rnn_type}")
            
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        # x có kích thước: (batch_size, seq_len, input_dim)
        if self.rnn_type == "LSTM":
            out, (h_n, c_n) = self.recurrent(x)
        else:
            out, h_n = self.recurrent(x)
            
        # Lấy hidden state của bước thời gian cuối cùng: out[:, -1, :]
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        prediction = self.fc(last_hidden)
        return prediction

def train_pytorch_model(model, train_loader, val_loader, epochs=25, lr=0.001, weight_decay=1e-4, 
                        device="cpu", save_path="models/pytorch_model.pt", patience=7):
    """
    Vòng lặp huấn luyện tường minh cho PyTorch:
    - Sử dụng optimizer AdamW
    - Hàm mất mát HuberLoss (giảm độ nhạy với ngoại lai so với MSE)
    - Gradient Clipping chống bùng nổ đạo hàm
    - Early Stopping lưu giữ checkpoint tối ưu nhất
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.to(device)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "epoch_times": []
    }
    
    best_val_loss = float("inf")
    patience_counter = 0
    start_train_time = time.time()
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            
            # Cắt tỉa đạo hàm chống bùng nổ
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item() * len(batch_y)
            
        train_loss = total_train_loss / len(train_loader.dataset)
        
        # Đánh giá trên tập Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                total_val_loss += loss.item() * len(batch_y)
                
        val_loss = total_val_loss / len(val_loader.dataset)
        epoch_time = time.time() - t0
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_times"].append(epoch_time)
        
        scheduler.step(val_loss)
        
        # Kiểm tra lưu checkpoint tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Early stopping
                break
                
    total_duration = time.time() - start_train_time
    history["total_duration"] = total_duration
    
    # Nạp lại trọng số tốt nhất
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        
    return model, history

def evaluate_pytorch_model(model, test_loader, pipeline, device="cpu"):
    """Đánh giá mô hình trên tập kiểm thử và khôi phục giá trị thang đo thực tế"""
    model.eval()
    y_preds_list, y_true_list = [], []
    inference_times = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            t0 = time.time()
            preds = model(batch_x)
            inference_times.append(time.time() - t0)
            
            y_preds_list.extend(preds.cpu().numpy().flatten())
            y_true_list.extend(batch_y.numpy().flatten())
            
    y_preds_scaled = np.array(y_preds_list)
    y_true_scaled = np.array(y_true_list)
    
    # Khôi phục về thang đo ban đầu
    y_preds = pipeline.inverse_transform_target(y_preds_scaled)
    y_true = pipeline.inverse_transform_target(y_true_scaled)
    
    # Tính toán các chỉ số đo lường hiệu năng
    rmse = np.sqrt(np.mean((y_preds - y_true) ** 2))
    mae = np.mean(np.abs(y_preds - y_true))
    mape = np.mean(np.abs((y_true - y_preds) / np.where(y_true == 0, 1e-6, y_true))) * 100.0
    ss_res = np.sum((y_true - y_preds) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    avg_latency_ms = (np.mean(inference_times) / len(test_loader.dataset)) * 1000.0
    
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "avg_latency_ms": float(avg_latency_ms)
    }
    
    return metrics, y_preds, y_true

if __name__ == "__main__":
    from data_preprocessing import TimeSeriesDataPipeline
    print("=== KIỂM THỬ XÂY DỰNG MÔ HÌNH PYTORCH ===")
    pipe = TimeSeriesDataPipeline(dataset_name="stock", window_size=30)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), info = pipe.temporal_split()
    
    train_loader = DataLoader(TimeSeriesDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_va, y_va), batch_size=64, shuffle=False)
    test_loader = DataLoader(TimeSeriesDataset(X_te, y_te), batch_size=64, shuffle=False)
    
    model = PyTorchRNNModel(input_dim=5, hidden_dim=32, num_layers=2, rnn_type="LSTM")
    print(model)
    model, hist = train_pytorch_model(model, train_loader, val_loader, epochs=5, save_path="models/test_stock_lstm.pt")
    metrics, preds, trues = evaluate_pytorch_model(model, test_loader, pipe)
    print("Kết quả đánh giá trên tập kiểm thử:")
    print(metrics)
