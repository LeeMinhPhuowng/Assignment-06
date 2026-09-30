"""
Module: utils.py
Mục đích: Cung cấp các hàm tiện ích đo lường chỉ số đánh giá chuỗi thời gian
và hỗ trợ trực quan hóa chuẩn mực học thuật (Times New Roman, #000000, 300 DPI).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Cấu hình đồ họa chuẩn mực học thuật
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['figure.facecolor'] = '#FFFFFF'
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['savefig.facecolor'] = '#FFFFFF'

def calculate_time_series_metrics(y_true, y_pred):
    """
    Tính toán 5 chỉ số đánh giá chất lượng dự báo chuỗi thời gian:
    - RMSE: Sai số bình phương trung bình dạng căn
    - MAE: Sai số tuyệt đối trung bình
    - MAPE: Phần trăm sai số tuyệt đối trung bình (%)
    - R2: Hệ số xác định
    - Max Error: Sai số cực đại
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    denom = np.where(y_true == 0, 1e-6, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    max_err = np.max(np.abs(y_true - y_pred))
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "max_error": float(max_err)
    }

def plot_forecast_comparison(y_true, y_pred, title, filename, xlabel="Bước thời gian (Timesteps)", ylabel="Giá trị thực tế"):
    """Vẽ biểu đồ đối sánh giữa giá trị thực tế và giá trị dự báo"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    
    ax.plot(y_true, color='#000000', linewidth=1.5, label='Thực tế (Ground Truth)')
    ax.plot(y_pred, color='#555555', linestyle='--', linewidth=1.3, label='Mô hình dự báo')
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5, color='#888888')
    ax.legend(loc='upper left', frameon=True, edgecolor='#000000')
    
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Đã lưu biểu đồ: {filename}")

def plot_training_loss(train_loss, val_loss, title, filename):
    """Vẽ đường cong mất mát qua các kỷ nguyên huấn luyện"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    ax.plot(train_loss, color='#000000', linestyle='-', linewidth=1.5, label='Mất mát huấn luyện (Train Loss)')
    ax.plot(val_loss, color='#666666', linestyle='--', linewidth=1.5, label='Mất mát thẩm định (Val Loss)')
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel('Kỷ nguyên (Epoch)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Hàm mất mát Huber Loss', fontsize=10, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5, color='#888888')
    ax.legend(frameon=True, edgecolor='#000000')
    
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Đã lưu biểu đồ: {filename}")
