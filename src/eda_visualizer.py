"""
Module: eda_visualizer.py
Mục đích: Trực quan hóa dữ liệu khám phá (EDA) cho 2 bộ dữ liệu:
1. Amazon Stock Price (Chứng khoán)
2. Gold Price (Giá vàng theo thời gian)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

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

def visualize_datasets(output_dir="notebooks"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Đọc dữ liệu
    df_stock = pd.read_csv("data/amazon_stock_price.csv")
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    df_stock = df_stock.sort_values('Date').reset_index(drop=True)
    
    df_gold = pd.read_csv("data/gold_price.csv")
    df_gold['date'] = pd.to_datetime(df_gold['date'])
    df_gold = df_gold.dropna(subset=['price']).sort_values('date').reset_index(drop=True)
    
    # 2. Vẽ tổng quan 2 chuỗi thời gian
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    ax1.plot(df_stock['Date'], df_stock['Close'], color='#000000', linewidth=1.2, label='Giá đóng cửa Amazon (USD)')
    ax1.set_ylabel('Giá cổ phiếu (USD)', fontsize=10, fontweight='bold')
    ax1.set_title('Chuỗi thời gian giá cổ phiếu Amazon (1997 - 2023)', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5, color='#888888')
    ax1.legend(loc='upper left', frameon=True, edgecolor='#000000')
    
    ax2.plot(df_gold['date'], df_gold['price'], color='#333333', linewidth=1.2, label='Giá vàng quốc tế (USD/Ounce)')
    ax2.set_ylabel('Giá vàng (USD/Ounce)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Năm giao dịch', fontsize=10, fontweight='bold')
    ax2.set_title('Chuỗi thời gian giá vàng thế giới (1968 - 2021)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5, color='#888888')
    ax2.legend(loc='upper left', frameon=True, edgecolor='#000000')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_datasets()
