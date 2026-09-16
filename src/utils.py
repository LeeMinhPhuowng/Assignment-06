"""
Module Visualization and Metrics Utilities
Được thiết kế chuẩn mực học thuật:
- Font chữ: Times New Roman
- Nền trắng 100% (#FFFFFF), văn bản và trục đen tương phản cao
- Độ phân giải: 300 DPI
- Không màu mè, nhãn rõ ràng
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Cấu hình thẩm mỹ học thuật
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
plt.rcParams['figure.facecolor'] = '#FFFFFF'
plt.rcParams['axes.facecolor'] = '#FFFFFF'

def evaluate_metrics(y_true, y_pred):
    """
    Tính toán 4 chỉ số phân loại đa lớp chuẩn:
    Accuracy, Macro-Precision, Macro-Recall, Macro-F1.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {
        'accuracy': acc,
        'macro_precision': prec,
        'macro_recall': rec,
        'macro_f1': f1
    }

def plot_dataset_samples(images, labels, class_names, filename='report_images/dataset_samples.png', title='Sample Images'):
    """
    Vẽ lưới ảnh mẫu của 10 lớp trong bộ dữ liệu.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5), facecolor='#FFFFFF')
    fig.suptitle(title, fontsize=12, fontweight='bold', color='#000000', y=1.02)
    
    unique_labels = sorted(list(set(labels)))
    for idx, ax in enumerate(axes.flat):
        if idx < len(unique_labels):
            target_class = unique_labels[idx]
            match_indices = np.where(labels == target_class)[0]
            img = images[match_indices[0]]
            
            # Nếu là ảnh Grayscale (1, H, W) hoặc (H, W)
            if img.ndim == 3 and img.shape[0] == 1:
                ax.imshow(img.squeeze(0), cmap='gray')
            elif img.ndim == 3 and img.shape[0] == 3: # (3, H, W)
                ax.imshow(np.transpose(img, (1, 2, 0)))
            elif img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
                
            ax.set_title(f"{class_names[target_class]}", fontsize=10, color='#000000')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Đã lưu ảnh mẫu: {filename}")

def plot_learning_curves(history, filename='report_images/learning_curves.png', title='Learning Curves'):
    """
    Vẽ biểu đồ lịch sử mất mát và độ chính xác qua các epoch.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor='#FFFFFF')
    
    # 1. Loss Curve
    ax1.plot(epochs, history['train_loss'], 'k-', label='Train Loss', linewidth=1.5)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax1.plot(epochs, history['val_loss'], 'k--', label='Val Loss', linewidth=1.5)
    ax1.set_title('Cross-Entropy Loss vs. Epochs', fontsize=11, fontweight='bold', color='#000000')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6, color='#888888')
    ax1.legend(loc='upper right', frameon=True, edgecolor='#000000')

    # 2. Accuracy Curve
    if 'val_acc' in history and len(history['val_acc']) > 0:
        ax2.plot(epochs, [a * 100 if a <= 1.0 else a for a in history.get('train_acc', history['val_acc'])], 
                 'k-', label='Train Acc', linewidth=1.5)
        ax2.plot(epochs, [a * 100 if a <= 1.0 else a for a in history['val_acc']], 
                 'k--', label='Val Acc', linewidth=1.5)
        ax2.set_title('Accuracy (%) vs. Epochs', fontsize=11, fontweight='bold', color='#000000')
        ax2.set_xlabel('Epoch', fontsize=10)
        ax2.set_ylabel('Accuracy (%)', fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.6, color='#888888')
        ax2.legend(loc='lower right', frameon=True, edgecolor='#000000')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Đã lưu biểu đồ Learning Curves: {filename}")

def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, filename='report_images/confusion_matrix.png', title='Normalized Confusion Matrix'):
    """
    Vẽ ma trận nhầm lẫn chuẩn hóa (Normalized Confusion Matrix) theo từng lớp.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor='#FFFFFF')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Greys', cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5, linecolor='#000000')
    
    ax.set_title(title, fontsize=11, fontweight='bold', color='#000000', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold', color='#000000')
    ax.set_ylabel('True Label', fontsize=10, fontweight='bold', color='#000000')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Đã lưu Confusion Matrix: {filename}")

def plot_high_confidence_errors(images, y_true, y_pred, confidences, class_names, filename='report_images/high_confidence_errors.png', num_samples=6):
    """
    Trích xuất và vẽ các mẫu dự đoán SAI nhưng có CONFIDENCE CAO NHẤT (Error Analysis mục 44).
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    wrong_mask = (y_true != y_pred)
    wrong_indices = np.where(wrong_mask)[0]
    
    if len(wrong_indices) == 0:
        print("Không có mẫu nào đoán sai để trực quan hóa lỗi!")
        return

    # Lấy confidence của các mẫu sai và sắp xếp giảm dần
    wrong_confs = confidences[wrong_indices]
    sorted_order = np.argsort(wrong_confs)[::-1]
    top_wrong_idx = wrong_indices[sorted_order[:num_samples]]
    
    cols = min(num_samples, 6)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 2.2, 2.6), facecolor='#FFFFFF')
    if cols == 1:
        axes = [axes]
        
    for i, idx in enumerate(top_wrong_idx):
        ax = axes[i]
        img = images[idx]
        if img.ndim == 3 and img.shape[0] == 1:
            ax.imshow(img.squeeze(0), cmap='gray')
        elif img.ndim == 3 and img.shape[0] == 3:
            ax.imshow(np.transpose(img, (1, 2, 0)))
        elif img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
            
        true_name = class_names[y_true[idx]]
        pred_name = class_names[y_pred[idx]]
        conf = confidences[idx] * 100
        
        ax.set_title(f"True: {true_name}\nPred: {pred_name}\nConf: {conf:.1f}%", 
                     fontsize=9, color='#000000')
        ax.axis('off')

    plt.suptitle("High-Confidence Error Predictions Analysis", fontsize=11, fontweight='bold', color='#000000', y=1.05)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Đã lưu ảnh phân tích lỗi: {filename}")
