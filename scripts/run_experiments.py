"""
Script Chạy Thực Nghiệm Gọn Gàng, Trực Diện và Chuẩn Mực
Tập trung vào các nội dung cốt lõi:
1. Nạp 3 bộ dữ liệu (MNIST, Fashion-MNIST, Digits)
2. Kiểm tra shape tensor và tính toán tham số (Inspecting Shapes & Parameter Counting)
3. Huấn luyện 3 mô hình (NumPy Scratch, Keras, PyTorch)
4. Sinh 4 biểu đồ cần thiết nhất (Ảnh mẫu, Learning Curves, Confusion Matrix, Error Analysis)
5. Xuất số liệu đối chuẩn học thuật
"""

import os
import sys
import time
import json

# Hỗ trợ hiển thị tiếng Việt trên console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Đảm bảo đường dẫn gốc được nhận diện
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from src.data_loader import load_all_three_datasets
from src.cnn_scratch import CNN_Scratch
from src.models_keras import build_keras_cnn
from src.models_pytorch import PyTorchCNN
from src.utils import evaluate_metrics

np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

def main():
    print("====================================================================")
    print("THỰC THI THỰC NGHIỆM CNN: 3 DATASET & 3 MỨC ĐỘ CÀI ĐẶT")
    print("====================================================================")

    os.makedirs('report_images', exist_ok=True)
    results = {}

    # ------------------------------------------------------------------
    # 1. NẠP 3 BỘ DỮ LIỆU
    # ------------------------------------------------------------------
    print("\n>>> [1/5] Nạp 3 bộ dữ liệu (2 ảnh + 1 vector bảng)...")
    datasets = load_all_three_datasets('./data', batch_size=64)
    print("  - Dataset 1 (Ảnh 1): MNIST (60,000 train, 10,000 test - Kích thước 28x28x1)")
    print("  - Dataset 2 (Ảnh 2): Fashion-MNIST (60,000 train, 10,000 test - Kích thước 28x28x1)")
    print("  - Dataset 3 (Bảng/Vector): UCI Digits (1,797 mẫu - Kích thước 64 đặc trưng)")

    # Vẽ Biểu đồ 1: Ảnh mẫu trực quan của các bộ dữ liệu ảnh
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5), facecolor='#FFFFFF')
    raw_m_train, _ = datasets['raw_mnist']
    raw_f_train, _ = datasets['raw_fmnist']
    
    fmnist_labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    for i in range(5):
        # Hàng 1: MNIST
        axes[0, i].imshow(raw_m_train.data[i].numpy(), cmap='gray')
        axes[0, i].set_title(f"Digit: {raw_m_train.targets[i].item()}", fontsize=10, color='#000000')
        axes[0, i].axis('off')
        # Hàng 2: Fashion-MNIST
        axes[1, i].imshow(raw_f_train.data[i].numpy(), cmap='gray')
        axes[1, i].set_title(f"{fmnist_labels[raw_f_train.targets[i].item()]}", fontsize=10, color='#000000')
        axes[1, i].axis('off')

    plt.suptitle("Hình 1: Ảnh mẫu đại diện của MNIST (Hàng trên) và Fashion-MNIST (Hàng dưới)", 
                 fontsize=11, fontweight='bold', color='#000000', y=0.98)
    plt.tight_layout()
    plt.savefig('report_images/fig1_dataset_samples.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("  [OK] Đã lưu Hình 1: report_images/fig1_dataset_samples.png")

    # ------------------------------------------------------------------
    # 2. KHÁM PHÁ KIẾN TRÚC: INSPECTING SHAPES & PARAMETER COUNTING
    # ------------------------------------------------------------------
    print("\n>>> [2/5] Khám phá mô hình CNN (Shape Inspection & Parameter Counting)...")
    pt_model = PyTorchCNN(in_channels=1, num_classes=10, spatial_size=28)
    dummy_input = torch.randn(2, 1, 28, 28)
    shapes = pt_model.inspect_shapes(dummy_input)
    print("  Duyệt shape trung gian qua từng khối:")
    for name, s in shapes:
        print(f"    * {name}: {s}")

    details, total_params = pt_model.count_parameters_detailed()
    print(f"  Tổng số tham số có thể huấn luyện: {total_params:,}")

    # ------------------------------------------------------------------
    # 3. HUẤN LUYỆN CNN FROM SCRATCH (NUMPY THUẦN)
    # ------------------------------------------------------------------
    print("\n>>> [3/5] Huấn luyện CNN from Scratch bằng NumPy thuần...")
    # Lấy tập con stratified
    raw_m_test = datasets['raw_mnist'][1]
    X_train_np = (raw_m_train.data.numpy()[:2500, None, :, :].astype(np.float32) / 255.0 - 0.1307) / 0.3081
    y_train_np = raw_m_train.targets.numpy()[:2500]
    X_val_np   = (raw_m_train.data.numpy()[2500:3000, None, :, :].astype(np.float32) / 255.0 - 0.1307) / 0.3081
    y_val_np   = raw_m_train.targets.numpy()[2500:3000]
    X_test_np  = (raw_m_test.data.numpy()[:1000, None, :, :].astype(np.float32) / 255.0 - 0.1307) / 0.3081
    y_test_np  = raw_m_test.targets.numpy()[:1000]

    model_scratch = CNN_Scratch(in_channels=1, num_classes=10)
    scratch_hist, time_scratch = model_scratch.fit(
        X_train_np, y_train_np, X_val_np, y_val_np, epochs=5, batch_size=64, lr=2e-3
    )
    preds_scratch = model_scratch.predict(X_test_np)
    metrics_scratch = evaluate_metrics(y_test_np, preds_scratch)
    metrics_scratch['latency_sec'] = time_scratch
    metrics_scratch['params'] = model_scratch.count_parameters()
    results['NumPy_Scratch'] = metrics_scratch
    print(f"  [OK] NumPy CNN Accuracy: {metrics_scratch['accuracy']*100:.2f}% (Thời gian: {time_scratch:.2f}s)")

    # ------------------------------------------------------------------
    # 4. HUẤN LUYỆN CNN BẰNG KERAS / TENSORFLOW
    # ------------------------------------------------------------------
    print("\n>>> [4/5] Huấn luyện CNN bằng Keras / TensorFlow...")
    X_k_train = (raw_m_train.data.numpy()[:30000, :, :, None].astype(np.float32) / 255.0 - 0.1307) / 0.3081
    y_k_train = raw_m_train.targets.numpy()[:30000]
    X_k_val   = (raw_m_train.data.numpy()[30000:35000, :, :, None].astype(np.float32) / 255.0 - 0.1307) / 0.3081
    y_k_val   = raw_m_train.targets.numpy()[30000:35000]
    X_k_test  = (raw_m_test.data.numpy().astype(np.float32) / 255.0 - 0.1307) / 0.3081
    y_k_test  = raw_m_test.targets.numpy()

    keras_model = build_keras_cnn(input_shape=(28, 28, 1), num_classes=10)
    keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    t0_k = time.perf_counter()
    k_fit = keras_model.fit(X_k_train, y_k_train, validation_data=(X_k_val, y_k_val),
                            epochs=5, batch_size=128, verbose=0)
    time_k = time.perf_counter() - t0_k

    preds_k = np.argmax(keras_model.predict(X_k_test, verbose=0), axis=1)
    metrics_k = evaluate_metrics(y_k_test, preds_k)
    metrics_k['latency_sec'] = time_k
    metrics_k['params'] = keras_model.count_params()
    results['Keras_TensorFlow'] = metrics_k
    print(f"  [OK] Keras CNN Accuracy: {metrics_k['accuracy']*100:.2f}% (Thời gian: {time_k:.2f}s)")

    # ------------------------------------------------------------------
    # 5. HUẤN LUYỆN CNN BẰNG PYTORCH
    # ------------------------------------------------------------------
    print("\n>>> [5/5] Huấn luyện CNN bằng PyTorch...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pt_net = PyTorchCNN(in_channels=1, num_classes=10, spatial_size=28).to(device)
    optimizer = optim.Adam(pt_net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    pt_train_loss = []
    pt_val_loss = []
    t0_pt = time.perf_counter()

    for epoch in range(5):
        pt_net.train()
        running_l = 0.0
        tot = 0
        for imgs, lbls in datasets['mnist']['train']:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outs = pt_net(imgs)
            l = criterion(outs, lbls)
            l.backward()
            optimizer.step()
            running_l += l.item() * imgs.size(0)
            tot += imgs.size(0)
        pt_train_loss.append(running_l / tot)

        pt_net.eval()
        v_l, v_tot = 0.0, 0
        with torch.no_grad():
            for imgs, lbls in datasets['mnist']['val']:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outs = pt_net(imgs)
                l = criterion(outs, lbls)
                v_l += l.item() * imgs.size(0)
                v_tot += imgs.size(0)
        pt_val_loss.append(v_l / v_tot)

    time_pt = time.perf_counter() - t0_pt

    # Đánh giá tập test PyTorch & Trích xuất các dự đoán sai
    pt_net.eval()
    test_preds, test_targets, test_confs = [], [], []
    all_imgs = []
    with torch.no_grad():
        for imgs, lbls in datasets['mnist']['test']:
            all_imgs.append(imgs.cpu().numpy())
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = pt_net(imgs)
            probs = torch.softmax(outs, dim=1)
            conf, pred = probs.max(1)
            test_preds.extend(pred.cpu().numpy())
            test_targets.extend(lbls.cpu().numpy())
            test_confs.extend(conf.cpu().numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_confs = np.array(test_confs)
    all_imgs = np.concatenate(all_imgs, axis=0)

    metrics_pt = evaluate_metrics(test_targets, test_preds)
    metrics_pt['latency_sec'] = time_pt
    metrics_pt['params'] = sum(p.numel() for p in pt_net.parameters() if p.requires_grad)
    results['PyTorch'] = metrics_pt
    print(f"  [OK] PyTorch CNN Accuracy: {metrics_pt['accuracy']*100:.2f}% (Thời gian: {time_pt:.2f}s)")

    # ------------------------------------------------------------------
    # SINH CÁC BIỂU ĐỒ TRỌNG TÂM CẦN THIẾT
    # ------------------------------------------------------------------
    # Biểu đồ 2: So sánh Learning Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor='#FFFFFF')
    epochs_range = range(1, 6)
    
    # Loss curves
    ax1.plot(epochs_range, scratch_hist['train_loss'], 'k:', label='NumPy Train Loss', linewidth=1.4)
    ax1.plot(epochs_range, k_fit.history['loss'], 'k--', label='Keras Train Loss', linewidth=1.4)
    ax1.plot(epochs_range, pt_train_loss, 'k-', label='PyTorch Train Loss', linewidth=1.6)
    ax1.set_title('Cross-Entropy Loss vs. Epochs', fontsize=11, fontweight='bold', color='#000000')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True, edgecolor='#000000')

    # Accuracy curves
    ax2.plot(epochs_range, [a * 100 for a in scratch_hist['val_acc']], 'k:', label='NumPy Val Acc', linewidth=1.4)
    ax2.plot(epochs_range, [a * 100 for a in k_fit.history['val_accuracy']], 'k--', label='Keras Val Acc', linewidth=1.4)
    ax2.set_title('Validation Accuracy (%) vs. Epochs', fontsize=11, fontweight='bold', color='#000000')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(frameon=True, edgecolor='#000000')

    plt.suptitle("Hình 2: Diễn biến huấn luyện (Learning Curves) đối chuẩn giữa 3 cách cài đặt", 
                 fontsize=11, fontweight='bold', color='#000000', y=1.02)
    plt.tight_layout()
    plt.savefig('report_images/fig2_learning_curves_comparison.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("  [OK] Đã lưu Hình 2: report_images/fig2_learning_curves_comparison.png")

    # Biểu đồ 3: Normalized Confusion Matrix của mô hình tốt nhất (PyTorch)
    cm = confusion_matrix(test_targets, test_preds, normalize='true')
    plt.figure(figsize=(7, 6), facecolor='#FFFFFF')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Greys', cbar=True,
                xticklabels=[str(i) for i in range(10)],
                yticklabels=[str(i) for i in range(10)],
                linewidths=0.5, linecolor='#000000')
    plt.title("Hình 3: Ma trận nhầm lẫn chuẩn hóa của PyTorch CNN trên MNIST", 
              fontsize=11, fontweight='bold', color='#000000', pad=12)
    plt.xlabel("Nhãn Dự Đoán (Predicted)", fontsize=10, fontweight='bold', color='#000000')
    plt.ylabel("Nhãn Thực Tế (True Label)", fontsize=10, fontweight='bold', color='#000000')
    plt.tight_layout()
    plt.savefig('report_images/fig3_confusion_matrix_pytorch.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("  [OK] Đã lưu Hình 3: report_images/fig3_confusion_matrix_pytorch.png")

    # Biểu đồ 4: Phân tích các trường hợp sai có confidence cao nhất (Error Analysis)
    wrong_idx = np.where(test_targets != test_preds)[0]
    top_wrong = wrong_idx[np.argsort(test_confs[wrong_idx])[::-1][:5]]
    
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.5), facecolor='#FFFFFF')
    for i, idx in enumerate(top_wrong):
        axes[i].imshow(all_imgs[idx].squeeze(0), cmap='gray')
        axes[i].set_title(f"True: {test_targets[idx]}\nPred: {test_preds[idx]}\nConf: {test_confs[idx]*100:.1f}%",
                          fontsize=9, color='#000000')
        axes[i].axis('off')

    plt.suptitle("Hình 4: Phân tích các dự đoán sai có độ tin cậy (Confidence) cao nhất", 
                 fontsize=11, fontweight='bold', color='#000000', y=1.08)
    plt.tight_layout()
    plt.savefig('report_images/fig4_error_analysis.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("  [OK] Đã lưu Hình 4: report_images/fig4_error_analysis.png")

    # Lưu kết quả Benchmark
    with open('report_images/benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print("  [OK] Đã lưu bảng số liệu đối chuẩn: report_images/benchmark_results.json")

    print("\n====================================================================")
    print("BẢNG TỔNG HỢP KẾT QUẢ ĐỐI CHUẨN (BENCHMARK RESULTS):")
    print(f"{'Mô hình':<18} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Params':<10} | {'Thời gian (s)':<12}")
    print("-" * 72)
    for k, v in results.items():
        print(f"{k:<18} | {v['accuracy']*100:>8.2f}% | {v['macro_f1']*100:>8.2f}% | {v['params']:>10,d} | {v['latency_sec']:>10.2f}s")
    print("====================================================================")

if __name__ == '__main__':
    main()
