## Giới thiệu tổng quan

Dự án triển khai toàn diện và có hệ thống mô hình Mạng Nơ-ron Tích chập (Convolutional Neural Network - CNN) qua ba cấp độ hiện thực hóa:
1. **Từ số học thuần túy (NumPy from Scratch):** Tự xây dựng toàn bộ các tầng `Conv2D`, `MaxPool2D`, `ReLU`, `Flatten`, `Dense`, hàm mất mát `SoftmaxCrossEntropy` và bộ tối ưu hóa `Adam` sử dụng kỹ thuật vector hóa ma trận `im2col` và `col2im`.
2. **Kỹ nghệ công nghiệp cấp cao (TensorFlow / Keras):** Xây dựng kiến trúc CNN tiêu chuẩn với `Sequential API`, chuẩn hóa theo lô (`BatchNormalization`) và điều hòa `Dropout`.
3. **Mô hình nghiên cứu học thuật (PyTorch):** Thiết kế cấu trúc `nn.Module` linh hoạt với phương thức kiểm tra kích thước tensor trung gian (`inspect_shapes()`), hàm kiểm toán chi tiết tham số (`count_parameters_detailed()`) và vòng lặp huấn luyện tường minh 5 bước.

---

## Bảng đối chuẩn hiệu năng thực nghiệm

| Tiêu chí đối chuẩn | NumPy from Scratch | TensorFlow / Keras | PyTorch |
| :--- | :---: | :---: | :---: |
| **Độ chính xác kiểm thử (Accuracy)** | **93.00%** | **98.76%** | **99.19%** |
| **Macro-Precision** | 0.9304 | 0.9875 | **0.9918** |
| **Macro-Recall** | 0.9298 | 0.9874 | **0.9919** |
| **Macro-F1 Score** | 0.9299 | 0.9874 | **0.9918** |
| **Tổng số tham số học được** | 52,138 | 421,930 | 421,738 |
| **Thời gian huấn luyện** | 11.68 giây (1 epoch) | 68.18 giây (5 epochs) | 194.32 giây (5 epochs) |

---

## Cấu trúc thư mục

```text
├── BAO_CAO_ASSIGNMENT_04.pdf       # Báo cáo học thuật toàn văn (PDF)
├── notebooks/
│   └── Assignment_04_CNN_Benchmark.ipynb  # Sổ tay Jupyter Notebook thực nghiệm
├── src/
│   ├── data_loader.py               # Pipeline nạp & tiền xử lý 3 bộ dữ liệu
│   ├── models_keras.py              # Xây dựng mô hình Keras Sequential
│   ├── models_pytorch.py            # Mô hình PyTorch & hàm kiểm tra tensor
│   ├── utils.py                     # Hàm đánh giá chỉ số & vẽ biểu đồ 300 DPI
│   └── cnn_scratch/                 # Toàn bộ mã nguồn CNN tự viết bằng NumPy
│       ├── __init__.py
│       ├── layers.py                # Conv2D, MaxPool2D, ReLU, Flatten, Dense
│       ├── losses.py                # SoftmaxCrossEntropyLoss & AdamOptimizer
│       └── model.py                 # Lớp mô hình CNN_Scratch hoàn chỉnh
├── report_images/                   # Các biểu đồ kết quả trực quan hóa 300 DPI
│   ├── fig1_dataset_samples.png
│   ├── fig2_learning_curves_comparison.png
│   ├── fig3_confusion_matrix_pytorch.png
│   ├── fig4_error_analysis.png
│   └── benchmark_results.json
└── scripts/
    └── run_experiments.py          # Script chạy toàn bộ thực nghiệm đối chuẩn
```

---

## Hướng dẫn cài đặt và chạy thử

### 1. Cài đặt môi trường
```bash
git clone https://github.com/LeeMinhPhuowng/Assignment04.git
cd Assignment04
pip install numpy torch torchvision tensorflow scikit-learn matplotlib seaborn
```

### 2. Chạy sổ tay Jupyter Notebook
```bash
jupyter notebook notebooks/Assignment_04_CNN_Benchmark.ipynb
```

### 3. Tái lập toàn bộ thực nghiệm đối chuẩn qua dòng lệnh
```bash
python scripts/run_experiments.py
```
