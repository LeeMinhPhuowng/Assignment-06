# Assignment 06: Dự Báo Chuỗi Thời Gian Với Mạng Nơ-ron Hồi Quy (RNN, LSTM, GRU) & Triển Khai Dịch Vụ Thời Gian Thực

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![TensorFlow/Keras](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-red.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/Deployment-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Kho lưu trữ mã nguồn và báo cáo khoa học cho **Assignment 06** - Nghiên cứu thực nghiệm, đối chuẩn hiệu năng họ mạng nơ-ron hồi quy (Vanilla RNN, LSTM, GRU) trên hai tập dữ liệu chuỗi thời gian thực tế (**Cổ phiếu Amazon** và **Giá vàng thế giới**), đồng thời xây dựng hệ thống microservice triển khai suy luận thời gian thực với FastAPI.

---

## 📌 Tổng Quan Mục Tiêu Dự Án

Dự án hoàn thành toàn diện 5 mục tiêu học thuật và kỹ thuật:
1. **Cơ sở lý thuyết & Cài đặt NumPy Scratch:** Thiết lập nền tảng toán học giải tích, cơ chế cuộn mở qua thời gian (Unrolling), giải thuật lan truyền ngược qua thời gian (BPTT), chứng minh hiện tượng triệt tiêu/bùng nổ gradient qua tích chuỗi Jacobi, và triển khai lớp `VanillaRNNScratch` thuần túy bằng NumPy.
2. **Khảo sát & Tiền xử lý 2 tập dữ liệu chuỗi thời gian:**
   - **Cổ phiếu Amazon (`data/amazon_stock_price.csv`):** 6,684 phiên giao dịch (1997 - 2023), phân phối lợi suất đuôi dày, biến động phi tuyến tính.
   - **Giá vàng quốc tế (`data/gold_price.csv`):** 13,320 quan sát lịch sử (1968 - 2021), tài sản trú ẩn an toàn với các chu kỳ kinh tế vĩ mô.
   - Pipeline phân tách dữ liệu theo trật tự thời gian (70% Train, 15% Validation, 15% Test) chống rò rỉ dữ liệu (Data Leakage) và trích xuất cửa sổ trượt $W = 30$ ngày.
3. **Mô hình hóa với PyTorch:** Module hóa hướng đối tượng `PyTorchRNNModel` linh hoạt (RNN/LSTM/GRU), tối ưu AdamW kết hợp hàm mất mát Huber Loss và kiểm soát bùng nổ đạo hàm bằng Gradient Clipping.
4. **Mô hình hóa với TensorFlow/Keras:** Xây dựng mạng Sequential xếp tầng LSTM kết hợp Spatial Dropout và hệ thống Callbacks công nghiệp (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`).
5. **Triển khai hệ thống suy luận thời gian thực (Deployment):** Đóng gói Inference Engine độc lập, tách rời tham số chuẩn hóa sang JSON, triển khai RESTful API với FastAPI hỗ trợ dự báo 1 bước (`/predict`) và dự báo tự hồi quy đa bước (`/forecast`).

---

## 📊 Bảng Đối Chuẩn Hiệu Năng Thực Nghiệm (Benchmark Results)

Kết quả đo lường định lượng trên tập kiểm thử độc lập (Test Set):

| Tập dữ liệu | Kiến trúc mô hình | Nền tảng framework | RMSE | MAE | MAPE (%) | Hệ số $R^2$ | Độ trễ suy luận (ms/mẫu) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cổ phiếu Amazon** | Vanilla RNN | PyTorch | **3.23 USD** | **2.55 USD** | **2.15%** | **0.9641** | 0.0078 ms |
| **Cổ phiếu Amazon** | GRU | PyTorch | 3.35 USD | 2.59 USD | 2.18% | 0.9613 | 0.0244 ms |
| **Cổ phiếu Amazon** | LSTM | PyTorch | 4.19 USD | 3.31 USD | 2.76% | 0.9396 | **0.0078 ms** |
| **Cổ phiếu Amazon** | LSTM | Keras | 4.31 USD | 3.42 USD | 2.86% | 0.9361 | 2.0578 ms |
| **Giá vàng thế giới** | Vanilla RNN | PyTorch | **34.57 USD** | **25.61 USD** | **1.43%** | **0.9574** | 0.0049 ms |
| **Giá vàng thế giới** | GRU | PyTorch | 35.78 USD | 27.84 USD | 1.57% | 0.9544 | 0.0104 ms |
| **Giá vàng thế giới** | LSTM | PyTorch | 35.88 USD | 26.32 USD | 1.47% | 0.9541 | **0.0040 ms** |
| **Giá vàng thế giới** | LSTM | Keras | 37.72 USD | 28.05 USD | 1.66% | 0.9493 | 1.0741 ms |

---

## 🗂️ Cấu Trúc Thư Mục Dự Án

```text
├── data/                               # Dữ liệu chuỗi thời gian
│   ├── amazon_stock_price.csv          # Dữ liệu giá cổ phiếu Amazon
│   └── gold_price.csv                  # Dữ liệu giá vàng thế giới
├── models/                             # Checkpoints trọng số & tham số chuẩn hóa
│   ├── stock_pytorch_lstm.pt           # Trọng số PyTorch LSTM (Stock)
│   ├── gold_pytorch_lstm.pt            # Trọng số PyTorch LSTM (Gold)
│   ├── stock_keras_lstm.keras          # Trọng số Keras LSTM (Stock)
│   ├── gold_keras_lstm.keras           # Trọng số Keras LSTM (Gold)
│   ├── stock_scaler_params.json        # Metadata MinMax Scaler (Stock)
│   └── gold_scaler_params.json         # Metadata MinMax Scaler (Gold)
├── notebooks/                          # Jupyter Notebook thực nghiệm tương tác
│   └── Assignment06.ipynb              # Notebook hoàn chỉnh 4 phần
├── report_images/                      # 11 biểu đồ học thuật độ phân giải 300 DPI
│   ├── fig1_stock_series_overview.png
│   ├── fig2_stock_distribution_and_volatility.png
│   ├── fig3_retail_series_overview.png
│   ├── fig4_retail_seasonality_decomposition.png
│   ├── fig5_autocorrelation_analysis.png
│   ├── fig6_sliding_window_architecture.png
│   ├── fig7_learning_curves_stock.png
│   ├── fig8_learning_curves_retail.png
│   ├── fig9_stock_predictions_vs_actual.png
│   ├── fig10_retail_predictions_vs_actual.png
│   ├── fig11_deployment_latency_and_metrics.png
│   └── benchmark_results.json          # File JSON số liệu đối chuẩn thực nghiệm
├── src/                                # Mã nguồn module hóa
│   ├── rnn_scratch.py                  # Thuật toán Vanilla RNN thuần NumPy & BPTT
│   ├── data_preprocessing.py          # Pipeline tiền xử lý & Sliding Window
│   ├── models_pytorch.py               # Kiến trúc mạng & Vòng lặp PyTorch
│   ├── models_keras.py                 # Kiến trúc mạng & Callbacks Keras
│   ├── deploy_pytorch.py               # Engine suy luận & REST API PyTorch (FastAPI)
│   ├── deploy_keras.py                 # Engine suy luận & REST API Keras (FastAPI)
│   ├── eda_visualizer.py               # Module vẽ biểu đồ phân rã chuỗi thời gian
│   └── utils.py                        # Các hàm tính toán chỉ số đo lường (RMSE, MAPE, R2)
├── test_deploy_api.py                  # Kịch bản kiểm thử tự động toàn diện API
├── convert_report.py                   # Bộ biên dịch báo cáo MD sang HTML chuẩn in ấn
├── BAO_CAO_ASSIGNMENT_04.md            # Báo cáo khoa học học thuật chi tiết (~11,000 từ)
├── BAO_CAO_ASSIGNMENT_04.html          # Bản HTML chuẩn in ấn tích hợp MathJax 3
├── requirements.md                     # Đặc tả yêu cầu kỹ thuật & hình thức
└── README.md                           # Tài liệu hướng dẫn dự án
```

---

## 🚀 Cài Đặt & Hướng Dẫn Chạy

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 trở lên:
```bash
pip install torch torchvision torchaudio
pip install tensorflow keras
pip install fastapi uvicorn pydantic requests
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels markdown
```

### 2. Kiểm thử tự động hệ thống triển khai API (Fastest Test)
Chạy script kiểm thử tích hợp không cần bật server thủ công:
```bash
python test_deploy_api.py
```

### 3. Khởi chạy Server REST API thực tế (Production Mode)

- **Dịch vụ PyTorch Inference Service (Cổng 8000):**
  ```bash
  uvicorn src.deploy_pytorch:app --host 0.0.0.0 --port 8000 --reload
  ```
  Truy cập Swagger UI kiểm thử: [http://localhost:8000/docs](http://localhost:8000/docs)

- **Dịch vụ Keras Inference Service (Cổng 8001):**
  ```bash
  uvicorn src.deploy_keras:app --host 0.0.0.0 --port 8001 --reload
  ```
  Truy cập Swagger UI kiểm thử: [http://localhost:8001/docs](http://localhost:8001/docs)

### 4. Các Endpoints API chính

| Phương thức | Endpoint | Tham số | Chức năng |
| :---: | :--- | :--- | :--- |
| `GET` | `/health` | `?dataset=stock/gold` | Kiểm tra tình trạng sức khỏe dịch vụ |
| `GET` | `/model_info` | `?dataset=stock/gold` | Trả về thông tin tham số, kiến trúc mạng, kích thước cửa sổ |
| `POST` | `/predict` | `?dataset=stock/gold` | Dự báo giá đóng cửa bước tiếp theo ($t+1$) kèm độ trễ (`latency_ms`) |
| `POST` | `/forecast` | `?dataset=stock/gold` | Dự báo tự hồi quy đa bước tương lai ($H$ ngày tới) |

---

## 📖 Báo Cáo Khoa Học Học Thuật

Toàn văn báo cáo khoa học gồm 7 chương chi tiết được lưu trữ tại:
- **Tệp Markdown:** [BAO_CAO_ASSIGNMENT_04.md](BAO_CAO_ASSIGNMENT_04.md)
- **Tệp HTML chuẩn in ấn (Print-to-PDF):** [BAO_CAO_ASSIGNMENT_04.html](BAO_CAO_ASSIGNMENT_04.html)
  *(Mở tệp HTML trên trình duyệt $\rightarrow$ Nhấn `Ctrl + P` $\rightarrow$ Chọn `Save as PDF` để xuất tài liệu in ấn khổ A4 hoàn hảo)*.
