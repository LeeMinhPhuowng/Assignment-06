# Assignment 06: Du Bao Chuoi Thoi Gian Voi Mang No-ron Hoi Quy (RNN, LSTM, GRU) va Trien Khai Dich Vu Thoi Gian Thuc

Kho luu tru ma nguon va thuc nghiem cho Assignment 06 - Nghien cuu, doi chuan hieu nang ho mang no-ron hoi quy (Vanilla RNN, LSTM, GRU) tren hai tap du lieu chuoi thoi gian thuc te (Co phieu Amazon va Gia vang the gioi), dong thoi xay dung he thong REST API phuc vu suy luan thoi gian thuc voi FastAPI.

---

## 1. Tong Quan Yeu Cau Bai Tap

Du an giai quyet tron ven 4 noi dung chinh theo yeu cau:
1. Cac khai niem co ban ve RNN, bieu dien ham toan hoc, co che lan truyen nguoc qua thoi gian (BPTT), hien tuong triet tieu gradient va cai dat thuan NumPy scratch.
2. Khao sat va tien xu ly 2 tap du lieu chuoi thoi gian:
   - Co phieu Amazon (`data/amazon_stock_price.csv`): Thi truong chung khoan, do bien dong cao.
   - Gia vang the gioi (`data/gold_price.csv`): Chuoi giao dich hang hoa tich luy qua cac chu ky kinh te.
   - Pipeline cua so truot (Sliding Window, W = 30) va chia tap Train/Val/Test theo trat tu thoi gian tuyen tinh (70/15/15) chong ro ri thong tin.
3. Xay dung mo hinh RNN voi PyTorch de du doan va trien khai (Deploy):
   - Kien truc tuy bien ho tro Vanilla RNN, LSTM, GRU ke thua `torch.nn.Module`.
   - Vong lap huan luyen tuong minh voi AdamW, Huber Loss va Gradient Clipping.
   - Trien khai Engine suy luan va FastAPI service tai `src/deploy_pytorch.py`.
4. Xay dung mo hinh RNN voi TensorFlow/Keras de du doan va trien khai (Deploy):
   - Kien truc Sequential xep tang LSTM voi Spatial Dropout.
   - He thong Callbacks giam sat: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
   - Trien khai Engine suy luan va FastAPI service tai `src/deploy_keras.py`.

---

## 2. Bang Doi Chuan Hieu Nang Thuc Nghiem

Ket qua danh gia dinh luong tren tap kiem thu doc lap (Test Set):

| Tap du lieu | Kien truc mo hinh | Nen tang framework | RMSE | MAE | MAPE (%) | He so R2 | Do tre suy luan (ms/mau) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Co phieu Amazon | Vanilla RNN | PyTorch | 3.23 USD | 2.55 USD | 2.15% | 0.9641 | 0.0078 ms |
| Co phieu Amazon | GRU | PyTorch | 3.35 USD | 2.59 USD | 2.18% | 0.9613 | 0.0244 ms |
| Co phieu Amazon | LSTM | PyTorch | 4.19 USD | 3.31 USD | 2.76% | 0.9396 | 0.0078 ms |
| Co phieu Amazon | LSTM | Keras | 4.31 USD | 3.42 USD | 2.86% | 0.9361 | 2.0578 ms |
| Gia vang the gioi | Vanilla RNN | PyTorch | 34.57 USD | 25.61 USD | 1.43% | 0.9574 | 0.0049 ms |
| Gia vang the gioi | GRU | PyTorch | 35.78 USD | 27.84 USD | 1.57% | 0.9544 | 0.0104 ms |
| Gia vang the gioi | LSTM | PyTorch | 35.88 USD | 26.32 USD | 1.47% | 0.9541 | 0.0040 ms |
| Gia vang the gioi | LSTM | Keras | 37.72 USD | 28.05 USD | 1.66% | 0.9493 | 1.0741 ms |

---

## 3. Cau Truc Thu Muc Du An

```text
├── data/                               # Du lieu chuoi thoi gian
│   ├── amazon_stock_price.csv          # Du lieu gia co phieu Amazon
│   └── gold_price.csv                  # Du lieu gia vang the gioi
├── models/                             # Trong so mo hinh & tham so chuan hoa
│   ├── stock_pytorch_lstm.pt           # Checkpoint PyTorch LSTM (Stock)
│   ├── gold_pytorch_lstm.pt            # Checkpoint PyTorch LSTM (Gold)
│   ├── stock_keras_lstm.keras          # Checkpoint Keras LSTM (Stock)
│   ├── gold_keras_lstm.keras           # Checkpoint Keras LSTM (Gold)
│   ├── stock_scaler_params.json        # Tham so MinMaxScaler (Stock)
│   └── gold_scaler_params.json         # Tham so MinMaxScaler (Gold)
├── notebooks/                          # Jupyter Notebook thuc nghiem
│   └── Assignment06.ipynb              # Notebook hoan chinh 4 phan
├── src/                                # Ma nguon module hoa
│   ├── rnn_scratch.py                  # Thuat toan Vanilla RNN thuan NumPy & BPTT
│   ├── data_preprocessing.py          # Pipeline tien xu ly & Sliding Window
│   ├── models_pytorch.py               # Kien truc mo hinh PyTorch & Training Loop
│   ├── models_keras.py                 # Kien truc mo hinh Keras & Callbacks
│   ├── deploy_pytorch.py               # Engine suy luan & REST API PyTorch (FastAPI)
│   └── deploy_keras.py                 # Engine suy luan & REST API Keras (FastAPI)
├── requirements.md                     # Yeu cau bai tap
└── README.md                           # Tai lieu huong dan
```

---

## 4. Cai Dat va Huong Dan Su Dung

### Cai dat thu vien
Yeu cau moi truong Python 3.10 tro len:
```bash
pip install torch torchvision torchaudio
pip install tensorflow keras
pip install fastapi uvicorn pydantic requests
pip install numpy pandas matplotlib scikit-learn
```

### Chay thuc nghiem Jupyter Notebook
Mo va chay tuan tu cac cell trong notebook:
```bash
jupyter notebook notebooks/Assignment06.ipynb
```

### Khoi chay dich vu REST API suy luan thoi gian thuc

- Khoi dong API PyTorch (Cong 8000):
  ```bash
  uvicorn src.deploy_pytorch:app --host 0.0.0.0 --port 8000 --reload
  ```
  Swagger UI: http://localhost:8000/docs

- Khoi dong API Keras (Cong 8001):
  ```bash
  uvicorn src.deploy_keras:app --host 0.0.0.0 --port 8001 --reload
  ```
  Swagger UI: http://localhost:8001/docs

### Cac Endpoints API chinh

| Phuong thuc | Endpoint | Tham so | Chuc nang |
| :---: | :--- | :--- | :--- |
| GET | `/health` | `?dataset=stock/gold` | Kiem tra trang thai hoat dong cua he thong |
| GET | `/model_info` | `?dataset=stock/gold` | Thong tin kien truc mang, so tham so, kich thuoc cua so |
| POST | `/predict` | `?dataset=stock/gold` | Du bao gia dong cua buoc ke tiep (t+1) kem do tre (latency_ms) |
| POST | `/forecast` | `?dataset=stock/gold` | Du bao tu hoi quy da buoc tuong lai (H ngay tiep theo) |
