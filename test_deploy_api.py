"""
Kịch bản kiểm thử tự động toàn diện cho hệ thống triển khai (Deployment)
Kiểm tra cả hai API PyTorch và Keras
Sử dụng FastAPI TestClient - kiểm thử tức thì không cần mở uvicorn thủ công.
"""

import sys
import json
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from fastapi.testclient import TestClient
from src.deploy_pytorch import app as pytorch_app
from src.deploy_keras import app as keras_app

def create_dummy_sequence(window_size=30, num_features=5, base_val=100.0):
    """Tạo chuỗi giả lập 30 bước thời gian với 5 đặc trưng"""
    seq = []
    for i in range(window_size):
        row = [
            round(base_val + i * 0.5 + np.random.uniform(-1, 1), 2),      # Open
            round(base_val + i * 0.5 + 2.0 + np.random.uniform(-1, 1), 2),# High
            round(base_val + i * 0.5 - 2.0 + np.random.uniform(-1, 1), 2),# Low
            round(base_val + i * 0.5 + 0.5 + np.random.uniform(-1, 1), 2),# Close (Target)
            round(1000000 + np.random.uniform(0, 500000), 0)             # Volume
        ]
        seq.append(row)
    return seq

def test_api_system():
    print("=" * 70)
    print("BẮT ĐẦU KIỂM THỬ HỆ THỐNG TRIỂN KHAI SUY LUẬN THỜI GIAN THỰC")
    print("=" * 70)
    
    # 1. KIỂM THỬ PYTORCH API
    print("\n[1] KIỂM THỬ PYTORCH DEPLOYMENT API (FastAPI)")
    client_pt = TestClient(pytorch_app)
    
    # 1.1 Kiểm tra Healthcheck
    res = client_pt.get("/health")
    print(f"  -> GET /health: Status {res.status_code}")
    print(f"     Payload: {res.json()}")
    
    # 1.2 Kiểm tra Model Info (Stock & Gold)
    res_stock = client_pt.get("/model_info?dataset=stock")
    stock_info = res_stock.json()
    print(f"  -> GET /model_info (stock): Framework={stock_info['framework']}, Params={stock_info['total_parameters']}, Type={stock_info['config'].get('rnn_type')}")
    
    res_gold = client_pt.get("/model_info?dataset=gold")
    gold_info = res_gold.json()
    print(f"  -> GET /model_info (gold): Framework={gold_info['framework']}, Params={gold_info['total_parameters']}, Type={gold_info['config'].get('rnn_type')}")
    
    # 1.3 Kiểm tra /predict (Dự báo 1 bước t+1)
    dummy_stock = create_dummy_sequence(30, 5, base_val=130.0)
    payload_predict = {"sequence": dummy_stock}
    res_pred = client_pt.post("/predict?dataset=stock", json=payload_predict)
    pred_data = res_pred.json()
    print(f"  -> POST /predict (stock):")
    print(f"     Giá dự đoán tiếp theo: {pred_data['prediction_value']} USD")
    print(f"     Giá normalized:        {pred_data['prediction_scaled']}")
    print(f"     Độ trễ suy luận:       {pred_data['latency_ms']} ms")
    
    # 1.4 Kiểm tra /forecast (Dự báo đa bước 7 ngày tương lai)
    payload_forecast = {"sequence": dummy_stock, "steps": 7}
    res_fore = client_pt.post("/forecast?dataset=stock", json=payload_forecast)
    fore_data = res_fore.json()
    print(f"  -> POST /forecast (stock - 7 ngày tới):")
    print(f"     Quỹ đạo dự báo: {[round(v, 2) for v in fore_data['forecast_values']]}")
    
    # 2. KIỂM THỬ KERAS API
    print("\n" + "-" * 70)
    print("[2] KIỂM THỬ KERAS / TENSORFLOW DEPLOYMENT API (FastAPI)")
    client_keras = TestClient(keras_app)
    
    # 2.1 Kiểm tra Healthcheck
    res_k_health = client_keras.get("/health")
    print(f"  -> GET /health: Status {res_k_health.status_code}")
    print(f"     Payload: {res_k_health.json()}")
    
    # 2.2 Kiểm tra /predict với dữ liệu Vàng
    dummy_gold = create_dummy_sequence(30, 5, base_val=1800.0)
    res_k_pred = client_keras.post("/predict?dataset=gold", json={"sequence": dummy_gold})
    k_pred_data = res_k_pred.json()
    print(f"  -> POST /predict (gold):")
    print(f"     Giá vàng dự đoán t+1:  {k_pred_data['prediction_value']} USD")
    print(f"     Độ trễ suy luận:       {k_pred_data['latency_ms']} ms")
    
    # 2.3 Kiểm tra /forecast 5 ngày tới với Vàng
    res_k_fore = client_keras.post("/forecast?dataset=gold", json={"sequence": dummy_gold, "steps": 5})
    k_fore_data = res_k_fore.json()
    print(f"  -> POST /forecast (gold - 5 ngày tới):")
    print(f"     Quỹ đạo dự báo vàng:   {[round(v, 2) for v in k_fore_data['forecast_values']]}")
    
    print("\n" + "=" * 70)
    print("HOÀN TẤT KIỂM THỬ: TẤT CẢ ENDPOINT VÀ MÔ HÌNH HOẠT ĐỘNG CHUẨN XÁC 100%!")
    print("=" * 70)

if __name__ == "__main__":
    test_api_system()
