"""
Module: deploy_keras.py
Mục đích: Triển khai (Deployment) mô hình RNN/LSTM/GRU xây dựng bằng TensorFlow/Keras 
phục vụ môi trường thực tế (Production Inference).
Cung cấp:
1. KerasInferenceEngine: Tải mô hình .keras, tiền xử lý tuần tự, dự báo 1 bước hoặc đa bước.
2. REST API với FastAPI: Các endpoint tiêu chuẩn /health, /model_info, /predict, /forecast.
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class KerasInferenceEngine:
    """Engine suy luận Keras tối ưu cho production"""
    def __init__(self, model_path: str, scaler_path: Optional[str] = None, target_idx: int = 0):
        self.model_path = model_path
        self.target_idx = target_idx
        self.scaler_min = None
        self.scaler_max = None
        
        # Tải mô hình Keras đã huấn luyện
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print(f"[Keras Engine] Đã nạp thành công mô hình từ: {model_path}")
        else:
            raise FileNotFoundError(f"Không tìm thấy file mô hình Keras tại {model_path}")
            
        # Nạp tham số chuẩn hóa Min-Max
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, "r") as f:
                sdata = json.load(f)
                self.scaler_min = np.array(sdata["min"], dtype=np.float32)
                self.scaler_max = np.array(sdata["max"], dtype=np.float32)
                self.target_idx = sdata.get("target_idx", self.target_idx)
                
    def preprocess(self, sequence: List[List[float]]) -> np.ndarray:
        """Chuẩn hóa dữ liệu thô đầu vào"""
        arr = np.array(sequence, dtype=np.float32)
        if self.scaler_min is not None and self.scaler_max is not None:
            denom = np.where(self.scaler_max - self.scaler_min == 0, 1.0, self.scaler_max - self.scaler_min)
            arr = (arr - self.scaler_min) / denom
        # Keras yêu cầu shape (batch_size, window_size, num_features)
        arr = np.expand_dims(arr, axis=0)
        return arr
    
    def postprocess(self, pred_scaled: float) -> float:
        """Khôi phục về đơn vị gốc"""
        if self.scaler_min is not None and self.scaler_max is not None:
            min_val = self.scaler_min[self.target_idx]
            max_val = self.scaler_max[self.target_idx]
            return float(pred_scaled * (max_val - min_val) + min_val)
        return float(pred_scaled)
    
    def predict_next_step(self, sequence: List[List[float]]) -> Dict[str, Any]:
        """Dự báo 1 bước thời gian tiếp theo t+1"""
        t0 = time.time()
        arr = self.preprocess(sequence)
        pred_scaled = float(self.model.predict(arr, verbose=0).flatten()[0])
        pred_value = self.postprocess(pred_scaled)
        latency_ms = (time.time() - t0) * 1000.0
        
        return {
            "prediction_scaled": round(pred_scaled, 6),
            "prediction_value": round(pred_value, 4),
            "latency_ms": round(latency_ms, 2)
        }
        
    def forecast_multistep(self, sequence: List[List[float]], steps: int = 7) -> List[float]:
        """Dự báo đa bước tương lai theo cơ chế tự hồi quy (Autoregressive Rollout)"""
        current_seq = np.array(sequence, dtype=np.float32)
        predictions = []
        
        for _ in range(steps):
            res = self.predict_next_step(current_seq.tolist())
            pred_val = res["prediction_value"]
            predictions.append(pred_val)
            
            # Tạo vector mới
            next_row = current_seq[-1].copy()
            next_row[self.target_idx] = pred_val
            
            current_seq = np.vstack([current_seq[1:], next_row])
            
        return predictions

# Khởi tạo ứng dụng FastAPI cho Keras
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Time Series Keras RNN Deployment API",
    description="Hệ thống REST API phục vụ suy luận dự báo chuỗi thời gian dựa trên TensorFlow/Keras",
    version="1.0.0"
)

class SequencePayload(BaseModel):
    sequence: List[List[float]] = Field(..., description="Cửa sổ trượt các bước thời gian quá khứ [W x num_features]")

class ForecastPayload(BaseModel):
    sequence: List[List[float]] = Field(..., description="Cửa sổ trượt quá khứ")
    steps: int = Field(default=7, ge=1, le=30, description="Số bước cần dự báo tiếp theo")

engine_instances: Dict[str, KerasInferenceEngine] = {}

def get_engine(dataset: str = "stock") -> KerasInferenceEngine:
    dataset = dataset.lower()
    if dataset not in engine_instances:
        if dataset == "gold":
            model_path = "models/gold_keras_lstm.keras"
            scaler_path = "models/gold_scaler_params.json"
            target_idx = 0
        else:
            model_path = "models/stock_keras_lstm.keras"
            scaler_path = "models/stock_scaler_params.json"
            target_idx = 3
            
        engine_instances[dataset] = KerasInferenceEngine(
            model_path=model_path,
            scaler_path=scaler_path,
            target_idx=target_idx
        )
    return engine_instances[dataset]

@app.get("/health", summary="Kiểm tra trạng thái dịch vụ")
def health(dataset: str = "stock"):
    return {
        "status": "online",
        "dataset": dataset,
        "framework": "TensorFlow/Keras",
        "timestamp": time.time()
    }

@app.get("/model_info", summary="Thông tin kiến trúc mô hình")
def model_info(dataset: str = "stock"):
    eng = get_engine(dataset)
    return {
        "dataset": dataset,
        "framework": "TensorFlow/Keras",
        "total_parameters": eng.model.count_params(),
        "input_shape": str(eng.model.input_shape),
        "target_index": eng.target_idx
    }

@app.post("/predict", summary="Dự đoán một bước thời gian tiếp theo")
def predict(payload: SequencePayload, dataset: str = "stock"):
    eng = get_engine(dataset)
    try:
        result = eng.predict_next_step(payload.sequence)
        result["dataset"] = dataset
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi suy luận: {str(e)}")

@app.post("/forecast", summary="Dự báo tương lai nhiều bước")
def forecast(payload: ForecastPayload, dataset: str = "stock"):
    eng = get_engine(dataset)
    try:
        preds = eng.forecast_multistep(payload.sequence, steps=payload.steps)
        return {
            "dataset": dataset,
            "steps": payload.steps,
            "forecast_values": preds
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi dự báo đa bước: {str(e)}")

if __name__ == "__main__":
    print("=== KIỂM THỬ TRIỂN KHAI KERAS INFERENCE ENGINE ===")
    if os.path.exists("models/test_stock_keras.keras"):
        engine = KerasInferenceEngine("models/test_stock_keras.keras", target_idx=3)
        dummy_seq = np.random.uniform(50.0, 150.0, size=(30, 5)).tolist()
        pred_res = engine.predict_next_step(dummy_seq)
        print("Kết quả suy luận 1 bước:", pred_res)
        forecast_res = engine.forecast_multistep(dummy_seq, steps=5)
        print("Kết quả dự báo 5 ngày tiếp theo:", forecast_res)
        print("Engine Keras sẵn sàng tích hợp và vận hành Production!")
    else:
        print("Vui lòng huấn luyện mô hình Keras trước khi kiểm thử deploy.")
