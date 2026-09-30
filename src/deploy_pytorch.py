"""
Module: deploy_pytorch.py
Mục đích: Triển khai (Deployment) mô hình RNN/LSTM PyTorch phục vụ suy luận thời gian thực (Real-time Inference).
Cung cấp:
1. PyTorchInferenceEngine: Engine suy luận độc lập, nạp trọng số, chuẩn hóa và dự báo.
2. REST API với FastAPI: Các endpoint /health, /model_info, /predict, /forecast.
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from src.models_pytorch import PyTorchRNNModel
except ImportError:
    from models_pytorch import PyTorchRNNModel


class PyTorchInferenceEngine:
    """Engine suy luận PyTorch hiệu năng cao phục vụ môi trường Production"""
    def __init__(self, model_path: str, model_config: dict, scaler_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_config = model_config
        
        # Khởi tạo kiến trúc
        self.model = PyTorchRNNModel(
            input_dim=model_config.get("input_dim", 5),
            hidden_dim=model_config.get("hidden_dim", 64),
            num_layers=model_config.get("num_layers", 2),
            output_dim=model_config.get("output_dim", 1),
            dropout=model_config.get("dropout", 0.0),
            rnn_type=model_config.get("rnn_type", "LSTM")
        )
        
        # Nạp trọng số
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(f"[PyTorch Engine] Đã nạp thành công mô hình từ: {model_path}")
        else:
            print(f"[Cảnh báo] Không tìm thấy file trọng số {model_path}, mô hình sẽ sử dụng trọng số khởi tạo ban đầu.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Nạp tham số chuẩn hóa Min-Max
        self.scaler_min = None
        self.scaler_max = None
        self.target_idx = model_config.get("target_idx", 3)
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, "r") as f:
                sdata = json.load(f)
                self.scaler_min = np.array(sdata["min"], dtype=np.float32)
                self.scaler_max = np.array(sdata["max"], dtype=np.float32)
                self.target_idx = sdata.get("target_idx", self.target_idx)
                
    def preprocess(self, sequence: List[List[float]]) -> torch.Tensor:
        """Chuẩn hóa dữ liệu thô và chuyển thành tensor"""
        arr = np.array(sequence, dtype=np.float32)
        if self.scaler_min is not None and self.scaler_max is not None:
            denom = np.where(self.scaler_max - self.scaler_min == 0, 1.0, self.scaler_max - self.scaler_min)
            arr = (arr - self.scaler_min) / denom
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor
    
    def postprocess(self, pred_scaled: float) -> float:
        """Khôi phục về đơn vị đo lường thực tế"""
        if self.scaler_min is not None and self.scaler_max is not None:
            min_val = self.scaler_min[self.target_idx]
            max_val = self.scaler_max[self.target_idx]
            return float(pred_scaled * (max_val - min_val) + min_val)
        return float(pred_scaled)
    
    def predict_next_step(self, sequence: List[List[float]]) -> Dict[str, Any]:
        """Dự báo 1 bước thời gian tiếp theo t+1"""
        t0 = time.time()
        tensor = self.preprocess(sequence)
        with torch.no_grad():
            output = self.model(tensor)
            pred_scaled = output.item()
            
        pred_value = self.postprocess(pred_scaled)
        latency_ms = (time.time() - t0) * 1000.0
        
        return {
            "prediction_scaled": round(pred_scaled, 6),
            "prediction_value": round(pred_value, 4),
            "latency_ms": round(latency_ms, 2)
        }
        
    def forecast_multistep(self, sequence: List[List[float]], steps: int = 7) -> List[float]:
        """Dự báo đa bước tương lai theo phương thức tự hồi quy (Autoregressive Rollout)"""
        current_seq = np.array(sequence, dtype=np.float32)
        predictions = []
        
        for _ in range(steps):
            res = self.predict_next_step(current_seq.tolist())
            pred_val = res["prediction_value"]
            predictions.append(pred_val)
            
            # Tạo vector bước tiếp theo bằng cách lặp lại vector cuối và cập nhật biến mục tiêu
            next_row = current_seq[-1].copy()
            next_row[self.target_idx] = pred_val
            
            # Trượt cửa sổ sang phải
            current_seq = np.vstack([current_seq[1:], next_row])
            
        return predictions

# Khởi tạo ứng dụng FastAPI
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Time Series PyTorch RNN Deployment API",
    description="Hệ thống REST API phục vụ suy luận dự báo chuỗi thời gian dựa trên PyTorch RNN/LSTM",
    version="1.0.0"
)

# Schema xác thực dữ liệu đầu vào
class SequencePayload(BaseModel):
    sequence: List[List[float]] = Field(..., description="Cửa sổ trượt các bước thời gian quá khứ [W x num_features]")

class ForecastPayload(BaseModel):
    sequence: List[List[float]] = Field(..., description="Cửa sổ trượt quá khứ")
    steps: int = Field(default=7, ge=1, le=30, description="Số bước cần dự báo tiếp theo")

# Quản lý Engine cho từng bộ dữ liệu (stock hoặc gold)
engine_instances: Dict[str, PyTorchInferenceEngine] = {}

def get_engine(dataset: str = "stock") -> PyTorchInferenceEngine:
    dataset = dataset.lower()
    if dataset not in engine_instances:
        if dataset == "gold":
            config = {
                "input_dim": 5,
                "hidden_dim": 64,
                "num_layers": 2,
                "output_dim": 1,
                "rnn_type": "LSTM",
                "target_idx": 0
            }
            model_path = "models/gold_pytorch_lstm.pt"
            scaler_path = "models/gold_scaler_params.json"
        else:
            config = {
                "input_dim": 5,
                "hidden_dim": 64,
                "num_layers": 2,
                "output_dim": 1,
                "rnn_type": "LSTM",
                "target_idx": 3
            }
            model_path = "models/stock_pytorch_lstm.pt"
            scaler_path = "models/stock_scaler_params.json"
            
        engine_instances[dataset] = PyTorchInferenceEngine(
            model_path=model_path,
            model_config=config,
            scaler_path=scaler_path
        )
    return engine_instances[dataset]

@app.get("/health", summary="Kiểm tra trạng thái dịch vụ")
def health(dataset: str = "stock"):
    eng = get_engine(dataset)
    return {
        "status": "online",
        "dataset": dataset,
        "framework": "PyTorch",
        "model_type": eng.model_config.get("rnn_type"),
        "timestamp": time.time()
    }

@app.get("/model_info", summary="Thông tin kiến trúc mô hình")
def model_info(dataset: str = "stock"):
    eng = get_engine(dataset)
    total_params = sum(p.numel() for p in eng.model.parameters())
    return {
        "dataset": dataset,
        "framework": "PyTorch",
        "config": eng.model_config,
        "total_parameters": total_params,
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

@app.post("/forecast", summary="Dự báo tương lai nhiều bước (Autoregressive Rollout)")
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
    print("=== KIỂM THỬ TRIỂN KHAI PYTORCH INFERENCE ENGINE ===")
    config = {"input_dim": 5, "hidden_dim": 32, "num_layers": 2, "rnn_type": "LSTM", "target_idx": 3}
    dummy_engine = PyTorchInferenceEngine("models/test_stock_lstm.pt", config)
    
    # Giả lập cửa sổ 30 ngày với 5 thuộc tính
    dummy_seq = np.random.uniform(50.0, 150.0, size=(30, 5)).tolist()
    pred_res = dummy_engine.predict_next_step(dummy_seq)
    print("Kết quả suy luận 1 bước:", pred_res)
    
    forecast_res = dummy_engine.forecast_multistep(dummy_seq, steps=5)
    print("Kết quả dự báo 5 ngày tiếp theo:", forecast_res)
    print("Engine PyTorch sẵn sàng tích hợp và vận hành Production!")
