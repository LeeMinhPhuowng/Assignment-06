"""
Module: models_keras.py
Mục đích: Xây dựng, huấn luyện và đánh giá mô hình mạng nơ-ron hồi quy với TensorFlow/Keras.
Hỗ trợ các biến thể:
- Vanilla RNN (SimpleRNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
Bao gồm:
- Thiết kế mô hình tuần tự qua Keras Sequential API
- Callbacks công nghiệp: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Đánh giá các chỉ số thực tế: RMSE, MAE, MAPE, R2
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def build_keras_rnn_model(input_shape, hidden_dim=64, num_layers=2, dropout=0.2, rnn_type="LSTM"):
    """
    Xây dựng kiến trúc RNN với Keras:
    input_shape: (window_size, num_features)
    rnn_type: 'RNN' (SimpleRNN), 'LSTM', hoặc 'GRU'
    """
    model = keras.Sequential(name=f"Keras_{rnn_type.upper()}_Model")
    model.add(layers.Input(shape=input_shape))
    
    # Các tầng hồi quy xếp chồng (Stacked Recurrent Layers)
    for i in range(num_layers):
        is_last = (i == num_layers - 1)
        return_sequences = not is_last
        
        if rnn_type.upper() == "RNN":
            model.add(layers.SimpleRNN(hidden_dim, return_sequences=return_sequences))
        elif rnn_type.upper() == "LSTM":
            model.add(layers.LSTM(hidden_dim, return_sequences=return_sequences))
        elif rnn_type.upper() == "GRU":
            model.add(layers.GRU(hidden_dim, return_sequences=return_sequences))
        else:
            raise ValueError(f"Kiến trúc không được hỗ trợ: {rnn_type}")
            
        if dropout > 0.0:
            model.add(layers.Dropout(dropout))
            
    # Tầng Fully Connected Head
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1, activation="linear"))
    
    return model

def compile_and_train_keras(model, X_train, y_train, X_val, y_val, 
                            epochs=25, batch_size=64, lr=0.001, 
                            save_path="models/keras_model.keras", patience=7):
    """
    Biên dịch và huấn luyện mô hình Keras với các callbacks chống overfitting:
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    optimizer = optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="huber", metrics=["mae", "mse"])
    
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0),
        callbacks.ModelCheckpoint(filepath=save_path, monitor="val_loss", save_best_only=True, verbose=0)
    ]
    
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=0
    )
    total_duration = time.time() - t0
    
    hist_dict = {
        "train_loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "total_duration": total_duration
    }
    
    return model, hist_dict

def evaluate_keras_model(model, X_test, y_test, pipeline):
    """Đánh giá mô hình Keras trên tập kiểm thử"""
    t0 = time.time()
    y_preds_scaled = model.predict(X_test, verbose=0).flatten()
    total_infer_time = time.time() - t0
    
    y_true_scaled = y_test.flatten()
    
    # Khôi phục về thang đo gốc
    y_preds = pipeline.inverse_transform_target(y_preds_scaled)
    y_true = pipeline.inverse_transform_target(y_true_scaled)
    
    rmse = np.sqrt(np.mean((y_preds - y_true) ** 2))
    mae = np.mean(np.abs(y_preds - y_true))
    mape = np.mean(np.abs((y_true - y_preds) / np.where(y_true == 0, 1e-6, y_true))) * 100.0
    ss_res = np.sum((y_true - y_preds) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    avg_latency_ms = (total_infer_time / len(X_test)) * 1000.0
    
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "avg_latency_ms": float(avg_latency_ms)
    }
    
    return metrics, y_preds, y_true

if __name__ == "__main__":
    from data_preprocessing import TimeSeriesDataPipeline
    print("=== KIỂM THỬ XÂY DỰNG MÔ HÌNH KERAS ===")
    pipe = TimeSeriesDataPipeline(dataset_name="stock", window_size=30)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), info = pipe.temporal_split()
    
    model = build_keras_rnn_model(input_shape=(30, 5), hidden_dim=32, num_layers=2, rnn_type="LSTM")
    model.summary()
    model, hist = compile_and_train_keras(model, X_tr, y_tr, X_va, y_va, epochs=5, save_path="models/test_stock_keras.keras")
    metrics, preds, trues = evaluate_keras_model(model, X_te, y_te, pipe)
    print("Kết quả đánh giá trên tập kiểm thử Keras:")
    print(metrics)
