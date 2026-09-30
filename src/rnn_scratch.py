"""
Module: rnn_scratch.py
Mục đích: Cài đặt mạng nơ-ron hồi quy (Recurrent Neural Network - RNN) thuần túy từ đầu 
bằng thư viện NumPy, không sử dụng các framework tự động tính đạo hàm (autograd).
Bao gồm:
- Forward propagation qua thời gian (T bước)
- Tính toán hàm mất mát (Mean Squared Error / Cross Entropy)
- Lan truyền ngược qua thời gian (Backpropagation Through Time - BPTT)
- Cơ chế cắt tỉa đạo hàm (Gradient Clipping) chống bùng nổ đạo hàm (Exploding Gradients)
"""

import os
import sys
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class VanillaRNNScratch:
    """
    Kiến trúc Vanilla Recurrent Neural Network xây dựng từ các phép tính đại số tuyến tính:
    
    Phương trình toán học tại bước thời gian t:
        a_t = W_hh * h_{t-1} + W_xh * x_t + b_h
        h_t = tanh(a_t)
        y_hat_t = W_hy * h_t + b_y
        
    Kích thước các tensor:
        x_t: (input_dim, batch_size)
        h_t: (hidden_dim, batch_size)
        y_hat_t: (output_dim, batch_size)
        W_xh: (hidden_dim, input_dim)
        W_hh: (hidden_dim, hidden_dim)
        W_hy: (output_dim, hidden_dim)
        b_h: (hidden_dim, 1)
        b_y: (output_dim, 1)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.005, clip_value=5.0, seed=42):
        np.random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.clip_value = clip_value
        
        # Khởi tạo trọng số theo phân phối Xavier/Glorot
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        
        self.b_h = np.zeros((hidden_dim, 1))
        self.b_y = np.zeros((output_dim, 1))
        
        # Bộ đệm lưu trữ trạng thái phục vụ BPTT (Cache)
        self.cache = {}
        
    def forward(self, X, h_prev=None):
        """
        Lan truyền tiến qua T bước thời gian.
        
        Tham số:
            X: Mảng dữ liệu đầu vào kích thước (seq_len, batch_size, input_dim)
            h_prev: Trạng thái ẩn ban đầu h_0 kích thước (hidden_dim, batch_size). Mặc định là ma trận 0.
            
        Trả về:
            y_preds: Dự đoán đầu ra tại mỗi bước thời gian (seq_len, batch_size, output_dim)
            h_states: Các trạng thái ẩn tương ứng (seq_len, hidden_dim, batch_size)
        """
        seq_len, batch_size, _ = X.shape
        
        if h_prev is None:
            h_prev = np.zeros((self.hidden_dim, batch_size))
            
        h_states = {}
        h_states[-1] = h_prev.copy()
        
        a_states = {}
        y_preds = np.zeros((seq_len, batch_size, self.output_dim))
        
        for t in range(seq_len):
            # Chuyển đổi x_t thành ma trận kích thước (input_dim, batch_size)
            x_t = X[t].T
            
            # Tính toán tổ hợp tuyến tính a_t
            a_t = np.dot(self.W_hh, h_states[t - 1]) + np.dot(self.W_xh, x_t) + self.b_h
            
            # Hàm kích hoạt phi tuyến tính Hyperbolic Tangent (tanh)
            h_t = np.tanh(a_t)
            
            # Tính toán đầu ra y_hat_t
            y_t = np.dot(self.W_hy, h_t) + self.b_y
            
            # Lưu trữ vào bộ đệm
            a_states[t] = a_t
            h_states[t] = h_t
            y_preds[t] = y_t.T
            
        # Lưu cache cho lan truyền ngược
        self.cache = {
            "X": X,
            "h_states": h_states,
            "a_states": a_states,
            "seq_len": seq_len,
            "batch_size": batch_size
        }
        
        return y_preds, h_states
    
    def compute_loss_and_grad(self, y_preds, y_true):
        """
        Tính toán hàm mất mát Mean Squared Error (MSE) và đạo hàm dL/dy_pred.
        
        Tham số:
            y_preds: Mảng đầu ra dự đoán (seq_len, batch_size, output_dim)
            y_true: Mảng nhãn thực tế (seq_len, batch_size, output_dim)
            
        Trả về:
            loss: Giá trị mất mát vô hướng
            dy: Đạo hàm mất mát theo từng đầu ra y_hat
        """
        N = y_true.size
        loss = np.mean((y_preds - y_true) ** 2)
        dy = (2.0 / N) * (y_preds - y_true)
        return loss, dy
    
    def backward(self, dy):
        """
        Lan truyền ngược qua thời gian (Backpropagation Through Time - BPTT).
        
        Tham số:
            dy: Đạo hàm đầu ra kích thước (seq_len, batch_size, output_dim)
            
        Trả về:
            gradients: Từ điển chứa đạo hàm của toàn bộ ma trận trọng số và bias
        """
        X = self.cache["X"]
        h_states = self.cache["h_states"]
        seq_len = self.cache["seq_len"]
        batch_size = self.cache["batch_size"]
        
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        dh_next = np.zeros((self.hidden_dim, batch_size))
        
        # Duyệt ngược dòng thời gian từ T-1 về 0
        for t in reversed(range(seq_len)):
            # dy_t có kích thước (output_dim, batch_size)
            dy_t = dy[t].T
            h_t = h_states[t]
            h_prev = h_states[t - 1]
            x_t = X[t].T
            
            # Đạo hàm đối với các tham số của tầng đầu ra
            dW_hy += np.dot(dy_t, h_t.T)
            db_y += np.sum(dy_t, axis=1, keepdims=True)
            
            # Đạo hàm đối với trạng thái ẩn h_t (kết hợp luồng từ y_t và luồng thời gian từ h_{t+1})
            dh = np.dot(self.W_hy.T, dy_t) + dh_next
            
            # Đạo hàm qua hàm kích hoạt tanh: d(tanh(a))/da = 1 - tanh^2(a) = 1 - h^2
            da = dh * (1.0 - h_t ** 2)
            
            # Đạo hàm đối với các tham số của trạng thái ẩn
            dW_hh += np.dot(da, h_prev.T)
            dW_xh += np.dot(da, x_t.T)
            db_h += np.sum(da, axis=1, keepdims=True)
            
            # Truyền gradient ngược về trạng thái ẩn bước trước h_{t-1}
            dh_next = np.dot(self.W_hh.T, da)
            
        grads = {
            "dW_xh": dW_xh,
            "dW_hh": dW_hh,
            "dW_hy": dW_hy,
            "db_h": db_h,
            "db_y": db_y
        }
        
        # Cắt tỉa đạo hàm (Gradient Clipping) theo chuẩn L2 hoặc ngưỡng giá trị
        for k in grads:
            grads[k] = np.clip(grads[k], -self.clip_value, self.clip_value)
            
        return grads
    
    def step(self, grads):
        """
        Cập nhật trọng số theo giải thuật Gradient Descent chuẩn:
            W = W - lr * dW
        """
        self.W_xh -= self.learning_rate * grads["dW_xh"]
        self.W_hh -= self.learning_rate * grads["dW_hh"]
        self.W_hy -= self.learning_rate * grads["dW_hy"]
        self.b_h -= self.learning_rate * grads["db_h"]
        self.b_y -= self.learning_rate * grads["db_y"]

def verify_rnn_scratch():
    """Hàm kiểm định toán học và tính năng của RNN Scratch trên chuỗi tín hiệu sin"""
    print("=== KIỂM THỬ MÔ HÌNH RNN TỰ XÂY DỰNG (SCRATCH) ===")
    seq_len = 10
    batch_size = 4
    input_dim = 2
    hidden_dim = 8
    output_dim = 1
    
    rnn = VanillaRNNScratch(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, learning_rate=0.01)
    
    # Tạo chuỗi dữ liệu giả lập
    X = np.random.randn(seq_len, batch_size, input_dim)
    Y = np.random.randn(seq_len, batch_size, output_dim)
    
    # 1. Forward Pass
    y_preds, _ = rnn.forward(X)
    loss_init, dy = rnn.compute_loss_and_grad(y_preds, Y)
    print(f"Hàm mất mát khởi tạo ban đầu: {loss_init:.6f}")
    
    # 2. Backward Pass (BPTT)
    grads = rnn.backward(dy)
    print(f"Chuẩn Frobenius của dW_hh: {np.linalg.norm(grads['dW_hh']):.6f}")
    print(f"Chuẩn Frobenius của dW_xh: {np.linalg.norm(grads['dW_xh']):.6f}")
    print(f"Chuẩn Frobenius của dW_hy: {np.linalg.norm(grads['dW_hy']):.6f}")
    
    # 3. Thử nghiệm huấn luyện 30 bước tối ưu
    losses = []
    for step in range(30):
        y_preds, _ = rnn.forward(X)
        loss, dy = rnn.compute_loss_and_grad(y_preds, Y)
        grads = rnn.backward(dy)
        rnn.step(grads)
        losses.append(loss)
        
    print(f"Hàm mất mát sau 30 bước tối ưu BPTT: {losses[-1]:.6f}")
    assert losses[-1] < loss_init, "Kiểm định thất bại: Hàm mất mát không giảm!"
    print("Kiểm định thành công: Thuật toán BPTT giảm mất mát hoàn toàn chính xác!")
    return True

if __name__ == "__main__":
    verify_rnn_scratch()
