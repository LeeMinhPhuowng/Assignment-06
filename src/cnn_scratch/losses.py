"""
Module Losses and Optimizers for NumPy CNN
"""

import numpy as np

class SoftmaxCrossEntropyLoss:
    """
    Hàm mất mát Softmax Cross-Entropy ổn định số học.
    Đạo hàm giải tích: dL/dlogits = (probs - y) / batch_size
    """
    def __init__(self):
        self.probs = None
        self.y_one_hot = None
        self.batch_size = 0

    def forward(self, logits, y_true):
        self.batch_size = logits.shape[0]
        # Trừ max để chống overflow
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        if y_true.ndim == 1:
            self.y_one_hot = np.zeros_like(logits)
            self.y_one_hot[np.arange(self.batch_size), y_true] = 1.0
        else:
            self.y_one_hot = y_true

        # Thêm epsilon chống log(0)
        loss = -np.sum(self.y_one_hot * np.log(self.probs + 1e-12)) / self.batch_size
        return loss

    def backward(self):
        return (self.probs - self.y_one_hot) / self.batch_size


class AdamOptimizer:
    """
    Thuật toán tối ưu hóa Adam (Adaptive Moment Estimation).
    Cập nhật trọng số theo moment bậc 1 (m) và moment bậc 2 (v).
    """
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for idx, layer in enumerate(layers):
            if hasattr(layer, 'W'):
                if idx not in self.m:
                    self.m[idx] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                    self.v[idx] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}

                # Update W
                self.m[idx]['W'] = self.beta1 * self.m[idx]['W'] + (1 - self.beta1) * layer.dW
                self.v[idx]['W'] = self.beta2 * self.v[idx]['W'] + (1 - self.beta2) * (layer.dW ** 2)
                m_hat_w = self.m[idx]['W'] / (1 - self.beta1 ** self.t)
                v_hat_w = self.v[idx]['W'] / (1 - self.beta2 ** self.t)
                layer.W -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)

                # Update b
                self.m[idx]['b'] = self.beta1 * self.m[idx]['b'] + (1 - self.beta1) * layer.db
                self.v[idx]['b'] = self.beta2 * self.v[idx]['b'] + (1 - self.beta2) * (layer.db ** 2)
                m_hat_b = self.m[idx]['b'] / (1 - self.beta1 ** self.t)
                v_hat_b = self.v[idx]['b'] / (1 - self.beta2 ** self.t)
                layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
