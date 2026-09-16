"""
Module CNN Model from Scratch
Xây dựng mô hình CNN hoàn chỉnh từ các lớp NumPy theo cấu trúc Function Composition:
f = f_5 o f_4 o f_3 o f_2 o f_1
"""

import numpy as np
import time
from .layers import Conv2D_Scratch, ReLU_Scratch, MaxPool2D_Scratch, Flatten_Scratch, Dense_Scratch
from .losses import SoftmaxCrossEntropyLoss, AdamOptimizer

class CNN_Scratch:
    def __init__(self, in_channels=1, num_classes=10):
        # Kiến trúc chuẩn hóa theo bài tập 44 trong bài giảng:
        # Conv(3x3) -> ReLU -> MaxPool(2x2) -> Conv(3x3) -> ReLU -> MaxPool(2x2) -> Flatten -> Dense -> Dense
        self.layers = [
            Conv2D_Scratch(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1),
            ReLU_Scratch(),
            MaxPool2D_Scratch(size=2, stride=2),
            Conv2D_Scratch(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            ReLU_Scratch(),
            MaxPool2D_Scratch(size=2, stride=2),
            Flatten_Scratch()
        ]
        self.fc_initialized = False
        self.num_classes = num_classes

    def _init_fc(self, flattened_dim):
        self.layers.extend([
            Dense_Scratch(in_features=flattened_dim, out_features=64),
            ReLU_Scratch(),
            Dense_Scratch(in_features=64, out_features=self.num_classes)
        ])
        self.fc_initialized = True

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Flatten_Scratch):
                out = layer.forward(out)
                if not self.fc_initialized:
                    self._init_fc(out.shape[1])
            else:
                out = layer.forward(out)
        return out

    def backward(self, dout):
        grad = dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def count_parameters(self):
        total = 0
        for layer in self.layers:
            if hasattr(layer, 'W'):
                total += layer.W.size + layer.b.size
        return total

    def fit(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=64, lr=1e-3):
        criterion = SoftmaxCrossEntropyLoss()
        optimizer = AdamOptimizer(lr=lr)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        start_time = time.perf_counter()

        # Khởi tạo FC trước nếu cần
        if not self.fc_initialized:
            _ = self.forward(X_train[:2])

        n_samples = len(X_train)
        print(f"Bắt đầu huấn luyện NumPy CNN ({epochs} epochs, batch_size={batch_size})...")

        for epoch in range(epochs):
            perm = np.random.permutation(n_samples)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]
            
            epoch_loss = 0.0
            num_batches = 0

            for b in range(0, n_samples, batch_size):
                xb = X_shuffled[b:b+batch_size]
                yb = y_shuffled[b:b+batch_size]

                # Forward
                logits = self.forward(xb)
                loss = criterion.forward(logits, yb)
                epoch_loss += loss

                # Backward
                d_logits = criterion.backward()
                self.backward(d_logits)

                # Step
                optimizer.step(self.layers)
                num_batches += 1

            train_loss = epoch_loss / num_batches
            
            # Validation
            val_logits = self.forward(X_val)
            val_loss = criterion.forward(val_logits, y_val)
            val_preds = np.argmax(val_logits, axis=1)
            val_acc = np.mean(val_preds == y_val)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%")

        total_time = time.perf_counter() - start_time
        return history, total_time

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)
