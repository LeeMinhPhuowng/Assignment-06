"""
Module CNN Scratch Layers (NumPy)
Cài đặt các tầng mạng nơ-ron tích chập từ đầu (from scratch) bằng NumPy.
Sử dụng thuật toán vector hóa im2col và col2im để đạt hiệu năng tính toán cao.
"""

import numpy as np

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """
    Chuyển đổi tensor ảnh 4D (B, C, H, W) thành ma trận 2D (C*Kh*Kw, B*H_out*W_out).
    Biến phép tích chập không gian thành phép nhân ma trận chuẩn GEMM.
    """
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # Lấy kích thước
    N, C, H, W = x_shape
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return (k.astype(int), i.astype(int), j.astype(int))

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """
    Phép biến đổi ngược từ ma trận gradient 2D về tensor gradient ảnh 4D.
    Tích lũy gradient tại các vùng receptive field bị chồng lấn (overlap).
    """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Conv2D_Scratch:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Khởi tạo Kaiming/He Normal
        limit = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * limit
        self.b = np.zeros((out_channels, 1))
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None
        self.x_col = None

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        out_h = int((H + 2 * self.padding - self.kernel_size) / self.stride + 1)
        out_w = int((W + 2 * self.padding - self.kernel_size) / self.stride + 1)
        
        # im2col
        self.x_col = im2col_indices(x, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)
        W_row = self.W.reshape(self.out_channels, -1)
        
        out = np.dot(W_row, self.x_col) + self.b
        out = out.reshape(self.out_channels, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        # dout shape: (N, C_out, H_out, W_out)
        N, C, H, W = self.x.shape
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        
        # dW = dout @ x_col.T
        self.dW = np.dot(dout_reshaped, self.x_col.T).reshape(self.W.shape)
        # db = sum(dout)
        self.db = np.sum(dout_reshaped, axis=1, keepdims=True)
        
        # dX_col = W.T @ dout
        W_row = self.W.reshape(self.out_channels, -1)
        dx_col = np.dot(W_row.T, dout_reshaped)
        dx = col2im_indices(dx_col, self.x.shape, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)
        return dx


class MaxPool2D_Scratch:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride
        self.x = None
        self.x_col = None
        self.max_idx = None

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        h_out = int((H - self.size) / self.stride + 1)
        w_out = int((W - self.size) / self.stride + 1)
        
        x_reshaped = x.reshape(N * C, 1, H, W)
        self.x_col = im2col_indices(x_reshaped, self.size, self.size, padding=0, stride=self.stride)
        
        self.max_idx = np.argmax(self.x_col, axis=0)
        out = self.x_col[self.max_idx, np.arange(self.max_idx.size)]
        out = out.reshape(h_out, w_out, N, C).transpose(2, 3, 0, 1)
        return out

    def backward(self, dout):
        N, C, H, W = self.x.shape
        dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
        dx_col = np.zeros_like(self.x_col)
        dx_col[self.max_idx, np.arange(self.max_idx.size)] = dout_reshaped
        
        dx = col2im_indices(dx_col, (N * C, 1, H, W), self.size, self.size, padding=0, stride=self.stride)
        return dx.reshape(N, C, H, W)


class ReLU_Scratch:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        return dout * (self.x > 0)


class Flatten_Scratch:
    def __init__(self):
        self.orig_shape = None

    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.orig_shape)


class Dense_Scratch:
    def __init__(self, in_features, out_features):
        limit = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * limit
        self.b = np.zeros((1, out_features))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        dx = np.dot(dout, self.W.T)
        return dx
