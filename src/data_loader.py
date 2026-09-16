"""
Module Data Loader & Preprocessing
Quản lý nạp 3 bộ dữ liệu:
- Dataset 1 (Ảnh 1): MNIST (Chữ số viết tay 28x28)
- Dataset 2 (Ảnh 2): Fashion-MNIST (Quần áo, giày dép thời trang 28x28)
- Dataset 3 (Bộ dữ liệu số học / Bảng): UCI Digits (Ảnh dạng bảng 8x8 flattened 64 đặc trưng)
Tuân thủ nghiêm ngặt nguyên tắc No Data Leakage.
"""

import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_all_three_datasets(data_dir='./data', batch_size=64, random_seed=42):
    """
    Nạp đồng thời 3 bộ dữ liệu:
    1. MNIST (TensorFlow / PyTorch / NumPy)
    2. Fashion-MNIST (PyTorch / TensorFlow)
    3. Digits Tabular (Scikit-Learn / NumPy)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Pipeline cho MNIST
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 2. Pipeline cho Fashion-MNIST
    transform_fmnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    # Tải cục bộ từ folder data/ (đã tải sẵn, không cần download lại)
    mnist_train_full = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform_mnist)
    mnist_test = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform_mnist)
    
    fmnist_train_full = datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=transform_fmnist)
    fmnist_test = datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=transform_fmnist)

    # Chia train/val 90/10 chống data leakage
    mnist_train, mnist_val = random_split(
        mnist_train_full, [54000, 6000],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    fmnist_train, fmnist_val = random_split(
        fmnist_train_full, [54000, 6000],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # DataLoaders
    mnist_loaders = {
        'train': DataLoader(mnist_train, batch_size=batch_size, shuffle=True),
        'val': DataLoader(mnist_val, batch_size=batch_size, shuffle=False),
        'test': DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    }

    fmnist_loaders = {
        'train': DataLoader(fmnist_train, batch_size=batch_size, shuffle=True),
        'val': DataLoader(fmnist_val, batch_size=batch_size, shuffle=False),
        'test': DataLoader(fmnist_test, batch_size=batch_size, shuffle=False)
    }

    # 3. Dataset 3: Dữ liệu Bảng / Vector (UCI Digits 8x8 = 64 features)
    digits = load_digits()
    X_digits = digits.data.astype(np.float32)
    y_digits = digits.target.astype(np.int64)

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_digits, y_digits, test_size=0.2, random_state=random_seed, stratify=y_digits
    )
    scaler = StandardScaler()
    X_train_d = scaler.fit_transform(X_train_d)
    X_test_d = scaler.transform(X_test_d) # Fit train, transform test strictly

    digits_data = {
        'X_train': X_train_d, 'y_train': y_train_d,
        'X_test': X_test_d, 'y_test': y_test_d,
        'feature_dim': 64, 'num_classes': 10
    }

    return {
        'mnist': mnist_loaders,
        'fmnist': fmnist_loaders,
        'digits': digits_data,
        'raw_mnist': (mnist_train_full, mnist_test),
        'raw_fmnist': (fmnist_train_full, fmnist_test)
    }
