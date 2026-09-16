"""
Module PyTorch CNN Model
Xây dựng mô hình CNN bằng PyTorch theo triết lý Function Composition.
Hỗ trợ kiểm tra tensor shapes trung gian (Inspecting Shapes) và đối chiếu tham số (Parameter Counting).
"""

import torch
import torch.nn as nn

class PyTorchCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, spatial_size=28):
        super(PyTorchCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Block 1: Conv -> BN -> ReLU -> MaxPool
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: Conv -> BN -> ReLU -> MaxPool
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Kích thước không gian sau 2 lần MaxPool (/ 4)
        reduced_size = spatial_size // 4
        flatten_dim = 64 * reduced_size * reduced_size
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # f = f_classifier o f_block2 o f_block1
        x = self.block1(x)
        x = self.block2(x)
        logits = self.classifier(x)
        return logits

    def inspect_shapes(self, x):
        """
        In kích thước tensor qua từng khối biến đổi (Thực nghiệm mục 42 trong bài giảng).
        """
        shapes = []
        shapes.append(('Input', list(x.shape)))
        
        x1 = self.block1(x)
        shapes.append(('After Block 1 (Conv-BN-ReLU-Pool)', list(x1.shape)))
        
        x2 = self.block2(x1)
        shapes.append(('After Block 2 (Conv-BN-ReLU-Pool)', list(x2.shape)))
        
        out = self.classifier(x2)
        shapes.append(('Output Logits', list(out.shape)))
        return shapes

    def count_parameters_detailed(self):
        """
        Tính số tham số thủ công qua công thức giải tích và đối chiếu với PyTorch numel()
        (Thực nghiệm mục 43 trong bài giảng).
        """
        details = []
        total_pytorch = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                p_count = param.numel()
                total_pytorch += p_count
                details.append((name, list(param.shape), p_count))
                
        return details, total_pytorch
