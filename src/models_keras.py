"""
Module Keras / TensorFlow CNN Model
Định nghĩa kiến trúc mạng CNN chuẩn mực theo phong cách Keras Functional/Sequential API.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_keras_cnn(input_shape=(28, 28, 1), num_classes=10, dropout_rate=0.25):
    """
    Xây dựng mô hình CNN hoàn chỉnh với Keras.
    Kiến trúc gồm:
    - Block 1: Conv2D(32, 3x3, same) -> BatchNorm -> ReLU -> MaxPool2D(2x2)
    - Block 2: Conv2D(64, 3x3, same) -> BatchNorm -> ReLU -> MaxPool2D(2x2)
    - Classifier: Flatten -> Dense(128) -> ReLU -> Dropout -> Dense(num_classes, softmax)
    """
    model = models.Sequential([
        # Input
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False, name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.ReLU(name='relu1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),

        # Block 2
        layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False, name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.ReLU(name='relu2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),

        # Classifier Head
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='fc1'),
        layers.Dropout(dropout_rate, name='dropout'),
        layers.Dense(num_classes, activation='softmax', name='output_softmax')
    ], name='Keras_CNN_Classifier')

    return model
