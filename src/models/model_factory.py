#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model factory module for the Continual Learning System.
Provides different neural network architectures.
"""

import logging
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """
    A simple CNN model for image classification tasks.
    """
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        """
        Initialize the model.
        
        Args:
            input_shape (tuple): Shape of input images (channels, height, width)
            num_classes (int): Number of output classes
        """
        super(SimpleCNN, self).__init__()
        
        # Extract input dimensions
        channels, height, width = input_shape
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the size of feature maps after convolutions and pooling
        feature_size = (height // 8) * (width // 8) * 128
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MLP(nn.Module):
    """
    A simple multi-layer perceptron for simpler tasks.
    """
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, hidden_dim: int = 256):
        """
        Initialize the model.
        
        Args:
            input_shape (tuple): Shape of input images (channels, height, width)
            num_classes (int): Number of output classes
            hidden_dim (int): Dimensionality of hidden layers
        """
        super(MLP, self).__init__()
        
        # Calculate input dimensionality
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.layers(x)


class LeNet5(nn.Module):
    """
    LeNet-5 convolutional network architecture for image classification.
    """
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        """
        Initialize the model.
        
        Args:
            input_shape (tuple): Shape of input images (channels, height, width)
            num_classes (int): Number of output classes
        """
        super(LeNet5, self).__init__()
        
        # Extract input dimensions
        channels, height, width = input_shape
        
        # Feature extraction
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Calculate the size of feature maps before the fully connected layer
        # For MNIST (1x28x28), after two conv+pool layers, we get 16x4x4
        # Need to adjust dynamically for different input shapes
        conv1_out_size = (height - 5 + 1) // 2  # Conv 5x5 then pool 2x2
        conv2_out_size = (conv1_out_size - 5 + 1) // 2  # Conv 5x5 then pool 2x2
        fc_input_size = 16 * conv2_out_size * conv2_out_size
        
        # Classification
        self.fc1 = nn.Linear(fc_input_size, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x


class SmallResNet(nn.Module):
    """
    A small ResNet-like architecture suitable for continual learning experiments.
    """
    class ResBlock(nn.Module):
        """Basic residual block with two 3x3 convolutions."""
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super(SmallResNet.ResBlock, self).__init__()
            
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                 stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                 stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            # Shortcut connection
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
                
        def forward(self, x):
            residual = x
            
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            
            out = self.conv2(out)
            out = self.bn2(out)
            
            out += self.shortcut(residual)
            out = self.relu(out)
            
            return out
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        """
        Initialize the model.
        
        Args:
            input_shape (tuple): Shape of input images (channels, height, width)
            num_classes (int): Number of output classes
        """
        super(SmallResNet, self).__init__()
        
        channels, height, width = input_shape
        
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # Create residual blocks
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        
        # Calculate the size after the feature extraction layers
        # Each layer with stride=2 reduces dim by 2
        final_size = height // 4 if height > 8 else 2  # Don't reduce below 2x2
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(64, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        """Create a layer of residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(self.ResBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def get_model(model_name: str, input_shape: Tuple[int, int, int], num_classes: int):
    """
    Factory function to get the specified model.
    
    Args:
        model_name (str): Name of the model architecture
        input_shape (tuple): Shape of input data (channels, height, width)
        num_classes (int): Number of output classes
        
    Returns:
        nn.Module: Instantiated model
    """
    model_name = model_name.lower()
    
    if model_name == 'simple_cnn':
        model = SimpleCNN(input_shape, num_classes)
    elif model_name == 'mlp':
        model = MLP(input_shape, num_classes)
    elif model_name == 'lenet5':
        model = LeNet5(input_shape, num_classes)
    elif model_name == 'small_resnet':
        model = SmallResNet(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    
    logger.info(f"Created {model_name} model with input shape {input_shape} and {num_classes} output classes")
    
    return model


if __name__ == "__main__":
    # Test the model factory
    logging.basicConfig(level=logging.INFO)
    
    # Test creating different models
    models = {
        'simple_cnn': get_model('simple_cnn', (1, 28, 28), 10),
        'mlp': get_model('mlp', (1, 28, 28), 10),
        'lenet5': get_model('lenet5', (1, 28, 28), 10),
        'small_resnet': get_model('small_resnet', (1, 28, 28), 10)
    }
    
    # Print model architectures
    for name, model in models.items():
        print(f"\nModel: {name}")
        print(model)
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)  # Batch of 2 MNIST-like images
        y = model(x)
        print(f"Output shape: {y.shape}") 