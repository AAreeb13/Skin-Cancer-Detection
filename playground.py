import kagglehub
import os
import torch
import torch.nn as nn
# import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset
# from sklearn.metrics import f1_score, accuracy_score, recall_score
import numpy as np

from preprocessor import Preprocessor
# Initialize Preprocessor and load dataset
preprocessor = Preprocessor()
dataset = preprocessor.process()

# Create DataLoader
train_loader = DataLoader(dataset.train, batch_size=4, shuffle=True)

# Get one batch of images
images, labels = next(iter(train_loader))
print("Input Shape:", images.shape)  # Expected: (4, 3, 224, 224)

# Define Convolutional Layer (reduces size to 112x112)
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
relu = nn.ReLU()  # ReLU Activation Function

# Forward pass through convolutional layer
conv_output = conv_layer(images)
print("After Conv Shape:", conv_output.shape)  # Expected: (4, 16, 112, 112)

# Apply ReLU activation
relu_output = relu(conv_output)
print("After ReLU Shape:", relu_output.shape)  # Should be the same as after Conv



