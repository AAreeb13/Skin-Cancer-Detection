import kagglehub
import os
import torch
# import torch.nn as nn
# import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset
# from sklearn.metrics import f1_score, accuracy_score, recall_score
import numpy as np

# Step 1: Download Dataset
dataset_path = kagglehub.dataset_download("fanconic/skin-cancer-malignant-vs-benign")
print("Downloaded Dataset Path:", dataset_path)

# Define dataset directories
train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")

# Step 2: Data Preprocessing & Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize 
    transforms.ToTensor(),  # Convert 
    transforms.Normalize([0.5], [0.5])  # Normalize 
])

# Step 3: Load Dataset
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_dir, transform=transform)

full_data = ConcatDataset([train_data, test_data])


total_size = len(full_data)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size  # Ensures total size matches exactly

train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size])

print(f"Total images: {total_size}")
print(f"Training set: {train_size} images")
print(f"Validation set: {val_size} images")
print(f"Test set: {test_size} images")

# Step 4: Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)  # Shuffle training data
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)



