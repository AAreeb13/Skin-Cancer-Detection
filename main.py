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


class Dataset():
    def __init__(self, train_data, val_data, test_data):
        self.train = train_data
        self.validate = val_data
        self.test = test_data




class Preprocessor():
    def load_dataset(self):
        # Step 1: Download Dataset
        dataset_path = kagglehub.dataset_download("fanconic/skin-cancer-malignant-vs-benign")
        print("Downloaded Dataset Path:", dataset_path)

        # Define dataset directories
        train_dir = os.path.join(dataset_path, "train")
        test_dir = os.path.join(dataset_path, "test")
        return train_dir, test_dir

    def transform_dataset(self, train_dir, test_dir):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize 
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize([0.5], [0.5])  # Normalize 
        ])

        # Load dataset
        train_data = datasets.ImageFolder(root=train_dir, transform=transform)
        test_data = datasets.ImageFolder(root=test_dir, transform=transform)

        full_data = ConcatDataset([train_data, test_data])
        return full_data
    
    def split_dataset(self, dataset):
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size  # Ensures total size matches exactly

        train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        print(f"Total images: {total_size}")
        print(f"Training set: {train_size} images")
        print(f"Validation set: {val_size} images")
        print(f"Test set: {test_size} images")

        return Dataset(train_data, val_data, test_data)
    
        # train_loader = DataLoader(train_data, batch_size=32, shuffle=True)  # Shuffle training data
        # val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        # test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    def process(self):
        train_dir, test_dir = self.load_dataset()
        return self.split_dataset(self.transform_dataset(train_dir, test_dir))
    

if __name__ == '__main__':
    processed_dataset = Preprocessor().process()
    