import kagglehub
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset, random_split

class Dataset():
    def __init__(self, train_data, val_data, test_data):
        self.train = train_data
        self.validate = val_data
        self.test = test_data

class Preprocessor():
    def __init__(self, batch_size=32):
        self.batch_size = batch_size  # Default batch size
    
    def load_dataset(self):
        dataset_path = kagglehub.dataset_download("fanconic/skin-cancer-malignant-vs-benign")
        print("Downloaded Dataset Path:", dataset_path)

        train_dir = os.path.join(dataset_path, "train")
        test_dir = os.path.join(dataset_path, "test")
        return train_dir, test_dir

    def transform_dataset(self, train_dir, test_dir):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize for RGB
        ])

        train_data = datasets.ImageFolder(root=train_dir, transform=transform)
        test_data = datasets.ImageFolder(root=test_dir, transform=transform)

        full_data = ConcatDataset([train_data, test_data])
        return full_data

    def split_dataset(self, dataset):
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

        print(f"Total images: {total_size}")
        print(f"Training set: {train_size} images")
        print(f"Validation set: {val_size} images")
        print(f"Test set: {test_size} images")

        return Dataset(train_data, val_data, test_data)

    def get_dataloaders(self, dataset):
        return {
            "train": DataLoader(dataset.train, batch_size=self.batch_size, shuffle=True),
            "validate": DataLoader(dataset.validate, batch_size=self.batch_size, shuffle=False),
            "test": DataLoader(dataset.test, batch_size=self.batch_size, shuffle=False),
        }

    def process(self):
        train_dir, test_dir = self.load_dataset()
        dataset = self.split_dataset(self.transform_dataset(train_dir, test_dir))
        dataloaders = self.get_dataloaders(dataset)
        return dataloaders  # Returns dataloaders instead of raw dataset

if __name__ == '__main__':
    preprocessor = Preprocessor(batch_size=32)
    dataloaders = preprocessor.process()

    # Check if X is in the correct format
    for X, y in dataloaders["train"]:
        print(f"Sample batch shape: {X.shape}, Labels shape: {y.shape}")  # Expect (batch_size, 3, 224, 224)
        break
