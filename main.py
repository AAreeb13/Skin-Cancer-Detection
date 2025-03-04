import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessor import Preprocessor
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # Correct way
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2) # do i need more than 1?
        self.linear = nn.Linear(16 * 111 * 111, 1)  # Corrected input size

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxPool(x)  # results in shape: (16, 112, 112)
        x = x.view(x.size(0), -1)  # shape: (batch_size, 16*112*112)
        x = self.linear(x)             # shape: (batch_size, 1)
        return x

    def predict(self, X):
        self.eval()  # Switch to evaluation mode
        with torch.no_grad():
            outputs = torch.sigmoid(self.forward(X)).squeeze(1)
            return (outputs > 0.5).int()  # Convert to binary (0 or 1)

if __name__ == '__main__':
    print('Starting training')
    model = CNN()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    
    
    preprocessor = Preprocessor()
    dataloaders = preprocessor.process()

    optimizer=torch.optim.Adam(model.parameters())
    bce = nn.BCEWithLogitsLoss()

    model.to(device)
    epochs = 5

    # ----------------TRAIN STARTS HERE----------------
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        total_loss = 0.0
        for inputs, labels in dataloaders['train']:
        # Move data to GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = bce(outputs, labels.float().unsqueeze(1))  # Fix: add .unsqueeze(1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
        epoch_loss = total_loss / len(dataloaders['train'].dataset)
        print(f"Training Loss: {epoch_loss:.4f}")


    torch.save(model.state_dict(), 'skin_cancer_model_state.pth')
    print("Model parameters saved as skin_cancer_model_state.pth")

    # ----------------TRAIN ENDS HERE----------------
