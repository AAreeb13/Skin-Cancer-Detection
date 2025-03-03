import numpy as np
from preprocessor import Preprocessor
import torch.optim as optim
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_shape, learning_rate=0.01):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.layers = []
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X): # one forward pass through the neural network
        pass

    def backward(self, y_true): # one backward pass through the neural network
        pass

    def train(self, X, y, epochs): # training the dataset includes 
        preprocessor = Preprocessor(batch_size=32)
        dataloaders = preprocessor.process()
        trainloader = dataloaders["train"]

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                X, y = data["train"]

                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(X)
                loss = self.loss_function(outputs, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        

    def predict(self, X):
        pass
