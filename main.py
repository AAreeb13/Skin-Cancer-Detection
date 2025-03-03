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
        testloader = dataloaders["test"]

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        

    def predict(self, X):
        pass
