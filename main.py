import numpy as np

class CNN():
    def __init__(self, input_shape, learning_rate=0.01):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.layers = []
        self.loss_function = None
        self.optimizer = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X): # one forward pass through the neural network
        pass

    def backward(self, y_true): # one backward pass through the neural network
        pass

    def train(self, X, y, epochs): # training the dataset includes 
        pass

    def predict(self, X):
        pass
