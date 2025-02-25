import numpy as np

class CNN:
    def __init__(self, input_shape, learning_rate=0.01):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.layers = []
        self.loss_function = None
        self.optimizer = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        pass

    def backward(self, y_true):
        pass

    def update_parameters(self):
        pass

    def train(self, X, y, epochs):
        pass

    def predict(self, X):
        pass
