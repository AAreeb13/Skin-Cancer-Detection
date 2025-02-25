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

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, X):
        pass

    def backward(self, grad_output):
        pass

class ConvLayer(Layer):
    def __init__(self, num_filters, kernel_size, stride=1, padding=0):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = None  # To be initialized
        self.biases = None  # To be initialized

    def forward(self, X):
        pass

    def backward(self, grad_output):
        pass

class PoolingLayer(Layer):
    def __init__(self, pool_size, stride, mode="max"):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode

    def forward(self, X):
        pass

    def backward(self, grad_output):
        pass

class FullyConnectedLayer(Layer):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = None  # To be initialized
        self.biases = None  # To be initialized

    def forward(self, X):
        pass

    def backward(self, grad_output):
        pass

class ActivationLayer(Layer):
    def __init__(self, activation_function):
        super().__init__()
        self.activation_function = activation_function

    def forward(self, X):
        pass

    def backward(self, grad_output):
        pass

class LossFunction:
    def compute_loss(self, y_pred, y_true):
        pass

    def compute_gradient(self, y_pred, y_true):
        pass

class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, parameters, gradients):
        pass
