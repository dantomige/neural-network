from .loss_functions import LossFunction, MSE, LogLinear
from .layers import Layer, FullyConnected, Dropout, ReLU, Sigmoid
from .initialization import Initialization, RandomNormal, RandomUniform
from .optimizer import Optimizer, SGD, Adam, RMSProp
from .metrics import Metrics
from .utils import create_vector, dims
import random
import copy

class NeuralNetwork:

    def __init__(self, layers: list[Layer]): 

        self.layers = layers
        self.backprop = None
    
    def init_params(self):
        for layer in self.layers:
            layer.init_params()

    def forward(self, X, training=True):
        for layer in self.layers:
            output = layer.forward(X, training)
            X = output
        return output
    
    def backward(self, X, Y, Yhat, loss_function: LossFunction): 
        
        _ = loss_function.loss(Y, Yhat)
        pL_pOut = loss_function.backward()

        for layer in reversed(self.layers):
            pL_pIn = layer.backward(pL_pOut)
            pL_pOut = pL_pIn

    def update_params(self, optimizer):
        for layer in self.layers:
            layer.update_params(optimizer)

    def train(self, X, Y, loss_function: LossFunction, optimizer: Optimizer):
        
        Yhat = self.forward(X)
        self.backward(X, Y, Yhat, loss_function)
        self.update_params(optimizer)

    def evaluate(self, X, Y, metric: Metrics):
        assert len(X) == len(Y)

        Yhat = self.forward(X, training=False)
        output = metric.evaluate(Y, Yhat)
        
        return output
    
    def __call__(self, X):
        return self.forward(X, training=False)

    def __str__(self):     
        output = ""
        for layer in self.layers:
            output += str(layer)
            output += "\n"
        return output[:-1]
        
            

if __name__ == "__main__":
    pass


