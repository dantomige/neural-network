from loss_functions import LossFunction, MSE, LogLinear
from layers import Layer, FullyConnected, Dropout, ReLU, Sigmoid
from initialization import Initializations, RandomNormal, RandomUniform
from optimizer import Optimizer, SGD, Adam, RMSProp
from metrics import Metrics
from utils import create_vector, dims
import random
import copy

class NeuralNetwork:

    def __init__(self, layers: list[Layer]): 

        self.layers = layers
        self.backprop = None
    
    def init_params(self):
        for layer in self.layers:
            layer.init_params()

    def forward(self, input, training=True):
        for layer in self.layers:
            output = layer.forward(input, training)
            input = output
        return output
    
    def backward(self, input, model_output, expected_output, loss_function): 
        
        _ = loss_function.loss(model_output, expected_output)
        pL_pOut = loss_function.backward()

        for layer in reversed(self.layers):
            pL_pIn = layer.backward(pL_pOut)
            pL_pOut = pL_pIn

    def update_params(self, optimizer):
        for layer in self.layers:
            layer.update_params(optimizer)

    def train(self, inputs, outputs, loss_function: LossFunction, optimizer: Optimizer):
        
        for input, output in zip(inputs, outputs):
            model_output = self.forward(input)
            self.backward(input, model_output, output, loss_function)
            self.update_params(optimizer)

    def evaluate(self, inputs, outputs, metric: Metrics):
        assert len(inputs) == len(outputs)

        total_loss = 0
        num_datapoints = len(inputs)

        for input, output in zip(inputs, outputs):
            model_output = self.forward(input, training=False)
            datapoint_loss = metric.loss(model_output, output)
            total_loss += datapoint_loss
        
        return total_loss/num_datapoints
    
    def __str__(self):     
        output = ""
        for layer in self.layers:
            output += str(layer)
            output += "\n"
        return output[:-1]
        
            

if __name__ == "__main__":
    pass


