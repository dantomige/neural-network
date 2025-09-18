from loss_functions import MSE
from layers import Layer, FullyConnected, ReLU
from utils import create_vector
import random

class NeuralNetwork:

    def __init__(self, layers: list[Layer]): 

        self.layers = layers
        self.backprop = None
    
    def init_params(self):
        for layer in self.layers:
            layer.init_params()

    def forward(self, input):
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output
    
    def backward(self, input, model_output, expected_output, loss_function): 
        
        _ = loss_function.loss(model_output, expected_output)
        pL_pOut = loss_function.backward()

        for layer in reversed(self.layers):
            layer.backward(pL_pOut)
            pL_pOut = layer.pL_pIn

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.update_params(learning_rate)

    def train(self, inputs, outputs, learning_rate: int, loss_function: MSE):
        
        for input, output in zip(inputs, outputs):
            model_output = self.forward(input)
            self.backward(input, model_output, output, loss_function)
            self.update_parameters(learning_rate)
    
    def __str__(self):     
        output = ""
        for layer in self.layers:
            output += str(layer)
            output += "\n"
        return output[:-1]
        
            

if __name__ == "__main__":

    num_points = 1000
    m1, m2, b = -1, 0.5, 1

    inputs, outputs = [], []

    for _ in range(num_points):
        input = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        x1, x2 = input
        output = [m1 * x1 + m2 * x2 + b, 1/(m1 * x1 + m2 * x2 + b)]
        inputs.append(create_vector(input))
        outputs.append(create_vector(output))

    loss_function = MSE()

    nn = NeuralNetwork([
        FullyConnected(2, 5),
        ReLU(),
        FullyConnected(5, 2)
    ])

    nn.init_params()

    print("Before")
    print(nn)

    print(nn.forward(create_vector([1, 3]))) # expect around 1.5
    print(nn.forward(create_vector([1, 2]))) # expect around 1
    print(nn.forward(create_vector([2, 2]))) # expect around 0
    print("")
    
    nn.train(inputs, outputs, learning_rate=0.1, loss_function=loss_function)

    print("After")
    print(nn)

    print(nn.forward(create_vector([1, 3]))) # expect around 1.5, 0.667
    print(nn.forward(create_vector([1, 2]))) # expect around 1, 1
    print(nn.forward(create_vector([2, 2]))) # expect around 0, undefined


