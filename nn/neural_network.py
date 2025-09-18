from loss_functions import MSE, LogLinear
from layers import Layer, FullyConnected, Dropout, ReLU, Sigmoid
from utils import create_vector
import random

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

    num_points = 100000
    m1, m2, b = -1, 0.5, 1

    inputs, outputs = [], []

    for _ in range(num_points):
        input = [random.uniform(-3, 3), random.uniform(-0.5, 0.5)]
        x1, x2 = input
        output = [1 if (m1 * x1 + m2 * x2 + b) ** 2 > 2 else 0]
        inputs.append(create_vector(input))
        outputs.append(create_vector(output))

    loss_function = LogLinear()

    nn = NeuralNetwork([
        FullyConnected(2, 7),
        Dropout(0.25),
        ReLU(),
        FullyConnected(7, 1),
        Sigmoid()
    ])

    nn.init_params()

    print("Before")
    print(nn)

    print(nn.forward(create_vector([1, 3]), training=False), [1 if (m1 * 1 + m2 * 3 + b) ** 2 > 2 else 0])
    print(nn.forward(create_vector([1, 2]), training=False), [1 if (m1 * 1 + m2 * 2 + b) ** 2 > 2 else 0])
    print(nn.forward(create_vector([2, 2]), training=False), [1 if (m1 * 2 + m2 * 2 + b) ** 2 > 2 else 0])
    print("")
    
    nn.train(inputs, outputs, learning_rate=0.05, loss_function=loss_function)

    print("After")
    print(nn)

    print(nn.forward(create_vector([1, 3]), training=False), [1 if (m1 * 1 + m2 * 3 + b) ** 2 > 2 else 0])
    print(nn.forward(create_vector([1, 2]), training=False), [1 if (m1 * 1 + m2 * 2 + b) ** 2 > 2 else 0])
    print(nn.forward(create_vector([2, 2]), training=False), [1 if (m1 * 2 + m2 * 2 + b) ** 2 > 2 else 0])


