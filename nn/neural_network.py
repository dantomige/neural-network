from loss_functions import LossFunction, MSE, LogLinear
from layers import Layer, FullyConnected, Dropout, ReLU, Sigmoid
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
    
    def backward(self, input, model_output, expected_output, loss_function, weight_decay): 
        
        _ = loss_function.loss(model_output, expected_output)
        pL_pOut = loss_function.backward()

        for layer in reversed(self.layers):
            if layer.is_trainable:
                pL_pIn = layer.backward(pL_pOut, weight_decay)
            else:
                pL_pIn = layer.backward(pL_pOut)
            pL_pOut = pL_pIn

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.update_params(learning_rate)

    def train(self, inputs, outputs, learning_rate: int, loss_function: LossFunction, weight_decay=0):

        # print(inputs[0], outputs[0])
        
        for input, output in zip(inputs, outputs):
            model_output = self.forward(input)
            self.backward(input, model_output, output, loss_function, weight_decay)
            self.update_parameters(learning_rate)

    def evaluate(self, inputs, outputs, loss_function: LossFunction):
        assert len(inputs) == len(outputs)

        total_loss = 0
        num_datapoints = len(inputs)

        for input, output in zip(inputs, outputs):
            model_output = self.forward(input, training=False)
            datapoint_loss = loss_function.loss(model_output, output)
            total_loss += datapoint_loss
        
        return total_loss/num_datapoints
    
    def __str__(self):     
        output = ""
        for layer in self.layers:
            output += str(layer)
            output += "\n"
        return output[:-1]
        
            

if __name__ == "__main__":

    num_points = 100000
    val_num_points = int(num_points * 0.2)
    test_num_points = int(num_points * 0.1)
    train_num_points = num_points - val_num_points - test_num_points

    print(f"Data split, Train: {train_num_points}, Val: {val_num_points}, Test: {test_num_points}")

    epochs = 10

    m1, m2, b = -1, 0.5, 1

    inputs, outputs = [], []

    for _ in range(num_points):
        input = [random.uniform(-3, 3), random.uniform(-0.5, 0.5)]
        x1, x2 = input
        output = [1 if (m1 * x1 + m2 * x2 + b) ** 2 > 2 else 0]
        inputs.append(create_vector(input))
        outputs.append(create_vector(output))
 
    training = (inputs[:train_num_points], outputs[:train_num_points])
    validation = (inputs[train_num_points: -test_num_points], outputs[train_num_points: -test_num_points])
    test = (inputs[-test_num_points:], outputs[-test_num_points:])

    loss_function = LogLinear()

    nn = NeuralNetwork([
        FullyConnected(2, 7),
        Dropout(0.25),
        ReLU(),
        FullyConnected(7, 1),
        Sigmoid()
    ])

    nn.init_params()

    pretraining_loss = nn.evaluate(*test, loss_function)
    print(f"Pretraining loss: {pretraining_loss: .2f}")

    best_nn = None
    best_val_loss = float("inf")

    for iter in range(epochs):
        nn.train(*training, learning_rate=0.04, loss_function=loss_function, weight_decay=0.00001)
        loss = nn.evaluate(*validation, loss_function)

        if loss < best_val_loss:
            best_nn = copy.deepcopy(nn)
            best_val_loss = loss

        print(f"Epoch {iter} with validation loss of {loss:.2f}")
    posttraining_loss = best_nn.evaluate(*test, loss_function)
    print(f"Pretraining loss: {posttraining_loss: .2f}")

    # print("Before")
    # print(nn)

    # print(nn.forward(create_vector([1, 3]), training=False), [1 if (m1 * 1 + m2 * 3 + b) ** 2 > 2 else 0])
    # print(nn.forward(create_vector([1, 2]), training=False), [1 if (m1 * 1 + m2 * 2 + b) ** 2 > 2 else 0])
    # print(nn.forward(create_vector([2, 2]), training=False), [1 if (m1 * 2 + m2 * 2 + b) ** 2 > 2 else 0])
    # print("")

    # print("After")
    # print(nn)

    # print(nn.forward(create_vector([1, 3]), training=False), [1 if (m1 * 1 + m2 * 3 + b) ** 2 > 2 else 0])
    # print(nn.forward(create_vector([1, 2]), training=False), [1 if (m1 * 1 + m2 * 2 + b) ** 2 > 2 else 0])
    # print(nn.forward(create_vector([2, 2]), training=False), [1 if (m1 * 2 + m2 * 2 + b) ** 2 > 2 else 0])


