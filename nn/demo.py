from neural_network import NeuralNetwork
from layers import Layer, FullyConnected, Dropout, ReLU, Sigmoid
from initialization import Initializations, RandomNormal, RandomUniform
from optimizer import Optimizer, SGD, Adam, RMSProp
from loss_functions import LossFunction, MSE, LogLinear, CrossEntropyLoss
from utils import create_vector, dims
import copy
import random


def main():
    num_points = 100000
    val_num_points = int(num_points * 0.2)
    test_num_points = int(num_points * 0.1)
    train_num_points = num_points - val_num_points - test_num_points

    print(f"Data split, Train: {train_num_points}, Val: {val_num_points}, Test: {test_num_points}")

    epochs = 1

    m1, m2, b = -1, 0.5, 1

    inputs, outputs = [], []

    for _ in range(num_points):
        input = [random.uniform(-1000, 1000), random.uniform(-10, 10)]
        x1, x2 = input
        output = [1 if (m1 * x1 + m2 * x2 + b) ** 2 > 75 else 0]
        inputs.append(create_vector(input))
        outputs.append(create_vector(output))

    training = (inputs[:train_num_points], outputs[:train_num_points])
    validation = (inputs[train_num_points: -test_num_points], outputs[train_num_points: -test_num_points])
    test = (inputs[-test_num_points:], outputs[-test_num_points:])

    loss_function = LogLinear()

    nn = NeuralNetwork([
        FullyConnected(2, 5, init=RandomUniform(-0.1,0.1)),
        ReLU(),
        FullyConnected(5, 8),
        Dropout(0.4),
        ReLU(),
        FullyConnected(8, 3),
        FullyConnected(3, 1, init=RandomUniform(-0.1,0.1)),
        Sigmoid()
    ])

    nn.init_params()

    pretraining_loss = nn.evaluate(*test, loss_function)
    print(f"Pretraining loss: {pretraining_loss: .2f}")
    print(nn)

    # print(nn.forward(create_vector([1, 3]), training=False), [1 if (m1 * 1 + m2 * 3 + b) ** 2 > 75 else 0])
    # print(nn.forward(create_vector([1, 2]), training=False), [1 if (m1 * 1 + m2 * 2 + b) ** 2 > 75 else 0])
    # print(nn.forward(create_vector([2, 2]), training=False), [1 if (m1 * 2 + m2 * 2 + b) ** 2 > 75 else 0])

    best_nn = None
    best_val_loss = float("inf")

    for iter in range(epochs):
        nn.train(*training, learning_rate=0.005, loss_function=loss_function, weight_decay=0.1)
        loss = nn.evaluate(*validation, loss_function)

        if loss < best_val_loss:
            best_nn = copy.deepcopy(nn)
            best_val_loss = loss

        print(f"Epoch {iter} with validation loss of {loss:.2f}")
    posttraining_loss = best_nn.evaluate(*test, loss_function)
    print(f"Pretraining loss: {posttraining_loss: .2f}")

    # print("Before")
    # print(nn)

    # print("")

    # print("After")
    # print(nn)

    print(nn)

    # print(nn.forward(create_vector([1, 3]), training=False), [1 if (m1 * 1 + m2 * 3 + b) ** 2 > 75 else 0])
    # print(nn.forward(create_vector([1, 2]), training=False), [1 if (m1 * 1 + m2 * 2 + b) ** 2 > 75 else 0])
    # print(nn.forward(create_vector([2, 2]), training=False), [1 if (m1 * 2 + m2 * 2 + b) ** 2 > 75 else 0])

if __name__ == "__main__":
    main()