from nn.neural_network import NeuralNetwork
from nn.layers import Layer, FullyConnected, Dropout, ReLU, Sigmoid
from nn.initialization import Initialization, RandomNormal, RandomUniform
from nn.optimizer import Optimizer, SGD, Adam, RMSProp
from nn.loss_functions import LossFunction, MSE, LogLinear, CrossEntropyLoss
from nn.metrics import MSEMetric, MAEMetric
from nn.utils import create_vector, dims
import copy
import random


def main():
    # random.seed(1)

    num_points = 1000
    val_num_points = int(num_points * 0.2)
    test_num_points = int(num_points * 0.1)
    train_num_points = num_points - val_num_points - test_num_points

    print(f"Data split, Train: {train_num_points}, Val: {val_num_points}, Test: {test_num_points}")

    epochs = 500
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 0.01

    m1, m2, b = -1, 0.5, 1

    X, Y = [], []

    for _ in range(num_points):
        input = [random.uniform(-10, 10), random.uniform(-10, 10)]
        x1, x2 = input
        output = [m1 * x1 + m2 * x2 + b ** 2, m1 * x1 + m2 * x2]
        X.append(input)
        Y.append(output)

    training = (X[:train_num_points], Y[:train_num_points])
    validation = (X[train_num_points: -test_num_points], Y[train_num_points: -test_num_points])
    test = (X[-test_num_points:], Y[-test_num_points:])

    loss_function = MSE()
    optimizer = SGD(learning_rate=learning_rate)
    metric = MSEMetric()

    nn = NeuralNetwork([
        FullyConnected(2, 2)  # direct 2→2 mapping
    ])


    nn.init_params()

    pretraining_loss = nn.evaluate(*test, metric)
    print(f"Pre training loss: {pretraining_loss: .2f}")
    print("Pre training neural network: ")
    # print(nn)

    print("Some datapoint prediction pre training")
    print(nn.forward([[1, 3]], training=False), [m1 * 1 + m2 * 3 + b ** 2, m1 * 1 + m2 * 3])
    print(nn.forward([[1, 2]], training=False), [m1 * 1 + m2 * 2 + b ** 2, m1 * 1 + m2 * 2])
    print(nn.forward([[2, 2]], training=False), [m1 * 2 + m2 * 2 + b ** 2, m1 * 2 + m2 * 2])

    best_nn = None
    best_epoch = None
    best_val_loss = float("inf")

    for iter in range(epochs):
        nn.train(*training, loss_function, optimizer)
        # print(nn)
        loss = nn.evaluate(*validation, metric)

        if loss < best_val_loss:
            best_nn = copy.deepcopy(nn)
            best_epoch = iter
            best_val_loss = loss

        print(f"Epoch {iter} with validation loss of {loss:.2f}")
    posttraining_loss = best_nn.evaluate(*test, metric)
    print(f"Best validation loss (on epoch {best_epoch}): {best_val_loss:.2f}")
    print(f"Post training on test loss: {posttraining_loss:.2f}")

    # print("Post training neural network: ")
    # print(nn)

    print("Some datapoint prediction post training")
    print(best_nn.forward([[1, 3]], training=False), [m1 * 1 + m2 * 3 + b ** 2, m1 * 1 + m2 * 3])
    print(best_nn.forward([[1, 2]], training=False), [m1 * 1 + m2 * 2 + b ** 2, m1 * 1 + m2 * 2])
    print(best_nn.forward([[2, 2]], training=False), [m1 * 2 + m2 * 2 + b ** 2, m1 * 2 + m2 * 2])

if __name__ == "__main__":
    main()