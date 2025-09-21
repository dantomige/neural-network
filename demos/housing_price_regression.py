from nn.neural_network import NeuralNetwork
from nn.layers import Layer, FullyConnected, Dropout, ReLU, Sigmoid
from nn.initialization import Initialization, RandomNormal, RandomUniform
from nn.optimizer import Optimizer, SGD, Adam, RMSProp
from nn.loss_functions import LossFunction, MSE, LogLinear, CrossEntropyLoss
from nn.metrics import MSEMetric, MAEMetric
from nn.utils import create_vector, dims
from sklearn.datasets import fetch_california_housing
import copy
import random

housing_data = fetch_california_housing()

