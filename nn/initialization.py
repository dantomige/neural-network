from enum import Enum
import random
import math

class Initializations:

    def init_weights(self, input_dim, output_dim):
        return [[self.generate_weight(input_dim) for _ in range(input_dim)] for _ in range(output_dim)]

    def generate_weight(self):
        raise NotImplementedError

class HeKaiming(Initializations):

    def generate_weight(self, input_dim):
        mean = 0
        std = math.sqrt(2/input_dim)
        return random.gauss(mean, std)

class RandomUniform(Initializations):

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def generate_weight(self, input_dim=None):
        return random.uniform(self.low, self.high)

class RandomNormal(Initializations):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def generate_weight(self, input_dim=None):
        return random.gauss(self.mean, self.std)