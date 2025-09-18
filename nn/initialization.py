from enum import Enum
import random
import math

class Initializations:
    def __init__(self):
        pass

    def init_weights(self, input_dim, output_dim):
        return [[self.generate_weight(input_dim, output_dim) for _ in range(output_dim)] for _ in range(input_dim)]

    def generate_weight(self, input_dim, output_dim):
        raise NotImplementedError

class HeKaiming(Initializations):

    def __init__(self):
        pass

    def generate_weight(self, input_dim, output_dim):
        mean = 0
        std = math.sqrt(2/input_dim)
        return random.gauss(mean, std)

class RandomUniform(Initializations):

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def generate_weight(self, input_dim, output_dim):
        return random.uniform(self.low, self.high)

class RandomNormal(Initializations):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def generate_weight(self, input_dim, output_dim):
        return random.gauss(self.mean, self.std)