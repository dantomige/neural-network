import math

def sigmoid(x):
    out = 1/(1 + math.exp(-x))
    assert 0 <= out <= 1
    return out

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def log(x):
    return math.log(x)
