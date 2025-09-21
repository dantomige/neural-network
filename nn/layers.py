from utils import create_vector, create_matrix, transpose, matrix_add, scale_matrix, matrix_multiply, element_multiply_matrix, element_multiply_vector, broadcast, apply_func_matrix, dims, identity, apply_func_between_matrix_elementwise
from functions import sigmoid, sigmoid_deriv
from initialization import Initializations, HeKaiming, RandomNormal, RandomUniform
from optimizer import Optimizer, SGD, Adam, RMSProp
import random
import math

class Layer:

    def __init__(self):
        self.inputs = None
        self.pL_pIn = None
        self.is_trainable = False

    def forward(self, input, training=True):
        raise NotImplementedError

    def backward(self, pL_pOut):
        raise NotImplementedError
    
    def init_params(self, init: Initializations=None):
        pass

    def update_params(self, optimizer):
        pass

    def __str__(self):
        raise NotImplementedError
    
class FullyConnected(Layer):

    def __init__(self, input_dim, output_dim, init=HeKaiming()):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = init
        self.is_trainable = True
        self.inputs = None
        self.weights = None
        self.biases = None
        self.pL_pW = None
        self.pL_pB = None
        self.pL_pIn = None

    def forward(self, X, training=True):
        self.inputs = X
        Wt_X = matrix_multiply(X, transpose(self.weights))
        output = broadcast(self.biases, Wt_X, func=lambda a, b: a + b)
        return output 

    def init_params(self):
        self.weights = self.init.init_weights(self.input_dim, self.output_dim)
        self.biases = [[0 for _ in range(self.output_dim)]]

    def backward(self, pL_pOut):
        N = len(self.inputs)

        self.pL_pW = scale_matrix(1/N, matrix_multiply(transpose(pL_pOut), self.inputs))
        self.pL_pB = scale_matrix(1/N, [[sum(row) for row in transpose(pL_pOut)]])
        self.pL_pIn = matrix_multiply(pL_pOut, self.weights)

        return self.pL_pIn


    def update_params(self, optimizer: Optimizer):
        optimizer.step(self.weights, self.pL_pW)
        optimizer.step(self.biases, self.pL_pB)

    def __str__(self):
        if self.weights is None or self.biases is None:
            return "Weights and biases not initialized."

        num_row = len(self.weights)
        num_col = len(self.weights[0])
        col_width = 10  # space for each number

        output = ""

        # Weight rows
        for i in range(num_col):
            output += "       ".ljust(col_width) if i else "Weights".ljust(col_width)
            # output += "Weights".ljust(col_width)
            for j in range(num_row):
                output += f"{self.weights[j][i]:.2f}".ljust(col_width)
            output += "\n"

        # Bias row
        output += "Bias".ljust(col_width)
        for j in range(num_row):
            bias_val = self.biases[0][j]
            output += f"{bias_val:.2f}".ljust(col_width)
        output += "\n"

        return output
    
class Dropout(Layer):

    def __init__(self, dropout_prob=0.2):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.mask = None

    def forward(self, X, training=True):
        self.inputs = X
        N, input_dim = dims(self.inputs)
        if training:
            self.mask = [[0 if random.random() < self.dropout_prob else 1 for _ in range(input_dim)] for _ in range(N)]

            assert dims(self.inputs) == dims(self.mask)

            masked_input = apply_func_between_matrix_elementwise(lambda a,b: a * b, self.mask, self.inputs)
            output = scale_matrix(1/(1-self.dropout_prob), masked_input)
            return output
        else:
            self.mask = None
            return self.inputs

    def backward(self, pL_pOut):
        if self.mask is not None:
            masked_deriv = apply_func_between_matrix_elementwise(lambda a,b: a * b, self.mask, pL_pOut)
            self.pL_pIn = scale_matrix(1/(1-self.dropout_prob), masked_deriv)
            return self.pL_pIn
        else:
            self.pL_pIn = pL_pOut
            return self.pL_pIn

    def __str__(self):
        return f"Dropout(prob={self.dropout_prob})\n"

class LayerNorm(Layer):

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, X, training=True):
        self.inputs = X
        num_dims = len(X)
        mean = sum(sum(row) for row in X)/num_dims
        squared_error = apply_func_matrix(lambda x: (x - mean)**2)
        var = sum(sum(row) for row in squared_error)/num_dims
        output = apply_func_matrix(lambda x: (x - mean)/(var + self.epsilon)**(1/2))
        return output

    def backward(self, pL_pOut):
        raise NotImplementedError

    def __str__(self):
        return "LayerNorm()"

class ReLU(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, X, training=True):
        self.inputs = X
        return [[max(0, x) for x in row] for row in X]

    def backward(self, pL_pOut):
        pOut_pIn = [[1 if x > 0 else 0 for x in row] for row in self.inputs]
        mult = lambda a, b: a * b
        self.pL_pIn = apply_func_between_matrix_elementwise(mult, pOut_pIn, pL_pOut)
        return self.pL_pIn

    def __str__(self):
        return "ReLU()\n"
    
class Tanh(Layer):
    pass
    
class Sigmoid(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, X, training=True):
        self.inputs = X
        output = apply_func_matrix(lambda x: sigmoid(x), X)
        return output

    def backward(self, pL_pOut):
        pOut_pIn = apply_func_matrix(lambda x: sigmoid_deriv(x), self.inputs)
        mult = lambda a, b: a * b
        self.pL_pIn = apply_func_between_matrix_elementwise(mult, pOut_pIn, pL_pOut)
        return self.pL_pIn

    def __str__(self):
        return "Sigmoid()\n"

class Softmax():
    pass