from .utils import create_vector, create_matrix, transpose, matrix_add, scale_matrix, matrix_multiply, element_multiply_matrix, element_multiply_vector, broadcast, apply_func_matrix, dims, identity, apply_func_between_matrix_elementwise, create_diag_matrix
from .functions import sigmoid, sigmoid_deriv, tanh, tanh_deriv
from .initialization import Initialization, HeKaiming, RandomNormal, RandomUniform
from .optimizer import Optimizer, SGD, Adam, RMSProp
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
    
    def init_params(self, init: Initialization=None):
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

        self.pL_pW = matrix_multiply(transpose(pL_pOut), self.inputs)
        self.pL_pB = [[sum(row) for row in transpose(pL_pOut)]]
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

    def __init__(self, dim, epsilon=0.1, init=RandomNormal(0, 0.1)):
        super().__init__()
        self.epsilon = epsilon
        self.dim = dim
        self.scale_param = None
        self.shift_param = None

    def forward(self, X, training=True):
        self.inputs = X
        output = []

        for input in self.inputs:

            mean = sum(input)
            total_squared_error = sum((val - mean)**2 for val in input)
            std = (total_squared_error + self.epsilon) ** (1/2)

            normalized_input = [(val - mean)/std for val in input]
            scaled_input = []
            output.append(input)
            squared_error = apply_func_matrix(lambda x: (x - mean)/std, input)
            std = apply_func_matrix(lambda x: (x + self.epsilon)**(1/2), squared_error)
        
        num_dims = len(X)
        mean = sum(sum(row) for row in X)/num_dims
        squared_error = apply_func_matrix(lambda x: (x - mean)**2)
        var = sum(sum(row) for row in squared_error)/num_dims
        output = apply_func_matrix(lambda x: (x - mean)/(var + self.epsilon)**(1/2))
        raise NotImplementedError
    
    def init_params(self):
        self.scale_param = create_matrix(1, self.dim, 1)
        self.shift_param = create_matrix(1, self.dim, 0)

    def backward(self, pL_pOut):
        raise NotImplementedError

    def __str__(self):
        return "LayerNorm()"
    
class BatchNorm(Layer):

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, X, training=True):
        self.inputs = X
        num_dims = len(X)
        mean = sum(sum(row) for row in X)/num_dims
        squared_error = apply_func_matrix(lambda x: (x - mean)**2)
        var = sum(sum(row) for row in squared_error)/num_dims
        output = apply_func_matrix(lambda x: (x - mean)/(var + self.epsilon)**(1/2), squared_error)
        raise NotImplementedError

    def backward(self, pL_pOut):
        raise NotImplementedError

    def __str__(self):
        return "BatchNorm()"

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

    def __init__(self):
        super().__init__()

    def forward(self, X, training=True):
        self.inputs = X
        self.outputs = [[tanh(x) for x in row] for row in X]
        return self.outputs

    def backward(self, pL_pOut):
        pOut_pIn = apply_func_matrix(lambda y: 1 - y ** 2, self.outputs)
        mult = lambda a, b: a * b
        self.pL_pIn = apply_func_between_matrix_elementwise(mult, pOut_pIn, pL_pOut)
        return self.pL_pIn

    def __str__(self):
        return "Tanh()\n"
    
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

class Softmax(Layer):
    def __init__(self, epsilon=1e-9):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, X, training=True):
        self.inputs = X
        self.outputs = []

        for row in X:
            # 1. subtract max for stability
            max_val = max(row)
            exps = [math.exp(val - max_val) for val in row]
            sum_exps = sum(exps) + self.epsilon
            softmax_row = [exp_val / sum_exps for exp_val in exps]
            self.outputs.append(softmax_row)

        return self.outputs

    def backward(self, pL_pOut):
        self.pL_pIn = []

        for output, grad_output in zip(self.outputs, pL_pOut):
            # Create Jacobian matrix: J = diag(softmax) - softmax.T @ softmax
            diag_softmax = create_diag_matrix(output)
            output_vec = [output]  # 1 x N
            outer_product = matrix_multiply(transpose(output_vec), output_vec)
            jacobian = apply_func_between_matrix_elementwise(
                lambda x, y: x - y, diag_softmax, outer_product
            )

            # Gradient: pL_pIn = grad_output @ jacobian
            grad_input = matrix_multiply([grad_output], jacobian)[0]  # flatten 1 x N
            self.pL_pIn.append(grad_input)

        return self.pL_pIn