import pytest
import random
import math
import copy
from nn.utils import create_matrix, matrix_multiply, transpose, scale_matrix, dims
from nn.layers import FullyConnected, ReLU, Tanh, Sigmoid, Dropout, Softmax, LayerNorm, BatchNorm

EPS = 1e-5

def matrices_close(mat1, mat2, tol=1e-6):
    if len(mat1) != len(mat2):
        return False
    for r1, r2 in zip(mat1, mat2):
        if len(r1) != len(r2):
            return False
        for a, b in zip(r1, r2):
            if abs(a - b) > tol:
                return False
    return True

# ---------------------------
# FullyConnected Tests
# ---------------------------
def test_fully_connected_forward_backward():
    X_fc = [[1, 2], [3, 4]]
    fc = FullyConnected(input_dim=2, output_dim=2)
    fc.weights = [[1, 0], [0, 1]]  # identity
    fc.biases = [[1, 1]]

    # forward
    out_fc = fc.forward(X_fc)
    expected_fc = [[2, 3], [4, 5]]
    assert matrices_close(out_fc, expected_fc), "Forward failed"

    # backward
    pL_pOut = [[1, 1], [1, 1]]
    fc.inputs = X_fc
    fc.backward(pL_pOut)

    # numerical gradient
    numerical_grad = copy.deepcopy(fc.weights)
    for i in range(len(fc.weights)):
        for j in range(len(fc.weights[0])):
            orig = fc.weights[i][j]
            fc.weights[i][j] = orig + EPS
            plus = fc.forward(X_fc)
            fc.weights[i][j] = orig - EPS
            minus = fc.forward(X_fc)
            fc.weights[i][j] = orig
            N = len(X_fc)
            numerical_grad[i][j] = sum((plus[r][c] - minus[r][c]) * pL_pOut[r][c] for r in range(2) for c in range(2)) / (2 * EPS * N)

    assert matrices_close(fc.pL_pW, numerical_grad, tol=1e-4), "Weight gradient check failed"

# ---------------------------
# ReLU Tests
# ---------------------------
def test_relu_forward_backward():
    X = [[-1, 2], [0, -3]]
    relu = ReLU()
    out = relu.forward(X)
    assert matrices_close(out, [[0, 2], [0, 0]]), "ReLU forward failed"
    grad = relu.backward([[1, 1], [1, 1]])
    assert matrices_close(grad, [[0, 1], [0, 0]]), "ReLU backward failed"

# ---------------------------
# Sigmoid Tests
# ---------------------------
def test_sigmoid_forward_backward():
    X = [[0, 2]]
    sig = Sigmoid()
    out = sig.forward(X)
    expected_out = [[1/(1+math.exp(0)), 1/(1+math.exp(-2))]]
    assert matrices_close(out, expected_out), "Sigmoid forward failed"
    grad = sig.backward([[1, 1]])
    expected_grad = [[out[0][0]*(1-out[0][0]), out[0][1]*(1-out[0][1])]]
    assert matrices_close(grad, expected_grad), "Sigmoid backward failed"

# ---------------------------
# Tanh Tests
# ---------------------------
def test_tanh_forward_backward():
    X = [[0, 1]]
    t = Tanh()
    out = t.forward(X)
    grad = t.backward([[1, 1]])
    expected_grad = [[1 - out[0][0]**2, 1 - out[0][1]**2]]
    assert matrices_close(grad, expected_grad), "Tanh backward failed"

# ---------------------------
# Dropout Tests
# ---------------------------
def test_dropout_forward_backward():
    random.seed(42)
    X = [[1, 2], [3, 4]]
    drop = Dropout(dropout_prob=0.5)
    out1 = drop.forward(X, training=True)
    random.seed(42)
    out2 = drop.forward(X, training=True)
    assert matrices_close(out1, out2), "Dropout reproducibility failed"
    out_eval = drop.forward(X, training=False)
    assert matrices_close(out_eval, X), "Dropout inference failed"

# ---------------------------
# Softmax Tests
# ---------------------------
def test_softmax_forward():
    X = [[1, 2, 3]]
    sm = Softmax()
    out = sm.forward(X)
    s = sum(out[0])
    assert abs(s - 1.0) < 1e-6, "Softmax forward failed"

# ---------------------------
# LayerNorm / BatchNorm Tests (forward only)
# ---------------------------
def test_layernorm_forward():
    X = [[1, 2], [3, 4]]
    ln = LayerNorm(dim=2, epsilon=1e-5)
    out = ln.forward(X)
    # just check dimensions
    assert dims(out) == dims(X), "LayerNorm forward failed"

def test_batchnorm_forward():
    X = [[1, 2], [3, 4]]
    bn = BatchNorm(epsilon=1e-5)
    out = bn.forward(X)
    assert dims(out) == dims(X), "BatchNorm forward failed"
