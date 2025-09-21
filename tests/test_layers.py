import random
import math
import copy
from nn.layers import FullyConnected, Dropout, ReLU, Sigmoid, LayerNorm

EPS = 1e-5  # for numerical gradient check

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
# FullyConnected Forward Test
# ---------------------------
X_fc = [[1,2],[3,4]]
fc = FullyConnected(input_dim=2, output_dim=2)
fc.weights = [[1,0],[0,1]]  # identity
fc.biases = [[1,1]]
out_fc = fc.forward(X_fc)
expected_fc = [[2,3],[4,5]]  # X*I^T + b
assert matrices_close(out_fc, expected_fc), "FullyConnected forward failed"

# ---------------------------
# FullyConnected Backward Gradient Check
# ---------------------------
pL_pOut = [[1,1],[1,1]]
fc.inputs = X_fc
fc.backward(pL_pOut)
# Numerical gradient for weights
numerical_grad = copy.deepcopy(fc.weights)
for i in range(len(fc.weights)):
    for j in range(len(fc.weights[0])):
        orig = fc.weights[i][j]
        fc.weights[i][j] = orig + EPS
        plus = fc.forward(X_fc)
        fc.weights[i][j] = orig - EPS
        minus = fc.forward(X_fc)
        fc.weights[i][j] = orig
        # derivative approximated
        numerical_grad[i][j] = sum((plus[r][c]-minus[r][c])*pL_pOut[r][c] for r in range(2) for c in range(2))/(2*EPS)
assert matrices_close(fc.pL_pW, numerical_grad, tol=1e-4), "FullyConnected weight gradient check failed"

print("FullyConnected tests passed ✅")

# ---------------------------
# ReLU Forward and Backward
# ---------------------------
X_relu = [[-1,2],[0,-3]]
relu = ReLU()
out_relu = relu.forward(X_relu)
assert matrices_close(out_relu, [[0,2],[0,0]]), "ReLU forward failed"

grad_relu = relu.backward([[1,1],[1,1]])
assert matrices_close(grad_relu, [[0,1],[0,0]]), "ReLU backward failed"
print("ReLU tests passed ✅")

# ---------------------------
# Sigmoid Forward and Backward
# ---------------------------
X_sig = [[0,2]]
sig = Sigmoid()
out_sig = sig.forward(X_sig)
expected_sig = [[1/(1+math.exp(0)), 1/(1+math.exp(-2))]]
assert matrices_close(out_sig, expected_sig), "Sigmoid forward failed"

grad_sig = sig.backward([[1,1]])
expected_grad = [[out_sig[0][0]*(1-out_sig[0][0]), out_sig[0][1]*(1-out_sig[0][1])]]
assert matrices_close(grad_sig, expected_grad), "Sigmoid backward failed"
print("Sigmoid tests passed ✅")

# ---------------------------
# Dropout Forward and Backward
# ---------------------------
random.seed(42)
X_drop = [[1,2],[3,4]]
drop = Dropout(dropout_prob=0.5)
out1 = drop.forward(X_drop, training=True)
random.seed(42)
out2 = drop.forward(X_drop, training=True)
assert matrices_close(out1, out2), "Dropout forward reproducibility failed"

# Inference mode
out_eval = drop.forward(X_drop, training=False)
assert matrices_close(out_eval, X_drop), "Dropout inference failed"
print("Dropout tests passed ✅")

# ---------------------------
# LayerNorm Forward Test
# ---------------------------
X_ln = [[1,2],[3,4]]
ln = LayerNorm(epsilon=1e-5)
out_ln = ln.forward(X_ln)
mean = sum(sum(row) for row in X_ln)/4
var = sum(sum((x-mean)**2 for x in row) for row in X_ln)/4
expected_ln = [[(1-mean)/(var+1e-5)**0.5, (2-mean)/(var+1e-5)**0.5],
               [(3-mean)/(var+1e-5)**0.5, (4-mean)/(var+1e-5)**0.5]]
assert matrices_close(out_ln, expected_ln), "LayerNorm forward failed"
print("LayerNorm tests passed ✅")

print("All deterministic layer tests passed ✅")
