import pytest
import math
from nn.loss_functions import MSE, LogLinear, CrossEntropyLoss

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
# MSE tests
# ---------------------------
def test_mse_forward():
    mse = MSE()
    Y = [[1, 2], [3, 4]]
    Yhat = [[1, 3], [2, 5]]
    loss_val = mse.loss(Y, Yhat)
    expected_loss = ((0)**2 + (1)**2 + (1)**2 + (1)**2) / 2 / 2  # 4 elements, 1/N
    assert math.isclose(loss_val, expected_loss, rel_tol=1e-6)

def test_mse_backward():
    mse = MSE()
    Y = [[1, 2], [3, 4]]
    Yhat = [[1, 3], [2, 5]]
    mse.loss(Y, Yhat)
    grad = mse.backward()
    expected_grad = [[0, 1], [-1, 1]]  # 2/2 per formula
    assert matrices_close(grad, expected_grad, tol=1e-6)

# ---------------------------
# LogLinear tests
# ---------------------------
def test_loglinear_forward():
    loglinear = LogLinear()
    Y = [[1, 0], [0, 1]]
    Yhat = [[0.9, 0.1], [0.2, 0.8]]
    loss_val = loglinear.loss(Y, Yhat)
    # Manually compute expected
    import math
    eps = 1e-12
    expected = (-1)*(math.log(0.9) + math.log(0.9) + math.log(0.8) + math.log(0.8)) / 2
    assert math.isclose(loss_val, expected, rel_tol=1e-6)

def test_loglinear_backward():
    loglinear = LogLinear()
    Y = [[1, 0], [0, 1]]
    Yhat = [[0.9, 0.1], [0.2, 0.8]]
    loglinear.loss(Y, Yhat)
    grad = loglinear.backward()
    # Check shape matches
    assert len(grad) == len(Y)
    assert all(len(row) == len(Y[0]) for row in grad)

def test_loglinear_invalid_input():
    loglinear = LogLinear()
    Y = [[0, 2]]  # invalid value
    Yhat = [[0.5, 0.5]]
    with pytest.raises(AssertionError):
        loglinear.loss(Y, Yhat)

# ---------------------------
# CrossEntropyLoss placeholder
# ---------------------------
def test_crossentropyloss_placeholder():
    celoss = CrossEntropyLoss()
    # Just ensure instantiation works
    assert celoss is not None
