import pytest
import copy
from nn.utils import create_matrix
from nn.optimizer import SGD, Adam, RMSProp, Optimizer

EPS = 1e-6

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
# Optimizer.update test
# ---------------------------
def test_optimizer_update():
    params = [[1.0, 2.0], [3.0, 4.0]]
    new_params = [[5.0, 6.0], [7.0, 8.0]]
    Optimizer.update(params, new_params)
    assert matrices_close(params, new_params), "Optimizer.update failed"

# ---------------------------
# SGD Tests
# ---------------------------
def test_sgd_step():
    params = create_matrix(2, 2, 1.0)
    grad = create_matrix(2, 2, 0.1)
    sgd = SGD(learning_rate=0.1)
    old_params = copy.deepcopy(params)
    sgd.step(params, grad)

    # Check that params have changed
    assert not matrices_close(params, old_params), "SGD step did not update parameters"

# ---------------------------
# Adam Tests
# ---------------------------
def test_adam_step():
    params = create_matrix(2, 2, 1.0)
    grad = create_matrix(2, 2, 0.1)
    adam = Adam(learning_rate=0.1)
    old_params = copy.deepcopy(params)
    adam.step(params, grad)

    # Check params changed
    assert not matrices_close(params, old_params), "Adam step did not update parameters"

    # Step again to ensure moments are stored
    prev_params = copy.deepcopy(params)
    adam.step(params, grad)
    assert not matrices_close(params, prev_params), "Adam second step did not update parameters"

# ---------------------------
# RMSProp Tests
# ---------------------------
def test_rmsprop_step():
    params = create_matrix(2, 2, 1.0)
    grad = create_matrix(2, 2, 0.1)
    rms = RMSProp(learning_rate=0.1)
    old_params = copy.deepcopy(params)
    rms.step(params, grad)

    assert not matrices_close(params, old_params), "RMSProp step did not update parameters"

    # Step again to test state update
    prev_params = copy.deepcopy(params)
    rms.step(params, grad)
    assert not matrices_close(params, prev_params), "RMSProp second step did not update parameters"

# ---------------------------
# Test parameter update in place
# ---------------------------
def test_sgd_update_in_place():
    params = create_matrix(2, 2, 1.0)
    grad = create_matrix(2, 2, 0.5)
    sgd = SGD(learning_rate=0.1)
    # Save original object reference
    original_id = id(params)
    sgd.step(params, grad)
    assert id(params) == original_id, "SGD did not update parameters in place"

def test_adam_update_in_place():
    params = create_matrix(2, 2, 1.0)
    grad = create_matrix(2, 2, 0.5)
    adam = Adam(learning_rate=0.1)
    original_id = id(params)
    adam.step(params, grad)
    assert id(params) == original_id, "Adam did not update parameters in place"

def test_rmsprop_update_in_place():
    params = create_matrix(2, 2, 1.0)
    grad = create_matrix(2, 2, 0.5)
    rms = RMSProp(learning_rate=0.1)
    original_id = id(params)
    rms.step(params, grad)
    assert id(params) == original_id, "RMSProp did not update parameters in place"
