import pytest
import random
import math
from nn.initialization import HeKaiming, RandomUniform, RandomNormal

EPS = 1e-6

# ---------------------------
# HeKaiming
# ---------------------------
def test_hekaiming_weight_range():
    random.seed(42)
    hk = HeKaiming()
    input_dim = 4
    weights = [hk.generate_weight(input_dim) for _ in range(1000)]
    std_expected = math.sqrt(2/input_dim)
    mean_weights = sum(weights)/len(weights)
    std_weights = math.sqrt(sum((w - mean_weights)**2 for w in weights)/len(weights))
    # mean close to 0
    assert abs(mean_weights) < 0.1, f"Mean too far from 0: {mean_weights}"
    # std close to expected
    assert abs(std_weights - std_expected) < 0.1, f"Std not close: {std_weights} vs {std_expected}"

# ---------------------------
# RandomUniform
# ---------------------------
def test_randomuniform_bounds():
    random.seed(42)
    ru = RandomUniform(low=-1, high=1)
    weights = [ru.generate_weight() for _ in range(1000)]
    assert all(-1 <= w <= 1 for w in weights), "RandomUniform weight out of bounds"

# ---------------------------
# RandomNormal
# ---------------------------
def test_randomnormal_mean_std():
    random.seed(42)
    mean = 2.0
    std = 0.5
    rn = RandomNormal(mean=mean, std=std)
    weights = [rn.generate_weight() for _ in range(1000)]
    mean_weights = sum(weights)/len(weights)
    std_weights = math.sqrt(sum((w - mean_weights)**2 for w in weights)/len(weights))
    # mean close to target
    assert abs(mean_weights - mean) < 0.1, f"Mean not close: {mean_weights}"
    # std close to target
    assert abs(std_weights - std) < 0.1, f"Std not close: {std_weights}"
