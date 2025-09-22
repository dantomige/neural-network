import pytest
from nn.metrics import Accuracy, Precision, Recall, F1, MSEMetric, MAEMetric

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
# Classification tests
# ---------------------------
def test_accuracy_basic():
    Y = [[1], [0], [1], [0]]
    Yhat = [[0.9], [0.1], [0.8], [0.2]]
    acc = Accuracy()
    assert acc.evaluate(Y, Yhat) == 1.0

def test_precision_basic():
    Y = [[1], [0], [1], [0]]
    Yhat = [[0.9], [0.8], [0.2], [0.1]]  # two false positives
    prec = Precision()
    assert abs(prec.evaluate(Y, Yhat) - 0.5) < 1e-6

def test_recall_basic():
    Y = [[1], [0], [1], [0]]
    Yhat = [[0.9], [0.8], [0.2], [0.1]]  # one false negative
    rec = Recall()
    expected_recall = 1 / 2  # TP=1, FN=1
    assert abs(rec.evaluate(Y, Yhat) - expected_recall) < 1e-6

def test_f1_basic():
    Y = [[1], [0], [1], [0]]
    Yhat = [[0.9], [0.8], [0.2], [0.1]]
    f1_metric = F1()
    # TP=1, FP=1, FN=1
    precision = 1 / 2
    recall = 1 / 2
    expected_f1 = 2 * (precision * recall) / (precision + recall)
    assert abs(f1_metric.evaluate(Y, Yhat) - expected_f1) < 1e-6

def test_pred_threshold():
    from nn.metrics import Metrics
    Yhat = [[0.49], [0.5], [0.51]]
    preds = Metrics.pred(Yhat)
    assert preds == [[0], [1], [1]]

def test_confusion_matrix_basic():
    from nn.metrics import Metrics
    Y = [[1], [0], [1], [0]]
    Ypred = [[1], [1], [0], [0]]
    TP, TN, FP, FN = Metrics.confusion_matrix(Y, Ypred)
    assert (TP, TN, FP, FN) == (1, 1, 1, 1)

# ---------------------------
# Regression tests
# ---------------------------
def test_mse_metric():
    Y = [[1, 2], [3, 4]]
    Yhat = [[1, 3], [2, 5]]
    mse_metric = MSEMetric()
    value = mse_metric.evaluate(Y, Yhat)
    # Manual computation
    expected = ((0)**2 + 1**2 + (-1)**2 + 1**2)/4
    assert abs(value - expected) < 1e-6

def test_mae_metric():
    Y = [[1, 2], [3, 4]]
    Yhat = [[1, 3], [2, 5]]
    mae_metric = MAEMetric()
    value = mae_metric.evaluate(Y, Yhat)
    # Manual computation
    expected = (0 + 1 + 1 + 1)/4
    assert abs(value - expected) < 1e-6
