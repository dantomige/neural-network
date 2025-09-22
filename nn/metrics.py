from .utils import dims, apply_func_matrix, matrix_add, scale_matrix, create_matrix, transpose

class Metrics:
    
    def __init__(self):
        pass

    def evaluate(self, Y, Yhat):
        raise NotImplementedError

    @staticmethod
    def pred(Yhat):
        logit_to_class = lambda p: 1 if p >= 0.5 else 0
        Ypred = apply_func_matrix(logit_to_class, Yhat)
        return Ypred

    @staticmethod
    def confusion_matrix(Y, Ypred):
        assert dims(Y) == dims(Ypred)
        TP = TN = FP = FN = 0
        for rowY, rowYhat in zip(Y, Ypred):
            is_true = rowY[0]
            pred = rowYhat[0]

            TP += int(is_true == 1 and pred == 1)
            TN += int(is_true == 0 and pred == 0)
            FP += int(is_true == 0 and pred == 1)
            FN += int(is_true == 1 and pred == 0)
        return TP, TN, FP, FN
                

# categorical
class Accuracy(Metrics):

    def evaluate(self, Y, Yhat):
        Ypred = self.pred(Yhat)
        TP, TN, FP, FN = self.confusion_matrix(Y, Ypred)
        return (TP + TN)/(TP + TN + FP + FN)
    
class Precision(Metrics):
    
    def evaluate(self, Y, Yhat):
        Ypred = self.pred(Yhat)
        TP, TN, FP, FN = self.confusion_matrix(Y, Ypred)
        return TP / (TP + FP) if (TP + FP) != 0 else 0

class Recall(Metrics):

    def evaluate(self, Y, Yhat):
        Ypred = self.pred(Yhat)
        TP, TN, FP, FN = self.confusion_matrix(Y, Ypred)
        return TP / (TP + FN) if (TP + FN) != 0 else 0

class F1(Metrics):
    def evaluate(self, Y, Yhat):
        Ypred = self.pred(Yhat)
        TP, TN, FP, FN = self.confusion_matrix(Y, Ypred)

        # Safe precision and recall
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0

        # Safe F1
        if precision + recall == 0:
            return 0
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

# regression
class MSEMetric(Metrics):
    def evaluate(self, Y, Yhat):
        assert dims(Yhat) == dims(Y)

        M, N = dims(Yhat)
        num_elts = M * N

        error_vector = matrix_add(Yhat, scale_matrix(-1, Y))
        square_error_vector = apply_func_matrix(lambda x : x ** 2, error_vector)
        total_squared_error = sum(sum(row) for row in square_error_vector)

        self.Yhat = Yhat
        self.Y = Y

        return total_squared_error/num_elts

class MAEMetric(Metrics):
    def evaluate(self, Y, Yhat):
        assert dims(Yhat) == dims(Y)

        M, N = dims(Yhat)
        num_elts = M * N

        error_vector = matrix_add(Yhat, scale_matrix(-1, Y))
        square_error_vector = apply_func_matrix(lambda x : abs(x), error_vector)
        total_squared_error = sum(sum(row) for row in square_error_vector)

        self.Yhat = Yhat
        self.Y = Y

        return total_squared_error/num_elts

class MAEMetric(Metrics):
    def evaluate(self, Y, Yhat):
        assert dims(Yhat) == dims(Y)

        M, N = dims(Yhat)
        num_elts = M * N

        error_vector = matrix_add(Yhat, scale_matrix(-1, Y))
        square_error_vector = apply_func_matrix(lambda x : abs(x), error_vector)
        total_squared_error = sum(sum(row) for row in square_error_vector)

        self.Yhat = Yhat
        self.Y = Y

        return total_squared_error/num_elts
    
class R2Metric:
    def __init__(self):
        self.Y = None
        self.Yhat = None

    # Helper functions (self-contained)
    @staticmethod
    def dims(A):
        return (len(A), len(A[0]))

    @staticmethod
    def create_matrix(rows, cols, val=0):
        return [[val for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def matrix_add(A, B):
        return [[a + b for a, b in zip(rowA, rowB)] for rowA, rowB in zip(A, B)]

    @staticmethod
    def scale_matrix(scalar, A):
        return [[scalar * val for val in row] for row in A]

    @staticmethod
    def apply_func_matrix(func, A):
        return [[func(val) for val in row] for row in A]

    def evaluate(self, Y, Yhat):
        assert self.dims(Y) == self.dims(Yhat), "Y and Yhat must have same dimensions"
        M, N = self.dims(Y)

        # Compute column means
        column_means = [sum(Y[i][j] for i in range(M)) / M for j in range(N)]

        # Broadcast means to matrix
        mean_matrix = self.create_matrix(M, N, 0)
        for i in range(M):
            for j in range(N):
                mean_matrix[i][j] = column_means[j]

        # Residuals: Y - Yhat
        residuals = self.matrix_add(Yhat, self.scale_matrix(-1, Y))
        squared_residuals = self.apply_func_matrix(lambda x: x**2, residuals)
        ss_res = sum(sum(row) for row in squared_residuals)

        # Total sum of squares: Y - mean
        deviations = self.matrix_add(Y, self.scale_matrix(-1, mean_matrix))
        squared_deviations = self.apply_func_matrix(lambda x: x**2, deviations)
        ss_tot = sum(sum(row) for row in squared_deviations)

        r2 = 1 - (ss_res / ss_tot)

        self.Y = Y
        self.Yhat = Yhat
        return r2