from .utils import dims, apply_func_matrix, matrix_add, scale_matrix

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