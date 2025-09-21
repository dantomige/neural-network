from utils import dims, apply_func_matrix, matrix_add, scale_matrix

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

            TP += is_true and pred
            TN += is_true and not pred
            FP += not is_true and pred
            FN += not is_true and not pred
        
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
        return TP/(TP + FP)

class Recall(Metrics):

    def evaluate(self, Y, Yhat):
        Ypred = self.pred(Yhat)
        TP, TN, FP, FN = self.confusion_matrix(Y, Ypred)
        return TP/(TP + FN)

class F1(Metrics):
    def evaluate(self, Y, Yhat):
        Ypred = self.pred(Yhat)
        TP, TN, FP, FN = self.confusion_matrix(Y, Ypred)
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        return 2/(1/precision + 1/recall)

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