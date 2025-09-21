from utils import apply_func_matrix, element_multiply_vector, matrix_add, scale_matrix, dims, create_matrix, apply_func_between_matrix_elementwise
from functions import log

class LossFunction:
    def __init__(self):
        pass

    def loss(self, output, input):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class MSE:
    def __init__(self):
        self.Yhat = None
        self.Y = None

    def loss(self, Y, Yhat):
        assert dims(Yhat) == dims(Y)

        num_datapoints, dims_per_Y = dims(Y)
        num_elts = num_datapoints * dims_per_Y

        error_vector = matrix_add(Yhat, scale_matrix(-1, Y))
        square_error_vector = apply_func_matrix(lambda x : x ** 2, error_vector)
        total_squared_error = sum(sum(row) for row in square_error_vector)

        self.Yhat = Yhat
        self.Y = Y

        return total_squared_error/num_elts
    
    def backward(self):
        _, dims_per_Y = dims(self.Y)

        error_vector = matrix_add(self.Yhat, scale_matrix(-1, self.Y))
        pL_pIn = scale_matrix(2/dims_per_Y, error_vector)
        return pL_pIn

class LogLinear:
    def __init__(self):
        self.Y = None
        self.Yhat = None

    def loss(self, Yhat, Y):
        assert dims(Yhat) == dims(Y)
        for row in Y:
            for col in row:
                assert col == 1 or col == 0

        num_datapoints = len(Yhat)
        # y * log(p)
        y_logp = apply_func_between_matrix_elementwise(lambda a,b: a*b, apply_func_matrix(lambda x: log(x), Yhat))
        # (1-y) * log(1-p)
        ones = create_matrix(*dims(self.Y), val=1)
        one_minus_y = matrix_add(ones, scale_matrix(-1, Y))
        one_minus_p = matrix_add(ones, scale_matrix(-1, Yhat))
        eps = 1e-12
        safe_Yhat_one_minus_p = apply_func_matrix(lambda x: min(max(x, eps), 1 - eps), one_minus_p)
        one_minus_y__log_one_minus_p = apply_func_between_matrix_elementwise(lambda a,b: a*b, one_minus_y, apply_func_matrix(lambda x: log(x), safe_Yhat_one_minus_p))
        
        # total loss
        added_together = matrix_add(y_logp, one_minus_y__log_one_minus_p)
        negated = scale_matrix(-1, added_together)
        total_error = sum(sum(row) for row in negated)

        self.Y = Y
        self.Yhat = Yhat
        
        return total_error / num_datapoints

    def backward(self): # (p - y) / (p * (1-p))
        M, N = dims(self.Y)
        num_elts = M * N

        num = matrix_add(self.Yhat, scale_matrix(-1, self.Y) )  # (p - y)
        ones = create_matrix(*dims(self.Y), val=1)
        eps = 1e-12
        denom = apply_func_between_matrix_elementwise(lambda a,b: a*b, self.Yhat, matrix_add(ones, scale_matrix(-1, self.Yhat)))  # p * (1-p)
        denom = apply_func_matrix(lambda x: max(x, eps), denom) 
        frac = apply_func_between_matrix_elementwise(lambda a,b: a*b, num, apply_func_matrix(lambda x: 1/x, denom))
        pL_pIn = scale_matrix(1/num_elts, frac)
        return pL_pIn
    
class CrossEntropyLoss:
    pass