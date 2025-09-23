from .utils import apply_func_matrix, element_multiply_vector, matrix_add, scale_matrix, dims, create_matrix, apply_func_between_matrix_elementwise
from .functions import log

class LossFunction:
    def __init__(self):
        self.Yhat = None
        self.Y = None
        self.loss_value = None


    def loss(self, output, input):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class MSE(LossFunction):
    def __init__(self):
        super().__init__()

    def loss(self, Y, Yhat):
        assert dims(Yhat) == dims(Y)

        num_datapoints, dims_per_Y = dims(Y)
        num_elts = num_datapoints * dims_per_Y

        error_vector = matrix_add(Yhat, scale_matrix(-1, Y))
        square_error_vector = apply_func_matrix(lambda x : x ** 2, error_vector)
        total_squared_error = sum(sum(row) for row in square_error_vector)

        self.Yhat = Yhat
        self.Y = Y
        self.loss_value = total_squared_error / num_elts
        return self.loss_value

    def backward(self):
        N, dims_per_Y = dims(self.Y)
        num_elts = N * dims_per_Y

        error_vector = matrix_add(self.Yhat, scale_matrix(-1, self.Y))
        pL_pIn = scale_matrix(2/num_elts, error_vector)
        return pL_pIn

class LogLinear(LossFunction):
    def __init__(self):
        super().__init__()

    def loss(self, Y, Yhat):
        assert dims(Yhat) == dims(Y)
        for row in Y:
            for col in row:
                assert col == 1 or col == 0, f"Y should be binary (0 or 1): {col}"

        self.Y = Y
        self.Yhat = Yhat

        num_datapoints = len(Yhat)
        # y * log(p)
        y_logp = apply_func_between_matrix_elementwise(lambda a,b: a*b, self.Y, apply_func_matrix(lambda x: log(x), self.Yhat))
        # (1-y) * log(1-p)
        ones = create_matrix(*dims(self.Y), val=1)
        one_minus_y = matrix_add(ones, scale_matrix(-1, self.Y))
        one_minus_p = matrix_add(ones, scale_matrix(-1, self.Yhat))
        eps = 1e-12
        safe_Yhat_one_minus_p = apply_func_matrix(lambda x: min(max(x, eps), 1 - eps), one_minus_p)
        one_minus_y__log_one_minus_p = apply_func_between_matrix_elementwise(lambda a,b: a*b, one_minus_y, apply_func_matrix(lambda x: log(x), safe_Yhat_one_minus_p))
        
        # total loss
        added_together = matrix_add(y_logp, one_minus_y__log_one_minus_p)
        negated = scale_matrix(-1, added_together)
        total_error = sum(sum(row) for row in negated)

        self.loss_value = total_error / num_datapoints
        return self.loss_value

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
    
class CrossEntropyLoss(LossFunction):
    def __init__(self):
        super().__init__()

    def loss(self, Y, Yhat):
        assert dims(Yhat) == dims(Y)
        self.Y = Y
        self.Yhat = Yhat

        num_datapoints, num_classes = dims(Y)
        eps = 1e-12

        # Clip predictions to avoid log(0)
        safe_Yhat = apply_func_matrix(lambda x: min(max(x, eps), 1 - eps), Yhat)

        # elementwise y * log(yhat)
        y_logp = apply_func_between_matrix_elementwise(
            lambda a, b: a * log(b), self.Y, safe_Yhat
        )

        # total error = -sum(y * log(yhat)) / M
        total_error = -sum(sum(row) for row in y_logp)
        self.loss_value = total_error / num_datapoints
        return self.loss_value

    def backward(self):
        M, K = dims(self.Y)
        eps = 1e-12

        # Clip Yhat to avoid division by zero
        safe_Yhat = apply_func_matrix(lambda x: max(x, eps), self.Yhat)

        # gradient = -(Y / (M * Yhat))
        grad = apply_func_between_matrix_elementwise(
            lambda y, yhat: -y / (M * yhat), self.Y, safe_Yhat
        )
        return grad
