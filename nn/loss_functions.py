from utils import apply_func_matrix, element_multiply_vector, matrix_add, scale_matrix, dims, create_vector
from functions import log

class LossFunction:
    def __init__(self):
        pass


class MSE:
    def __init__(self):
        self.output = None
        self.expected_output = None

    def loss(self, output, expected_output):
        assert dims(output) == dims(expected_output)
        error_vector = matrix_add(output, scale_matrix(-1, expected_output))
        square_error_vector = apply_func_matrix(lambda x : x ** 2, error_vector)
        total_squared_error = sum(sum(row) for row in square_error_vector)
        self.output = output
        self.expected_output = expected_output
        return total_squared_error/len(output)
    
    def backward(self):
        num_dims = len(self.output)
        error_vector = matrix_add(self.output, scale_matrix(-1, self.expected_output))
        pL_pIn = scale_matrix(2/num_dims, error_vector)
        return pL_pIn
    
class LogLinear:
    def __init__(self):
        self.output = None
        self.expected_output = None

    def loss(self, output, expected_output):
        assert dims(output) == dims(expected_output)
        for row in expected_output:
            for col in row:
                assert col == 1 or col == 0

        num_dims = len(output)
        # y * log(p)
        y_logp = element_multiply_vector(expected_output, apply_func_matrix(lambda x: log(x), output))
        # (1-y) * log(1-p)
        ones = create_vector([1] * num_dims)
        one_minus_y = matrix_add(ones, scale_matrix(-1, expected_output))
        one_minus_p = matrix_add(ones, scale_matrix(-1, output))
        eps = 1e-12
        safe_output_one_minus_p = apply_func_matrix(lambda x: min(max(x, eps), 1 - eps), one_minus_p)
        one_minus_y__log_one_minus_p = element_multiply_vector(one_minus_y, apply_func_matrix(lambda x: log(x), safe_output_one_minus_p))
        
        # total loss
        added_together = matrix_add(y_logp, one_minus_y__log_one_minus_p)
        negated = scale_matrix(-1, added_together)
        total_error = sum(sum(row) for row in negated)

        self.output = output
        self.expected_output = expected_output
        return total_error / num_dims

    def backward(self): # (p - y) / (p * (1-p))
        num_dims = len(self.expected_output)
        num = matrix_add(self.output, scale_matrix(-1, self.expected_output) )  # (p - y)
        ones = create_vector([1] * len(self.expected_output))
        eps = 1e-12
        denom = element_multiply_vector(self.output, matrix_add(ones, scale_matrix(-1, self.output)))  # p * (1-p)
        denom = apply_func_matrix(lambda x: max(x, eps), denom) 
        frac = element_multiply_vector(num, apply_func_matrix(lambda x: 1/x, denom))
        pL_pIn = scale_matrix(1/num_dims, frac)
        return pL_pIn