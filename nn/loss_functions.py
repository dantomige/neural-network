from utils import apply_func_matrix, matrix_add, scale_matrix, dims

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