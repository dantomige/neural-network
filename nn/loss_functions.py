class LossFunction:
    def __init__(self):
        pass


class MSE:
    def __init__(self):
        self.output = None
        self.expected_output = None

    def loss(self, output, expected_output):
        # print(output, expected_output)
        total_squared_error = sum((output_val - expected_output_val)**2 for output_val, expected_output_val in zip(output, expected_output))
        self.output = output
        self.expected_output = expected_output
        return total_squared_error/len(output)
    
    def backward(self):
        out = []
        num_dims = len(self.output)
        for output_val, expected_output_val in zip(self.output, self.expected_output):
            out.append([2 * (output_val - expected_output_val) / num_dims])
        return out