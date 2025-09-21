from .utils import create_matrix, matrix_add, scale_matrix, apply_func_matrix, apply_func_between_matrix_elementwise, dims

class Optimizer:
    
    def __init__(self):
        raise NotImplementedError

    def step(self, params, grad):
        raise NotImplementedError
    
    @staticmethod
    def update(old_param, new_param):
        """Updates old parameters with new parameters in place"""
        num_rows, num_cols = dims(old_param)
        for i in range(num_rows):
            for j in range(num_cols):
                old_param[i][j] = new_param[i][j]

class SGD(Optimizer): # default optimizer
    def __init__(self, learning_rate=0.01, momentum=0.0, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.momentum_buffer = {}

    def step(self, params, grad):
        # print("grad", grad)
        # weight decay
        grad_with_weighted_params = matrix_add(grad, scale_matrix(self.weight_decay, params))

        # calculating grad based on momentum
        v_t = self.momentum_buffer.get(id(params), create_matrix(*dims(params), 0))
        v_tplus1 = matrix_add(scale_matrix(self.momentum, v_t), grad_with_weighted_params)

        # grad descent step
        new_params = matrix_add(params, scale_matrix(-1 * self.learning_rate, v_tplus1))

        # update parameter maintaining location in memory
        self.update(params, new_params)

        # save momentum
        self.momentum_buffer[id(params)] = v_tplus1
    
class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.first_moment = {}
        self.second_moment = {}
        self.t = 0

    def step(self, params, grad):
        # weight decay
        grad_with_weighted_params = matrix_add(grad, scale_matrix(self.weight_decay, params))

        # getting previous first_moment
        v_t = self.first_moment.get(id(params), create_matrix(*dims(params), 0))

        # calculating grad based on momentum: v_tplus1 = beta1 * v_t + (1 - beta1) * grad
        v_tplus1 = matrix_add(scale_matrix(self.beta1, v_t), scale_matrix(1 - self.beta1, grad_with_weighted_params))

        # getting previous second_moment
        s_t = self.second_moment.get(id(params), create_matrix(*dims(params), 0))

        # s_tplus1 = alpha * s_t + (1 - alpha) * grad**2
        s_tplus1 = matrix_add(scale_matrix(self.beta2, s_t), scale_matrix(1 - self.beta2, apply_func_matrix(lambda x: x**2, grad_with_weighted_params)))

        # adjust t for bias adjustment
        self.t += 1

        # bias adjustment
        vhat_tplus1 = scale_matrix(1/(1 - self.beta1**self.t), v_tplus1)
        shat_tplus1 = scale_matrix(1/(1 - self.beta1**self.t), s_tplus1)

        # adjusting s_tplus1 scale: sqrt(s_tplus1) + epsilon
        adjusted_scale = apply_func_matrix(lambda x: x**(1/2) + self.epsilon, shat_tplus1)
        grad_scaled = apply_func_between_matrix_elementwise(lambda a, b: a/b, vhat_tplus1, adjusted_scale)

        # grad descent step
        new_params = matrix_add(params, scale_matrix(-1 * self.learning_rate, grad_scaled))

        # update parameter maintaining location in memory
        self.update(params, new_params)

        # save moments
        self.first_moment[id(params)] = v_tplus1
        self.second_moment[id(params)] = s_tplus1

class RMSProp(Optimizer):

    def __init__(self, learning_rate=0.01, alpha=0.9, weight_decay=0.0, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.second_moment = {}
    
    def step(self, params, grad):
        # weight decay
        grad_with_weighted_params = matrix_add(grad, scale_matrix(self.weight_decay, params))

        # getting previous second_moment
        s_t = self.second_moment.get(id(params), create_matrix(*dims(params), 0))

        # s_tplus1 = alpha * s_t + (1 - alpha) * grad**2
        s_tplus1 = matrix_add(scale_matrix(self.alpha, s_t), scale_matrix(1 - self.alpha, apply_func_matrix(lambda x: x**2, grad_with_weighted_params)))

        # adjusting s_tplus1 scale: sqrt(s_tplus1) + epsilon
        adjusted_scale = apply_func_matrix(lambda x: x**(1/2) + self.epsilon, s_tplus1)
        grad_scaled = apply_func_between_matrix_elementwise(lambda a, b: a/b, grad_with_weighted_params, adjusted_scale)

        # grad descent step
        new_params = matrix_add(params, scale_matrix(-1 * self.learning_rate, grad_scaled))

        # update parameter maintaining location in memory
        self.update(params, new_params)

        # save moment
        self.second_moment[id(params)] = s_tplus1
