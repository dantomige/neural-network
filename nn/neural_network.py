from utils import matrix_multiply, element_multiply, dot_product, identity, dims
import random
class Percepton:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = None
        self.bias = None
        self.output = None

    def initialize_params(self):
        self.weights = [random.uniform(-1, 1) for _ in range(self.input_dim)] 
        self.bias = random.random()

    def update_parameters(self, weights_grad, bias_grad, learning_rate):
        self.weights = [weight + grad * learning_rate for weight, grad in zip(self.weights, weights_grad)]
        self.bias = self.bias + bias_grad[0] * learning_rate

    def forward(self, input):
        output = dot_product(self.weights, input) + self.bias
        self.output = output
        return output

class NeuralNetwork:

    def __init__(self, layer_dims: list[int]): 

        self.layer_dims = layer_dims
        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]
        self.layers: list[list[Percepton]] = self._build_neural_network(layer_dims)
        self.backprop = None

    def _build_neural_network(self, layer_dims):
        input_size = layer_dims[0]
        layers = []
        for index in range(1, len(layer_dims)):
            layer_dim = layer_dims[index]
            layer = []
            for _ in range(layer_dim):
                layer.append(Percepton(input_dim=input_size))
            layers.append(layer)
            input_size = layer_dim
        return layers
    
    def initalize_params(self):
        for layer in self.layers:
            for perceptron in layer:
                perceptron.initialize_params()

    def forward(self, input):
        for layer in self.layers:
            output = []
            for perceptron in layer:
                output.append(perceptron.forward(input))
            input = output
        
        return output
    
    def backward(self, input, expected_output): # for each layer calculate pOut/pW (input X output dim, inputs stacked), pOut/pIn (input x output dim, weights matrix), pL/pB (1 x output dim, all ones vector)
        
        backprop = []
        pL_pOut = identity(len(expected_output))
        for index in reversed(range(len(self.layers))):
            pOut_pW = []
            pOut_pB = []
            pOut_pIn = []
            layer = self.layers[index]
            if index:
                prev_layer = self.layers[index - 1]
                prev_layer_inputs = [perceptron.output for perceptron in prev_layer] # prev layer out == input
            else:
                prev_layer_inputs = input
            
            for perceptron in layer:
                
                pOut_pW.append(prev_layer_inputs)
                pOut_pB.append([1])
                pOut_pIn.append(perceptron.weights)

            backprop.append(
                {
                    "pL/pW": element_multiply(pL_pOut, pOut_pW),
                    "pL/pB": element_multiply(pL_pOut, pOut_pB)
                    }
                )
            
            # print("dims", dims(pL_pOut), dims(pOut_pW), dims(matrix_multiply(pL_pOut, pOut_pW)))
            # print("dims",dims(pL_pOut), dims(pOut_pB), dims(matrix_multiply(pL_pOut, pOut_pB)))

            pL_pOut = matrix_multiply(pL_pOut, pOut_pIn)

        self.backprop = list(reversed(backprop))
        return self.backprop

    def update_parameters(self, learning_rate):
        for grads, layer in zip(self.backprop, self.layers):
            layer_pL_pW = grads["pL/pW"]
            layer_pL_pB = grads["pL/pB"]

            for perceptron, weights_grad, bias_grad in zip(layer, layer_pL_pW, layer_pL_pB):
                # print("before", perceptron.weights, perceptron.bias)
                perceptron.update_parameters(weights_grad, bias_grad, learning_rate)
                # print("after", perceptron.weights, perceptron.bias)

    def train(self, inputs, outputs, learning_rate, loss_function):
        pass
    
    def __str__(self):     
        output = ""
        for layer in self.layers:
            for perceptron in layer:
                perceptron_weights_strings = [str(weight) for weight in perceptron.weights]
                output += f'{" ".join(perceptron_weights_strings)}    {perceptron.bias}\n'
            output += "\n"
        return output[:-1]
        
            

if __name__ == "__main__":
    nn = NeuralNetwork(layer_dims=[2, 3, 2, 1])
    nn.initalize_params()
    input = [1, 0]
    expected_output = [1]
    output = nn.forward(input)
    nn.backward(input, expected_output)
    nn.update_parameters(learning_rate=0.2)


