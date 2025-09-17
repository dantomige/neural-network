import random

class Percepton:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = None
        self.bias = None

    def initialize_params(self):
        self.weights = [random.uniform(-5, 5) for _ in range(self.input_dim)] 
        self.bias = random.random()

    def forward(self, input):
        if len(input) != self.input_dim:
            raise ValueError
        
        output = 0
        for weight, input_value in zip(self.weights, input):
            output += weight * input_value

        output += self.bias

        return output

class NeuralNetwork:

    def __init__(self, layer_dims: list[int]):
        self.layer_dims = layer_dims
        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]
        self.layers: list[list[Percepton]] = self._build_neural_network(layer_dims)

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
    output = nn.forward(input)
    print(nn)
    print(input, output)


