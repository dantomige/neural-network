from nn import *
import random

random.seed(1)

# Tiny dataset (3 samples, 2 features)
X = [
    [1, 0],
    [0, 1],
    [1, 1]
]

# Target outputs (2 outputs per sample)
Y = [
    [1, 0],
    [0, 1],
    [1, 1]
]

# Build a tiny neural network
nn = NeuralNetwork([
    FullyConnected(input_dim=2, output_dim=5),
    ReLU(),
    FullyConnected(input_dim=5, output_dim=2)
])

# Initialize weights
nn.init_params()

# Optimizer
optimizer = SGD(learning_rate=0.1)

# Loss function
loss_fn = MSE()

Yhat_start = nn.forward(X)
print("Predictions before training:", Yhat_start)

# Training loop
print("=== Tiny Dataset Overfit Test ===")
for epoch in range(2500):
    nn.train(X, Y, loss_fn, optimizer)
    Yhat = nn.forward(X)
    loss = loss_fn.loss(Y, Yhat)
    print(f"Epoch {epoch}: Loss {loss:.6f}")

# Predictions after training
Yhat_final = nn.forward(X)
print("Predictions after training:", Yhat_final)
