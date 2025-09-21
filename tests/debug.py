import random
from neural_network import NeuralNetwork
from layers import FullyConnected, ReLU
from loss_functions import MSE
from optimizer import SGD
from metrics import MSEMetric

# -------------------------
# 1️⃣ Check Initialization
# -------------------------
print("=== Initialization Check ===")
nn = NeuralNetwork([
    FullyConnected(2, 5),
    ReLU(),
    FullyConnected(5, 2)
])
nn.init_params()
print(nn)

# Forward a simple input
X_test = [[1.0, 1.0]]
Y_hat = nn.forward(X_test)
print("Forward output:", Y_hat)

# -------------------------
# 2️⃣ Tiny Dataset Overfit Test
# -------------------------
print("\n=== Tiny Dataset Overfit Test ===")
# 3 points
X_train = [[0.1,0.1],[0.2,0.2],[0.3,0.3]]
Y_train = [[0.2,0.2],[0.4,0.4],[0.6,0.6]]

loss_fn = MSE()
optimizer = SGD(learning_rate=1e-3)
metric = MSEMetric()

for epoch in range(50):
    nn.train(X_train, Y_train, loss_fn, optimizer)
    loss = nn.evaluate(X_train, Y_train, metric)
    print(f"Epoch {epoch}: Loss {loss:.6f}")

# Check predictions
print("Predictions after training:", nn.forward(X_train))

# -------------------------
# 3️⃣ Gradient Norm Check
# -------------------------
print("\n=== Gradient Norm Check ===")
Y_hat = nn.forward(X_train)
nn.backward(X_train, Y_train, Y_hat, loss_fn)

for i, layer in enumerate(nn.layers):
    if hasattr(layer, "pL_pW") and layer.pL_pW is not None:
        grad_norm = sum(sum(abs(v) for v in row) for row in layer.pL_pW)
        print(f"Layer {i} dW norm: {grad_norm:.6f}")
    if hasattr(layer, "pL_pB") and layer.pL_pB is not None:
        grad_norm = sum(sum(abs(v) for v in row) for row in layer.pL_pB)
        print(f"Layer {i} dB norm: {grad_norm:.6f}")