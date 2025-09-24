# Neural-Network

Neural Networks from scratch in **pure Python** — no PyTorch, no TensorFlow, no NumPy 😱. This project implements core neural network functionality from the ground up, focusing on clarity and educational value.  

It supports **Fully Connected, Dropout, ReLU, Sigmoid, Tanh, Softmax layers**, as well as **Layer Normalization** and **Batch Normalization**.  

Loss functions include **MSE** and **LogLinear / Cross-Entropy**, and evaluation metrics include **accuracy, precision, recall, and F1-score**.

---

## Features

- **Layer Types**
  - Fully Connected (Dense)  
  - Dropout  
  - ReLU, Tanh, Sigmoid, Softmax   

- **Loss Functions**
  - Mean Squared Error (MSE)  
  - LogLinear / Cross-Entropy  

- **Optimization**
  - SGD, Adam, RMSProp  

- **Evaluation Metrics**
  - Accuracy, Precision, Recall, F1-score  

- **Additional Utilities**
  - Vector and matrix operations implemented manually  
  - Weight initialization: He/Kaiming, Random Normal, Random Uniform  

---

## Demo Applications

- **Housing Price** – Regression example using California housing price data.  
- **Make Moons** – Binary classification example.
- **Wine Identification** – Multi-classification example using winde data.  

---

## Getting Started

### Prerequisites

- Python 3.9+  
- (Optional) `uv` for dependency management  

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/neural-network.git
cd neural-network
```

TODOS:
* update utils
* check styling
* make typing more consistent
* update input types to take in numpy and dataframes array (keep internal logic using the same)
* create full test suite
* fully implement layer and batch norm
