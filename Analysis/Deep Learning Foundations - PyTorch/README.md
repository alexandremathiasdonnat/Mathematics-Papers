# Deep Learning Foundations with PyTorch

**Tensors, Autograd, Neural Networks, Optimization & Generalization**

## 1. About

This notebook provides a structured and hands-on introduction to the foundations of deep learning using PyTorch, with an emphasis on understanding how neural networks are built, trained, and evaluated rather than treating them as black-box models.

The objective is to bridge the gap between mathematical concepts (linear algebra, calculus, optimization) and their concrete implementation in modern deep learning frameworks.

Rather than relying immediately on high-level abstractions, the notebook progressively builds intuition by starting from low-level tensor operations and manual optimization loops before moving to full neural network training.

**We explore:**

- Tensor operations, broadcasting, and memory views
- Automatic differentiation and dynamic computational graphs
- Gradient-based optimization from scratch
- Custom data pipelines with `Dataset` and `DataLoader`
- Neural network modeling with `nn.Sequential` and `nn.Module`
- Training loops, loss functions, and optimizers
- Model capacity, activation functions, and generalization

Here we combines conceptual explanations with empirical experiments to provide a solid foundation for further work in deep learning and neural network modeling.

## 2. Learning Problem Setup

We consider supervised learning problems in a regression setting.

### Regression

Observations are pairs:

$$(X_i, Y_i), \quad Y_i \in \mathbb{R}$$

Models aim to approximate an unknown function $f$ such that:

$$\hat{y} = f_\theta(x)$$

Training is performed by minimizing a Mean Squared Error (MSE) loss:

$$L(\theta) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

Both linear and non-linear models are studied, using synthetic data with controlled noise.

## 3. Tensors and Automatic Differentiation

We begin with PyTorch **Tensors**, the core data structure of deep learning.

**Concepts covered include:**

- Tensor creation and device placement (CPU/GPU)
- Broadcasting and vectorized operations
- Tensor views vs memory copies
- Gradient tracking with `requires_grad`
- Backpropagation using `autograd`

A simple analytical example illustrates how PyTorch builds a dynamic computational graph and computes gradients automatically using the chain rule.

## 4. Linear Regression from Scratch with Autograd

To demystify neural network training, we implement linear regression from scratch using PyTorch's automatic differentiation:

$$\hat{y} = Wx + b$$

**We explicitly code:**

- forward pass
- MSE loss computation
- backpropagation via `loss.backward()`
- gradient descent updates
- gradient resetting

This section highlights how modern deep learning optimizers are built on top of basic gradient descent mechanics.

## 5. Custom Data Pipelines

We introduce PyTorch's data abstraction layer through:

- a custom `Dataset` generating noisy sine data
- a `DataLoader` handling batching and shuffling

This separation clarifies the distinction between:

- **what the data is** (`Dataset`)
- **how it is accessed during training** (`DataLoader`)

This design mirrors real-world deep learning workflows.

## 6. Neural Network Modeling in PyTorch

Two modeling paradigms are explored:

### `nn.Sequential`

- Fast and concise definition of feed-forward architectures
- Suitable for simple stacked models

### `nn.Module`

- Full control over architecture and data flow
- Standard approach for research and production models

A multilayer perceptron (MLP) with two hidden layers is used to approximate a non-linear sine function.

## 7. Training and Optimization Loop

A full training loop is implemented using:

- Mean Squared Error loss (`nn.MSELoss`)
- Adam optimizer (`torch.optim.Adam`)

**The standard five-step training choreography is emphasized:**

1. Zero gradients
2. Forward pass
3. Loss computation
4. Backpropagation
5. Parameter update

Training dynamics are monitored through loss curves and model predictions.

## 8. Model Evaluation and Generalization

Model performance is assessed using:

- convergence plots (training loss)
- visual comparison between predictions and data

A train/validation split is introduced to study generalization behavior.

The comparison of training and validation loss highlights whether the model overfits or generalizes well.

## 9. Empirical Studies (Next Steps)

Several controlled experiments are conducted:

### Model Capacity

Reducing the hidden dimension illustrates underfitting and the role of model expressiveness.

### Activation Functions

Tanh, ReLU, and Sigmoid activations are compared in terms of:

- convergence speed
- training dynamics
- quality of the learned function

### Generalization

Training and validation losses are compared for the baseline model to assess overfitting.

These experiments show how architectural and optimization choices affect learning behavior.

## Core takeaways

- **Tensors** are the fundamental abstraction behind deep learning computations.
- **Automatic differentiation** enables scalable gradient-based optimization.
- **Neural network training** is an extension of basic gradient descent.
- **Data pipelines** are as important as model architectures.
- **Model capacity** controls underfitting and expressiveness.
- **Activation functions** mainly affect training dynamics and smoothness.
- **Validation curves** are essential to assess generalization.
- **Empirical observation** is critical to understanding deep learning behavior.

## Dependencies

- `numpy`
- `matplotlib`
- `torch` (PyTorch)

---

***Alexandre Mathias DONNAT, Sr***
