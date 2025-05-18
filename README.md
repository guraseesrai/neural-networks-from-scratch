# 🧠 Feedforward Neural Network from Scratch

This project implements a **fully custom neural network** using only **NumPy**, trained on a 2-class CIFAR image dataset. It includes training, evaluation, and hyperparameter tuning without using high-level ML libraries like PyTorch or TensorFlow.

---

## 🚀 Features

- Custom-built components:
  - `LinearLayer`
  - `ReLU` activation
  - `Sigmoid + Cross-Entropy` loss
- Forward and backward pass logic implemented manually
- Mini-batch stochastic gradient descent with:
  - Momentum
  - Weight decay
- Real-time performance monitoring (loss/accuracy plots)
- Hyperparameter tuning experiments for:
  - Batch size
  - Learning rate
  - Hidden layer width

---

## 📊 Dataset

- **Input**: Preprocessed CIFAR-10 subset with 2 classes
- **Format**: Pickled Python dictionary (`cifar_2class_py3.p`)
- **Shape**: Flattened image vectors (e.g., 3072 features)

---

## 📈 Results & Tuning Insights

- Tuned for optimal batch size, learning rate, and network depth
- Final test accuracy achieved: ~81%
- Visualizations show convergence behavior and model performance

---

## 🏷 Tags

`numpy` `neural-networks` `deep-learning` `from-scratch` `backpropagation`  
`feedforward-network` `machine-learning` `python` `cifar`

---

## 🛠 Run Instructions

1. Place the dataset `cifar_2class_py3.p` in your working directory (or adjust the path in the notebook).
2. Run the notebook step-by-step in Google Colab or locally with Jupyter.
