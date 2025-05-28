# CIFAR-10 Image Classification with Custom ResNet-14

[![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.0-blue)](https://pytorch.org/) 
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight implementation of ResNet-14 for CIFAR-10 image classification, achieving **85.32%** test accuracy with minimal parameters and training optimizations.

## 🎯 Project Overview

This project implements a custom 14-layer Residual Network (ResNet-14) from scratch using PyTorch for the CIFAR-10 dataset. The model demonstrates the effectiveness of residual connections in training deeper networks while maintaining computational efficiency.

### Key Features

- **Custom ResNet-14 Architecture**: Built from scratch with basic residual blocks
- **Dual Training Approaches**: Comparison between basic SGD and optimized Adam training
- **Data Augmentation**: Random cropping and horizontal flipping for improved generalization
- **TensorBoard Integration**: Real-time monitoring of training metrics and sample images
- **Lightweight Design**: ~0.27M parameters, suitable for resource-constrained environments

## 🏗️ Architecture Details

### ResNet-14 Structure
```
Input (3x32x32)
    ↓
Conv2d(3→16, 3x3) + BatchNorm + ReLU
    ↓
Layer 1: 2 × BasicBlock(16→16)
    ↓
Layer 2: 2 × BasicBlock(16→32, stride=2)
    ↓
Layer 3: 2 × BasicBlock(32→64, stride=2)
    ↓
AvgPool(8x8) → FC(64→10)
```

### BasicBlock Components
- Two 3×3 convolutional layers
- Batch normalization after each convolution
- ReLU activation functions
- Skip connections with dimension matching
- Optional 1×1 convolution for downsampling

## 📊 Results

### Training Comparison

| Method | Final Test Accuracy | Training Strategy | Key Features |
|--------|-------------------|------------------|--------------|
| **Basic SGD** | 72.00% | SGD (lr=0.01), 30 epochs | Simple normalization only |
| **Optimized Adam** | **85.32%** | Adam + scheduler + augmentation | Data augmentation, warmup, decay |

### Training Curves
The optimized approach shows:
- Faster convergence in early epochs
- Better generalization gap management
- Stable learning after epoch 20
- Consistent improvement with data augmentation

## 🚀 Quick Start

### Prerequisites
```bash
# Required packages
torch>=1.9.0
torchvision>=0.10.0
tensorboard>=2.7.0
```

### Installation
```bash
git clone https://github.com/guraseesrai/cifar10-resnet14-classifier.git
cd cifar10-resnet14-classifier
pip install -r requirements.txt
```

### Basic Usage
```python
# Run the Jupyter notebook
jupyter notebook CIFAR_10_Image_Classification_with_Custom_ResNet_14.ipynb

# Or extract the model class for standalone use:
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Initialize ResNet-14 model
def ResNet14():
    return ResNet(BasicBlock, [2, 2, 2])
```

## 🔧 Training Configuration

### Optimized Training Setup
```python
# Data augmentation for training
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
])

# Optimizer with weight decay
optimizer = torch.optim.Adam(model.parameters(), 
                            lr=1e-3, weight_decay=0.0001)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                           step_size=10, gamma=0.1)
```

### Training Features
- **Warmup**: Lower learning rate (1e-4) for first epoch
- **Data Augmentation**: Random crop with padding=4, horizontal flip
- **Normalization**: Channel-wise normalization with CIFAR-10 statistics
- **Regularization**: Weight decay (0.0001) for better generalization
- **Scheduling**: StepLR with decay every 10 epochs

## 📈 Monitoring with TensorBoard

The project includes comprehensive TensorBoard logging:

```bash
tensorboard --logdir runs
```

**Tracked Metrics:**
- Training and validation loss
- Training and validation accuracy
- Sample training images
- Learning rate schedule

**Available Runs:**
- `resnet14_experiment`: Basic SGD training
- `resnet14_tuned_adam`: Optimized Adam training

## 🧪 Experimental Results

### Performance Analysis
- **Parameter Efficiency**: Achieves 85%+ accuracy with minimal parameters
- **Training Stability**: Consistent convergence without overfitting
- **Data Augmentation Impact**: ~13% accuracy improvement over baseline
- **Optimization Benefits**: Adam + scheduling outperforms basic SGD

### Key Observations
1. **Residual connections** enable training of deeper networks effectively
2. **Data augmentation** significantly improves generalization
3. **Proper normalization** and **learning rate scheduling** are crucial
4. **Warmup strategy** helps stabilize early training phases

## 🔍 Code Structure

```
├── CIFAR_10_Image_Classification_with_Custom_ResNet_14.ipynb  # Main notebook
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── runs/                # TensorBoard logs (generated during training)
    ├── resnet14_experiment/
    └── resnet14_tuned_adam/
```

## 🛠️ Customization Options

### Model Architecture
- Modify `num_blocks` in ResNet constructor for different depths
- Adjust `expansion` factor for wider networks
- Change initial channel count for capacity tuning

### Training Parameters
- **Batch size**: 128 (adjustable based on memory)
- **Learning rate**: 1e-3 with Adam, 0.01 with SGD
- **Epochs**: 30 (sufficient for convergence)
- **Weight decay**: 0.0001 for regularization

## 📚 CIFAR-10 Dataset Info

- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Images**: 60,000 (50K train, 10K test)
- **Resolution**: 32×32 RGB
- **Challenges**: Low resolution, diverse intra-class variation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Implementing other ResNet variants (ResNet-18, ResNet-34)
- Adding mixed precision training
- Exploring different data augmentation strategies
- Implementing model ensemble techniques


## 🙏 Acknowledgments

- Original ResNet paper: "Deep Residual Learning for Image Recognition" by He et al.
- CIFAR-10 dataset by Alex Krizhevsky
- PyTorch team for the excellent deep learning framework

---

*Built with ❤️ using PyTorch*
