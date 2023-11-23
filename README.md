# Introduction

CARNet is a modified version of ResNet. The modified code is based on Resnet18. By designing a lightweight attention module to recalibrate features, CARNet reduces 94.2% parameters (from 11.2M to 0.65M), 95% computations (from 32.6 GFLOPs to 1.63 GFLOPs), and 94.2% model size (from 618MB to 360MB) compared to original ResNet, while maintaining comparable accuracy on the same dataset.

## Good Preference for Limited Resources

In previous experiments, training the standard ResNet18 on the dataset repeatedly resulted in top-1 accuracies consistently above 77.73%, averaging 79.14% after 50 epochs. While ResNet displayed excellent generalization and high accuracy, its large model size demanded substantial computational resources, making training challenging without high-end hardware. Initial attempts to train ResNet on a laptop with 24GB RAM and an 8GB RTX 4060 GPU were unsuccessful due to its high computational demand, causing multiple crashes. We then switched to a more powerful desktop with 32GB RAM and an 11GB RTX 2080Ti GPU, which still experienced high utilization rates above 85%.

In contrast, our modified CRANet, with reduced parameters and computational needs, trained effortlessly on the same laptop setup, with GPU usage dropping to around 80%. On a workstation with a 2080Ti GPU, CRANet training for 50 epochs took under 1.7 hours and achieved a stable top-1 accuracy of over 81%, averaging 81.8%, surpassing traditional ResNet. These findings demonstrate CRANet's ability to deliver comparable accuracy with significantly lower computational requirements and training time, offering a major advantage for deployment on edge devices with limited resources.

![Fig1](https://github.com/Fai-yong/CARNet/blob/main/Accuracy-time.png)

## How to use

The image dataset is from this Kaggle link: [Kaggle Dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification)

- `SportsClassify.py` -- The traditional ResNet18 was used to classify the dataset
- `CARNet.py` -- Using CARNet to classify the dataset
- `ResNet18.txt` -- Detailed information of the ResNet18 model (various layers, params, sizes)
- `CARNet.txt` -- Detailed information of the CARNet model (various layers, params, sizes)

## Why to propose CARNet?

In the original ResNet architecture, the BasicBlock module is used as the fundamental building unit of the residual network. However, we identify two issues with the BasicBlock design:

1. The convolutional layers in BasicBlock lead to a large number of parameters, increasing model complexity.
2. The simplicity of the BasicBlock structure limits the representation power of ResNet for complex tasks.

To overcome these limitations, we propose to substitute the BasicBlock with a customized module named GroupConvBlock in our improved ResNet. As shown in Fig2, the GroupConvBlock contains the following components:

![Fig2](https://github.com/Fai-yong/CARNet/blob/main/Components.svg)

- **Grouped Convolution Layer (conv1):** Instead of regular convolution, we adopt grouped convolution in conv1, which divides the input channels into groups and performs convolution only within each group. This design significantly reduces parameter size as the connection between channels is sparse. For example, with 256 input channels divided into 64 groups, the parameter size is decreased to 1/64 of the original.
- **Batch Normalization (bn1):** This is applied after conv1 to normalize activations and stabilize training.
- **Attention Module (attn):** We incorporate a channel-wise attention module, which emphasizes informative features by multiplying the input with an attention mask. The mask is computed by global average pooling to capture contextual information, followed by two convolutional layers to generate the attention weights through a sigmoid activation.
- **1x1 Convolution (conv2):** At the end of GroupConvBlock, a 1x1 convolution adjusts channel size to match residual connections. It involves minimal parameters due to 1x1 kernels.
