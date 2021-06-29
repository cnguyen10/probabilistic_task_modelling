A **PyTorch** implementation of the paper "[**Probabilistic task modelling for meta-learning**](https://arxiv.org/abs/2106.04802)" published at International Conference on Uncertainty in Artificial Intelligence (UAI) 2021.

## Requirements
- PyTorch 1.0 or above with or without GPU support
- Tensorboard

## Datasets
Two popular datasets: **Omniglot** and **mini-ImageNet** are considered in the paper. Note that the image size on both datasets is set to *64-by-64* pixel<sup>2</sup> to ease the design of VAE architecture.

## Tensorboard
The implementation also has Tensorboard integrated to monitor the training progress.