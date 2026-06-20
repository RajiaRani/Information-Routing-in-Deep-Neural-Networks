# A Unified Study of Information Routing in Deep Neural Networks
## Dynamic Routing in Capsule Networks vs. Dynamic Attention in Vision Transformers

This repository contains the code, experiments, and visualizations for a study
comparing dynamic routing-by-agreement (Capsule Networks) with dynamic
self-attention (Vision Transformers). The project investigates whether the two
are the same mechanism, and shows that routing coefficients c_ij and attention
weights alpha_ij are instances of a single adaptive information routing operator,
differing only in iteration, compatibility measure, and normalisation axis.

Author: Rajia Rani, Computer Science, University of South Dakota.

## Overview

A classical layer is statically wired (y = Wx + b), so every input follows the
same computational path once trained. Capsule Networks and Transformers break
this by routing information according to the current input. This work derives
both mechanisms from first principles, implements them, and evaluates them on two
datasets of increasing difficulty (MNIST and CIFAR-10) across classification
accuracy, data efficiency, and noise robustness.

## Key Findings

- The Capsule Network outperformed the Vision Transformer on both datasets
  (MNIST: 99.41% vs. 97.18%; CIFAR-10: 63.73% vs. 61.05%).
- Routing was consistently more data-efficient and more noise-robust, and the
  ordering held on both datasets, indicating the advantage is a property of the
  mechanism rather than of MNIST.
- These gains came at roughly 30x the training cost of the Transformer.
- Conclusion: attention is not a strict special case of routing; both are
  configurations of one operator.

## What Is Implemented

System A, Capsule Network:
Input -> Convolution -> Primary Capsules -> Digit Capsules (Dynamic Routing, r=3)
-> Classification.

System B, Vision Transformer:
Patch Embedding -> Positional Encoding -> Self-Attention -> MLP ->
Classification Head.

Experiments (run on MNIST and CIFAR-10):
1. Classification performance (accuracy and training loss).
2. Data efficiency (10, 25, 50, 100 percent of training data).
3. Noise robustness (additive Gaussian noise, sigma = 0.0 to 0.5).

Visualizations:
- Visualization A: routing coefficient heatmaps (c_ij).
- Visualization B: attention maps (alpha_ij) and CLS-token attention over patches.
- Visualization C: side-by-side comparison of routing and attention.

Advanced component (both implemented):
- Idea 1: Attention-Guided Capsule Routing (replaces the iterative routing loop
  with a single query-key attention step).
- Idea 2: Iterative Attention (repeats the Transformer's attention three times).

## Repository Structure

.
├── Capsnet_and_Vit_complete_project.ipynb   Main notebook: models, experiments, visualizations
├── outputs/                                  Generated figures and numeric results
│   ├── exp1_classification.png
│   ├── exp2_data_efficiency.png
│   ├── exp2_data_efficiency_cifar.png
│   ├── exp3_noise.png
│   ├── exp3_noise_cifar.png
│   ├── vizA_routing.png
│   ├── vizA_routing_cifar.png
│   ├── vizB_attention.png
│   ├── vizB_attention_cifar.png
│   ├── vizC_side_by_side.png
│   ├── vizC_side_by_side_cifar.png
│   └── results.json                          All numeric results
└── README.md



## Requirements

- Python 3
- PyTorch and torchvision
- matplotlib
- numpy

Install with:

pip install torch torchvision matplotlib numpy



## How to Run

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
jupyter notebook Capsnet_and_Vit_complete_project.ipynb



Run all cells. Set DATASET = 'mnist' or DATASET = 'cifar10' in the configuration
cell to switch datasets. All figures and results.json are written to the outputs
folder.

## Experimental Setup

Both models share the same training configuration for a fair comparison:

| Setting              | Value                      |
|----------------------|----------------------------|
| Optimiser            | Adam                       |
| Learning rate        | 1e-3                       |
| Batch size           | 128                        |
| Epochs               | 5 (MNIST) / 15 (CIFAR-10)  |
| Routing iterations r | 3                          |
| ViT depth / heads    | 4 / 4                      |
| Embedding dim        | 64                         |

Models are trained on a single GPU with a fixed random seed for the data splits
and weight initialisation; results are reported from a single run per
configuration.

## Results Summary

| Metric             | CapsNet                   | ViT             |
|--------------------|---------------------------|-----------------|
| MNIST accuracy     | 99.41%                    | 97.18%          |
| CIFAR-10 accuracy  | 63.73%                    | 61.05%          |
| Data efficiency    | Leads at every fraction   | Trails          |
| Noise robustness   | Degrades more gracefully  | Degrades faster |
| Training cost      | About 30x slower          | Faster (single pass) |
