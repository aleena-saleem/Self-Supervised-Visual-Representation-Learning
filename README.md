# Self-Supervised Visual Representation Learning

A PyTorch implementation of **SimCLR** for learning visual representations from images without using labels during representation learning.

The project uses **ResNet-50** as the visual encoder and evaluates the learned representations using linear probing, kNN classification, and t-SNE visualization.
A supervised ResNet-50 baseline is also trained on STL-10 for comparison.

## Project Objectives

- Implement SimCLR-based self-supervised learning
- Learn image representations without labels
- Use contrastive learning with the InfoNCE loss
- Extract high-dimensional visual representations from ResNet-50
- Evaluate learned representations using linear probing and kNN
- Visualize learned feature spaces using t-SNE
- Compare self-supervised representations with a supervised baseline

## Method

The SimCLR pipeline follows:

Image -> Data Augmentation -> ResNet-50 Encoder -> 2048-D Representation -> Projection Head -> 128-D Projection -> InfoNCE Contrastive Loss

The encoder learns useful visual representations by bringing augmented views of the same image closer together while separating representations from different images.

## Dataset

The project uses the **STL-10** image dataset.

- 10 image classes
- 96 × 96 RGB images
- Unlabeled images are used for self-supervised representation learning
- Labeled data is used for downstream evaluation

## Model Architecture

### ResNet-50 Encoder

A pretrained ResNet-50 is used as the image encoder.

The original ImageNet classification layer is removed.

Output:

2048-dimensional visual representation.

### Projection Head

A multilayer projection head maps the encoder representation to a lower-dimensional space used for contrastive learning.

2048 → 512 → 128

The 128-dimensional projection is used by the contrastive loss.

The encoder representation is retained for downstream evaluation.

## Contrastive Learning

Two augmented views are generated from the same image.

These form a positive pair.

Representations from different images act as negative examples.

The InfoNCE loss is used to encourage similar representations for positive pairs and different representations for negative pairs.

## Training Configuration

| Parameter | Value |
|---|---:|
| Dataset | STL-10 |
| Encoder | ResNet-50 |
| Image Size | 96 × 96 |
| Maximum Training Samples | 5,000 |
| Batch Size | 64 |
| Epochs | 100 |
| Learning Rate | 0.0003 |
| Temperature | 0.5 |
| Projection Dimension | 128 |
| Device | CUDA / NVIDIA Tesla T4 |

## Evaluation

The learned representations are evaluated using multiple downstream methods.

### Linear Probe

The ResNet-50 encoder is used as a fixed feature extractor.

A linear classifier is trained on the extracted representations to measure how useful the learned features are for classification.

### kNN Evaluation

Feature vectors are normalized and classified using k-nearest neighbors.

This provides a simple non-parametric evaluation of representation quality.

### t-SNE

t-SNE is used to visualize the learned representations in a lower-dimensional space.

This helps inspect whether images from different classes form meaningful clusters.

## Results Summary

| Method | Accuracy |
|---|---:|
| SimCLR | 84.94% |
| SimCLR kNN | 87.70% |
| Supervised ResNet-50 | 93.04% |


The supervised model provides a baseline for comparing representation learning with conventional supervised training.

The supervised baseline achieves higher classification accuracy, while the SimCLR experiment demonstrates that useful visual representations can be learned without using class labels during representation learning.

## Repository Structure

```text
Self-Supervised-Visual-Representation-Learning/
│
├── models/
│   ├── __init__.py
│   ├── resnet_encoder.py
│   ├── projection_head.py
│   └── simclr.py
│
├── losses/
│   └── info_nce.py
│
├── data_utils/
│   └── simclr_dataset.py
│
├── train/
│   ├── train_ssl.py
│   └── train_supervised.py
│
├── evaluation/
│   ├── linear_probe.py
│   ├── knn.py
│   ├── tsne.py
│   └── metrics.py
│
├── configs/
│   └── simclr_config.py
│
├── results/
│   ├── simclr_training_loss.png
│   ├── simclr_tsne.png
│   ├── knn_results.json
│   ├── results_summary.json
│   └── results_comparison.csv
│
├── checkpoints/
├── report/
├── utils/
├── requirements.txt
├── .gitignore
└── README.md
