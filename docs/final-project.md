# Final Project: Steel Surface Defect Classification

## Project overview

This project implements an end-to-end, five-class computer vision pipeline built to inspect steel surfaces. It covers everything from raw image preprocessing and custom dataset splitting to training a convolutional neural network, tracking metrics, generating Grad-CAM visualizations, and deploying an interactive Streamlit app.

The model classifies surfaces into five categories: no_defect, defect_1 (crack), defect_2 (pitting), defect_3 (scratch), and defect_4 (corrosion).

## Data preprocessing

To keep inputs consistent across the board, all images are resized to 256 × 256 pixels and normalized using ImageNet's mean and standard deviation. During training, I apply random horizontal flips and adjustments to brightness and contrast to help the model generalize. Validation, testing, and inference rely strictly on deterministic transforms so that every evaluation metric remains reliable and repeatable.

## Dataset and splits

Since real industrial data isn't always readily available, I built a generator that produces 300 synthetic RGB images (60 balanced examples per class). Using a fixed random seed and stratified sampling, the dataset splits cleanly into roughly 70% training, 15% validation, and 15% testing (resulting in 209, 45, and 46 images, respectively).

## Model architecture

`SteelCNN` contains three convolution blocks. Each block uses a 3 × 3 convolution,
batch normalization, ReLU, and 2 × 2 max pooling. Channel depth increases from 3
to 32, 64, and 128. Adaptive average pooling produces a fixed 128-element feature
vector. The classifier uses a 128 → 64 linear layer, ReLU, 30% dropout, and a final
five-logit output layer.

The model has **102,277 trainable parameters**. It returns raw logits because
`CrossEntropyLoss` performs the required log-softmax internally.

## Training

The training loop follows standard PyTorch best practices: clearing gradients, running the forward pass, computing loss, backpropagation, and updating weights via the optimizer. Validation runs in evaluation mode with gradient tracking disabled.

The baseline configuration uses:
Loss function: Cross-entropy
Optimizer: Adam
Learning rate: 0.001
Batch size: 32
Checkpointing: Saves the model only when validation accuracy improves

Run training with:

```bash
python -m steel_defect.train --epochs 20 --batch_size 32 --lr 0.001
```

## Results and inference

AAfter a verified 10-epoch CPU training run, the model hit 95.6% validation accuracy and 93.5% held-out test accuracy (correctly classifying 43 out of 46 images). Cracks proved to be the toughest category at 66.7% recall, while pitting and corrosion both hit 100% on this synthetic split.

## Quality verification

The course suite exercises transforms, RGB conversion, scanning, stratification,
model shape and gradient flow, optimizer behavior, validation immutability,
checkpoint loading, and probability outputs. The completed project passes all
**88 tests**. 


## Challenges and learnings

Version Control & Repository Structure: Mastered GitHub workflows and best practices for proper repository organization and structure.
Parameter & Variable Management: Learned to implement and configure new parameters and variables while understanding their functional behavior within the codebase.
Model Training & Execution: Gained hands-on experience in training, configuring, and executing machine learning models.

Looking ahead, the natural next step is moving past synthetic data. Transitioning this to a real-world production line would require gathering actual images from the target facility, collaborating with subject-matter experts to define robust labels, preventing data leakage across splits, and tuning the model to handle harsh industrial lighting and material variations.
