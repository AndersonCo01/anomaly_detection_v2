# Data Preprocessing

## Training Transformations

Training images receive augmentation to improve generalization.

- Resize
- Random Horizontal Flip
- Random Rotation
- Random Crop
- Tensor Conversion
- Normalization

### Why Data Augmentation?

Augmentation exposes the model to different versions of the same image, helping prevent overfitting and improving robustness.

## Validation & Test

Validation and testing only include:

- Resize
- Tensor conversion
- Normalization

No random augmentations are used during evaluation.
