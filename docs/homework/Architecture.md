# CNN Architecture

The model consists of multiple convolutional layers followed by pooling operations and fully connected layers.

```
Input Image

↓

Conv2D

↓

ReLU

↓

MaxPool

↓

Conv2D

↓

ReLU

↓

MaxPool

↓

Flatten

↓

Fully Connected

↓

Dropout

↓

Output (4 Classes)
```

## Design Decisions

- ReLU for non-linearity
- MaxPooling for feature reduction
- Dropout to reduce overfitting
- Fully connected classifier for final prediction

Include a diagram if available:

![](assets/cnn_architecture.png)
