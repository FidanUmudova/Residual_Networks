# ResNet50 Implementation from Scratch (Keras / TensorFlow)

This repository contains a complete, from-scratch implementation of the **ResNet50** deep residual network architecture using **TensorFlow/Keras** Functional API. This project was completed as part of the Deep Learning Specialization (Convolutional Neural Networks course).

---

## 🏗️ Project Architecture & Components

The implementation is broken down into modular components as defined in the original ResNet paper:

1. **Identity Block (`identity_block`):** 
   - Used when the input and output dimensions (channels and spatial resolution) of the main path match.
   - Consists of three convolutional layers (1x1, 3x3, 1x1) with batch normalization and ReLU activations, plus a shortcut skip connection.

2. **Convolutional Block (`convolutional_block`):** 
   - Used when the input and output dimensions differ (e.g., changing channel depth or downsampling spatial dimensions using strides).
   - Features a matching convolutional layer on the shortcut path to align dimensions.

3. **ResNet50 Architecture (`ResNet50`):**
   - **Stage 1:** Initial 7x7 Convolution, Batch Normalization, ReLU, and Max Pooling.
   - **Stage 2 to 5:** A sequence of Convolutional Blocks and multiple Identity Blocks scaling up filter depths (`[64, 256]`, `[128, 512]`, `[256, 1024]`, `[512, 2048]`).
   - **Classifier:** Global Average Pooling, Flatten, and a Dense Softmax output layer for multi-class classification.

---

## 💻 Code Snippet (ResNet50 Core)

Here is a quick look at how the main `ResNet50` architecture is structured using the functional API:

```python
def ResNet50(input_shape=(64, 64, 3), classes=6, training=False):
    X_input = Input(input_shape)
    
    # Zero-Padding & Stage 1
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
    X = identity_block(X, 3, [256, 256, 1024])
    # ... (more identity blocks)

    # Stage 5 & Output
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    X = AveragePooling2D()(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)
    
    model = Model(inputs=X_input, outputs=X)
    return model
