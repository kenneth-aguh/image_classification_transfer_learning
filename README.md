# image_classification_transfer_learning
```markdown
# Image Classification with MobileNetV2 Transfer Learning

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Project Overview

This project adapts a pre-trained MobileNetV2 model for classifying images within a custom dataset. Key steps include data loading, preprocessing, model modification, training, and evaluation. The notebook assumes a specific dataset directory structure.

## Dataset

The dataset should be structured as follows in the `dataset/` directory:

```
dataset/
    class1/
        image1.jpg
        image2.jpg
        ...
    class2/
        image1.jpg
        ...
    ...
    class10/
        image1.jpg
        ...
```

Each subdirectory represents a class. An 80/20 split for training and validation is used.

## Dependencies

- Python 3.6+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- pathlib

Install dependencies:

```
pip install tensorflow numpy matplotlib pathlib
```

## Usage

1.  Clone the repository.
2.  Organize your dataset in the `dataset/` directory.
3.  Run the `image_classification_transfer_learning.ipynb` notebook in Jupyter or Google Colab, executing cells sequentially.

## Model Architecture

*   MobileNetV2 pre-trained on ImageNet is used as the base.
*   The original classification layer is removed (`model.layers.pop()`).
*   A new `Dense` output layer with `softmax` activation for 10 classes is added.
*   All layers of MobileNetV2 except the last 4 are frozen to preserve pre-trained weights.

## Training

*   Batch size: 16
*   Image size: 224x224
*   Optimizer: Adam
*   Loss: Categorical Crossentropy
*   Epochs: 10
*   Validation split: 20%
*   `ImageDataGenerator` is used for data augmentation and normalization (`rescale=1/255`).
*   `ModelCheckpoint` saves the best model to `model_mobilenet.h5`.

## Evaluation

The model is evaluated on the validation set after training. Best weights are loaded from the checkpoint file. Validation loss and accuracy are printed.

## Results

The notebook generates plots showing training/validation accuracy and loss.

Example output:

```
Found 399 images belonging to 10 classes.
Found 96 images belonging to 10 classes.
...
Val loss: 0.xx
Val Accuracy: xx.xx%
```

Plots:

*   Model Accuracy (training and validation)
*   Model Loss (training and validation)
