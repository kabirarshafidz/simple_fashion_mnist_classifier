
# Simple Fashion MNIST Classifier

This repository contains a Jupyter Notebook that implements a simple deep learning model for classifying images in the Fashion MNIST dataset.

## Overview

The Fashion MNIST dataset is a collection of grayscale images of clothing items (e.g., shirts, trousers, shoes), with 10 different categories. This notebook demonstrates the following steps:
- Loading and preprocessing the dataset
- Building a neural network using a popular deep learning framework
- Training the model and evaluating its performance
- Visualizing the results

## Key Features
- Data preprocessing techniques such as normalization and one-hot encoding
- Model architecture includes a simple feedforward neural network with activation functions
- Utilizes a learning rate finder to select an optimal learning rate
- Tracks model performance using metrics like accuracy and loss

## Prerequisites

- Python 3.7 or above
- Jupyter Notebook
- Required libraries: TensorFlow, NumPy, Matplotlib

## How to Use

1. Clone this repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. Install the required dependencies:
    ```bash
    pip install tensorflow numpy matplotlib
    ```

3. Open the notebook in Jupyter:
    ```bash
    jupyter notebook simple_fashion_mnist_classifier.ipynb
    ```

4. Run the notebook step-by-step to train and evaluate the model.

## Results

The notebook includes:
- A visualization of the learning rate finder to identify the optimal learning rate.
- A plot of training and validation accuracy/loss over epochs.
- Predictions on test data with example outputs.

## Example Output

### Learning Rate vs Loss
![Learning Rate vs Loss](example_plot.png)

### Sample Predictions
| Image           | Predicted Label | True Label |
|------------------|-----------------|------------|
| ![img1](img1.png) | Sneaker         | Sneaker    |
| ![img2](img2.png) | Pullover        | Pullover   |

## Notes
- The chosen learning rate based on the learning rate finder is `0.0018`.
- Feel free to experiment with the model architecture and hyperparameters.

## License
This project is licensed under the MIT License.
