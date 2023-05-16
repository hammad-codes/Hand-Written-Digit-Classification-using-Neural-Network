# Handwritten Digit Classification using Neural Network
This repository contains code for training and evaluating a simple neural network model on the MNIST handwritten digits dataset for digit classification. The neural network is implemented using PyTorch, a popular deep learning framework.

# Dataset
The MNIST dataset is a widely used benchmark dataset in the field of machine learning. It consists of a large number of 28x28 grayscale images of handwritten digits (0-9) along with their corresponding labels.

In this code, the MNIST dataset is automatically downloaded and loaded using the torchvision library. The dataset is split into training and test sets. The images are transformed into tensors and normalized to be in the range [0, 1].

# Neural Network Architecture
The neural network used for digit classification is a simple feedforward neural network with three fully connected layers. The input layer has 784 neurons corresponding to the flattened image size (28x28), the hidden layers have 128 and 64 neurons, respectively, and the output layer has 10 neurons representing the possible digit classes.

# Training
The network is trained using the stochastic gradient descent (SGD) optimizer with a learning rate of 0.01 and momentum of 0.5. The training is performed for a specified number of epochs, with each epoch consisting of multiple iterations over the training data in batches.

During training, the loss is calculated using the cross-entropy loss function, which measures the dissimilarity between the predicted class probabilities and the true labels. Gradients are computed using backpropagation, and the optimizer updates the model's weights accordingly.

# Evaluation
After each epoch of training, the trained network is evaluated on the test dataset to measure its accuracy. The accuracy is calculated as the percentage of correctly classified images out of the total number of test images.

# Usage
To run the code, make sure you have Python installed along with the required dependencies: numpy, torch, torchvision, and matplotlib.

Simply execute the code in a Python environment or run each code block sequentially in Jupyter Notebook.

# Results
After training the network for the specified number of epochs, the code prints the accuracy of the network on the test dataset. In the provided example, an accuracy of 94% was achieved, indicating the model's proficiency in classifying handwritten digits.

Feel free to explore the code and experiment with different network architectures, hyperparameters, and optimization techniques to improve the accuracy further.

# 
Acknowledgements
This code is a basic implementation of a neural network for digit classification and is intended for educational purposes. It utilizes the torchvision library to access the MNIST dataset and PyTorch for building and training the neural network model.

To further enhance your understanding of neural networks, I recommend watching the insightful YouTube playlist "3Blue1Brown Neural Network" by 3Blue1Brown, which provides a visual and intuitive explanation of the underlying concepts.
