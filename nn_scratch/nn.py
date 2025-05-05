"""
nn_scratch/nn.py

A simple three-layer neural network built from scratch using NumPy,
with **two hidden layers** (hidden_dim1 and hidden_dim2) and a sigmoid output.
Implements batch gradient descent and a tanh activation in hidden layers.
"""

import numpy as np
from perceptron.base import Classifier

class NeuralNetScratch(Classifier):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim,
                 lr=0.01, epochs=10):
        """
        Initialize network hyperparameters and architecture dimensions.

        Args:
            input_dim    (int): number of features in input layer
            hidden_dim1  (int): number of units in first hidden layer
            hidden_dim2  (int): number of units in second hidden layer
            output_dim   (int): number of classes / output units
            lr           (float): learning rate for gradient descent
            epochs       (int): number of full passes over the data
        """
        # Store dimensions and training settings
        self.input_dim   = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim  = output_dim
        self.lr          = lr
        self.epochs      = epochs
        # We will initialize weights/biases in initialize_weights()
    
    def initialize_weights(self):
        """
        Randomly initialize weight matrices and zero biases for:
          - W1, b1: input -> hidden1
          - W2, b2: hidden1 -> hidden2
          - W3, b3: hidden2 -> output
        """
        self.W1 = np.random.randn(self.input_dim,  self.hidden_dim1) * 0.01
        self.b1 = np.zeros((1, self.hidden_dim1))
        self.W2 = np.random.randn(self.hidden_dim1, self.hidden_dim2) * 0.01
        self.b2 = np.zeros((1, self.hidden_dim2))
        self.W3 = np.random.randn(self.hidden_dim2, self.output_dim) * 0.01
        self.b3 = np.zeros((1, self.output_dim))

    def sigmoid(self, z):
        """
        Sigmoid activation function.

        Args:
            z (ndarray): pre-activation input

        Returns:
            ndarray: sigmoid(z)
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the network on data X and labels y using batch gradient descent.

        Args:
            X (ndarray): shape (n_samples, input_dim)
            y (ndarray): shape (n_samples,), integer class labels 0..output_dim-1
        """
        n = X.shape[0]  # number of training samples

        # One-hot encode labels into shape (n_samples, output_dim)
        Y = np.eye(self.output_dim)[y]

        # Initialize learnable parameters
        self.initialize_weights()

        # Perform 'epochs' passes over the entire dataset
        for epoch in range(self.epochs):
            # ---- Forward Pass ----
            # Layer 1 pre-activation: Z1 = X·W1 + b1
            Z1 = X.dot(self.W1) + self.b1            # shape (n, hidden_dim1)
            # Layer 1 activation: A1 = tanh(Z1)
            A1 = np.tanh(Z1)                         # shape (n, hidden_dim1)
            # Layer 2 pre-activation: Z2 = A1·W2 + b2
            Z2 = A1.dot(self.W2) + self.b2           # shape (n, hidden_dim2)
            # Layer 2 activation: A2 = tanh(Z2)
            A2 = np.tanh(Z2)                         # shape (n, hidden_dim2)
            # Output layer pre-activation: Z3 = A2·W3 + b3
            Z3 = A2.dot(self.W3) + self.b3           # shape (n, output_dim)
            # Output activation: A3 = sigmoid(Z3)
            A3 = self.sigmoid(Z3)                    # shape (n, output_dim)

            # ---- Backward Pass ----
            # dZ3: gradient of loss w.r.t. Z3
            dZ3 = A3 - Y                             # shape (n, output_dim)
            # Gradients for W3, b3
            dW3 = (A2.T.dot(dZ3)) / n                # shape (hidden_dim2, output_dim)
            db3 = np.sum(dZ3, axis=0, keepdims=True) / n

            # Backpropagate into second hidden layer
            dA2 = dZ3.dot(self.W3.T)                 # shape (n, hidden_dim2)
            # derivative of tanh is (1 - tanh^2)
            dZ2 = dA2 * (1 - A2**2)                  # shape (n, hidden_dim2)
            dW2 = (A1.T.dot(dZ2)) / n                # shape (hidden_dim1, hidden_dim2)
            db2 = np.sum(dZ2, axis=0, keepdims=True) / n

            # Backpropagate into first hidden layer
            dA1 = dZ2.dot(self.W2.T)                 # shape (n, hidden_dim1)
            dZ1 = dA1 * (1 - A1**2)                  # shape (n, hidden_dim1)
            dW1 = (X.T.dot(dZ1)) / n                 # shape (input_dim, hidden_dim1)
            db1 = np.sum(dZ1, axis=0, keepdims=True) / n

            # ---- Parameter Updates ----
            self.W3 -= self.lr * dW3
            self.b3 -= self.lr * db3
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

    def predict(self, X):
        """
        Perform a forward pass and return predicted class indices.

        Args:
            X (ndarray): shape (n_samples, input_dim)

        Returns:
            ndarray: shape (n_samples,), predicted labels 0..output_dim-1
        """
        # Forward propagation through all layers
        Z1 = X.dot(self.W1) + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = np.tanh(Z2)
        Z3 = A2.dot(self.W3) + self.b3
        A3 = self.sigmoid(Z3)
        # Choose the class with highest activation
        return np.argmax(A3, axis=1)
