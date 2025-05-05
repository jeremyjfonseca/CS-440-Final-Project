"""
nn_scratch/nn.py

Simple three-layer neural network built from scratch using NumPy,
with **two hidden layers** (hidden_dim1 and hidden_dim2) and a softmax output.
Implements batch gradient descent and tanh activations in the hidden layers.
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
        Randomly initialize weight matrices and zero biases for all layers:
          - W1, b1: input → hidden1
          - W2, b2: hidden1 → hidden2
          - W3, b3: hidden2 → output
        """
        self.W1 = np.random.randn(self.input_dim,  self.hidden_dim1) * 0.01
        self.b1 = np.zeros((1, self.hidden_dim1))
        self.W2 = np.random.randn(self.hidden_dim1, self.hidden_dim2) * 0.01
        self.b2 = np.zeros((1, self.hidden_dim2))
        self.W3 = np.random.randn(self.hidden_dim2, self.output_dim) * 0.01
        self.b3 = np.zeros((1, self.output_dim))

    def softmax(self, Z):
        """
        Softmax activation function for multi-class output.

        Args:
            Z (ndarray): pre-activation array of shape (n_samples, output_dim)

        Returns:
            ndarray: softmax probabilities of same shape
        """
        # subtract max for numerical stability
        exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def fit(self, X, y):
        """
        Train the network on data X and labels y with batch gradient descent.

        Args:
            X (ndarray): shape (n_samples, input_dim)
            y (ndarray): shape (n_samples,), integer class labels 0..output_dim-1
        """
        n = X.shape[0]  # number of training samples

        # One-hot encode labels into shape (n_samples, output_dim)
        Y = np.eye(self.output_dim)[y]

        # Initialize all weights and biases
        self.initialize_weights()

        for epoch in range(self.epochs):
            # ---- Forward Pass ----
            Z1 = X.dot(self.W1) + self.b1            # (n, hidden_dim1)
            A1 = np.tanh(Z1)                         # (n, hidden_dim1)
            Z2 = A1.dot(self.W2) + self.b2           # (n, hidden_dim2)
            A2 = np.tanh(Z2)                         # (n, hidden_dim2)
            Z3 = A2.dot(self.W3) + self.b3           # (n, output_dim)
            A3 = self.softmax(Z3)                    # (n, output_dim)

            # ---- Backward Pass ----
            # Gradient of softmax + cross-entropy: dZ3 = A3 - Y
            dZ3 = A3 - Y                             # (n, output_dim)
            dW3 = (A2.T.dot(dZ3)) / n                # (hidden_dim2, output_dim)
            db3 = np.sum(dZ3, axis=0, keepdims=True) / n

            # Backpropagate into second hidden layer
            dA2 = dZ3.dot(self.W3.T)                 # (n, hidden_dim2)
            dZ2 = dA2 * (1 - A2**2)                  # tanh' = 1 - tanh^2
            dW2 = (A1.T.dot(dZ2)) / n                # (hidden_dim1, hidden_dim2)
            db2 = np.sum(dZ2, axis=0, keepdims=True) / n

            # Backpropagate into first hidden layer
            dA1 = dZ2.dot(self.W2.T)                 # (n, hidden_dim1)
            dZ1 = dA1 * (1 - A1**2)                  # (n, hidden_dim1)
            dW1 = (X.T.dot(dZ1)) / n                 # (input_dim, hidden_dim1)
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
        Perform a forward pass through the trained network and return class indices.

        Args:
            X (ndarray): shape (n_samples, input_dim)

        Returns:
            ndarray: predicted labels, shape (n_samples,)
        """
        Z1 = X.dot(self.W1) + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = np.tanh(Z2)
        Z3 = A2.dot(self.W3) + self.b3
        A3 = self.softmax(Z3)
        return np.argmax(A3, axis=1)
