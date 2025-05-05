"""
perceptron/model.py

Perceptron implementation from scratch using NumPy.

Implements the classic update rule:
    w <- w + lr * (y_i - ŷ_i) * x_i
with labels internally mapped to ±1.
"""

import numpy as np
from .base import Classifier

class Perceptron(Classifier):
    def __init__(self, lr=1.0, epochs=10):
        """
        Args:
            lr     (float): learning rate
            epochs (int): number of full passes over the data
        """
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        """
        Train the perceptron on X (n_samples×n_features) with labels y (0/1).
        Internally maps y to –1/+1 for the update rule.
        """
        n_samples, n_features = X.shape
        # initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # remap labels {0,1} -> {-1, +1}
        y_mod = np.where(y <= 0, -1, 1)

        for _ in range(self.epochs):
            # iterate through each example
            for xi, target in zip(X, y_mod):
                # compute current prediction (±1)
                raw = np.dot(xi, self.weights) + self.bias
                pred = 1 if raw >= 0 else -1
                # perceptron update
                update = self.lr * (target - pred)
                self.weights += update * xi
                self.bias    += update

    def predict(self, X):
        """
        Predict labels (0/1) for feature matrix X.
        Applies the learned weights and bias with a threshold at 0.
        """
        raw = X.dot(self.weights) + self.bias
        # threshold at zero -> map back to {0,1}
        return np.where(raw >= 0, 1, 0)
