"""
perceptron/base.py

Defines an abstract Classifier interface that all models must follow.
This ensures a consistent fit()/predict() API across implementations.
"""

class Classifier:
    def fit(self, X, y):
        """
        Train the model on features X and labels y.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("fit() must be implemented by subclass")

    def predict(self, X):
        """
        Predict labels for the given feature matrix X.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("predict() must be implemented by subclass")
