"""
nn_pytorch/model.py

Three-layer neural network (two hidden layers) implemented in PyTorch.
 - Hidden layer 1: Linear → ReLU
 - Hidden layer 2: Linear → ReLU
 - Output layer: Linear (logits) → CrossEntropyLoss

Wraps the model to conform to the Classifier API with fit()/predict().
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from perceptron.base import Classifier

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        """
        Define the network architecture.

        Args:
            input_dim   (int): size of input feature vector
            hidden_dim1 (int): units in first hidden layer
            hidden_dim2 (int): units in second hidden layer
            output_dim  (int): number of classes / output units
        """
        super().__init__()
        # First hidden layer
        self.fc1   = nn.Linear(input_dim,  hidden_dim1)
        self.relu1 = nn.ReLU()
        # Second hidden layer
        self.fc2   = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        # Output layer (raw logits)
        self.fc3   = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: raw logits of shape (batch_size, output_dim)
        """
        x = self.fc1(x)    # Linear transform 1
        x = self.relu1(x)  # ReLU activation 1
        x = self.fc2(x)    # Linear transform 2
        x = self.relu2(x)  # ReLU activation 2
        x = self.fc3(x)    # Final linear (logits)
        return x

class PyTorchNN(Classifier):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim,
                 lr=0.01, epochs=10, batch_size=32):
        """
        Wrapper for the PyTorch Net, matching the Classifier API.

        Args:
            input_dim    (int): number of input features
            hidden_dim1  (int): size of first hidden layer
            hidden_dim2  (int): size of second hidden layer
            output_dim   (int): number of output classes
            lr           (float): learning rate for optimizer
            epochs       (int): number of training epochs
            batch_size   (int): mini-batch size for DataLoader
        """
        # Initialize model, loss function, and optimizer
        self.model      = Net(input_dim, hidden_dim1, hidden_dim2, output_dim)
        self.criterion  = nn.CrossEntropyLoss()  # includes softmax
        self.optimizer  = optim.SGD(self.model.parameters(), lr=lr)
        self.epochs     = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        """
        Train the PyTorch model on NumPy data X, y.

        Args:
            X (ndarray): shape (n_samples, input_dim)
            y (ndarray): shape (n_samples,), integer labels
        """
        # Convert NumPy arrays to torch tensors
        X_t = torch.from_numpy(X).float()  # features as floats
        y_t = torch.from_numpy(y).long()   # labels as longs

        # Wrap in TensorDataset and DataLoader for batching
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader  = torch.utils.data.DataLoader(dataset,
                                              batch_size=self.batch_size,
                                              shuffle=True)

        self.model.train()  # set model to training mode
        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()         # reset gradients
                logits = self.model(X_batch)       # forward pass
                loss   = self.criterion(logits, y_batch)  # compute loss
                loss.backward()                    # backpropagate
                self.optimizer.step()              # update parameters

    def predict(self, X):
        """
        Perform inference on NumPy data X and return predictions.

        Args:
            X (ndarray): shape (n_samples, input_dim)

        Returns:
            ndarray: predicted class labels, shape (n_samples,)
        """
        self.model.eval()  # set model to evaluation mode
        with torch.no_grad():  # disable gradient tracking
            # Convert to tensor and do forward pass
            X_t    = torch.from_numpy(X).float()
            logits = self.model(X_t)
            preds  = torch.argmax(logits, dim=1)  # pick highest logit
        return preds.numpy()
