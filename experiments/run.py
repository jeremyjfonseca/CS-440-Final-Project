"""
experiments/run.py

Command-line interface to run classification experiments
on Perceptron, Scratch NN, and PyTorch NN with two hidden layers.
"""

import argparse
import time
import numpy as np

from data_loader.loader import load_data, get_split
from perceptron.model import Perceptron
from nn_scratch.nn import NeuralNetScratch
from nn_pytorch.model import PyTorchNN


def accuracy(y_true, y_pred):
    """
    Compute simple classification accuracy.
    """
    return np.mean(y_true == y_pred)


def main():
    parser = argparse.ArgumentParser(
        description="Run classification experiments with three algorithms"
    )
    parser.add_argument(
        "--algo",
        choices=["perceptron", "scratch", "pytorch"],
        required=True,
        help="Which algorithm to run"
    )
    parser.add_argument(
        "--pct",
        type=int,
        required=True,
        help="Percent of data for training (10–100)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of random trials to average"
    )
    parser.add_argument(
        "--hid1",
        type=int,
        default=100,
        help="Size of the first hidden layer for NNs"
    )
    parser.add_argument(
        "--hid2",
        type=int,
        default=50,
        help="Size of the second hidden layer for NNs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for all models"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for all models"
    )
    args = parser.parse_args()

    # Load datasets; here we're using the digit set as default
    (_, _), (X, y) = load_data()
    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    # Run the specified number of trials
    for run in range(args.runs):
        # Create a pct% train split, rest is test
        X_train, y_train, X_test, y_test = get_split(X, y, args.pct, seed=run)

        # Instantiate the chosen model
        if args.algo == "perceptron":
            model = Perceptron(lr=args.lr, epochs=args.epochs)
        elif args.algo == "scratch":
            model = NeuralNetScratch(
                input_dim=n_features,
                hidden_dim1=args.hid1,
                hidden_dim2=args.hid2,
                output_dim=n_classes,
                lr=args.lr,
                epochs=args.epochs
            )
        elif args.algo == "pytorch":
            model = PyTorchNN(
                input_dim=n_features,
                hidden_dim1=args.hid1,
                hidden_dim2=args.hid2,
                output_dim=n_classes,
                lr=args.lr,
                epochs=args.epochs
            )
        else:
            raise ValueError(f"Unknown algorithm: {args.algo}")

        # Train and measure time
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time

        # Evaluate accuracy
        preds = model.predict(X_test)
        acc = accuracy(y_test, preds)

        # Print a concise summary
        print(f"{args.algo} | pct={args.pct} | run={run} "
              f"| time={elapsed:.3f}s | acc={acc:.3f}")


if __name__ == "__main__":
    main()
