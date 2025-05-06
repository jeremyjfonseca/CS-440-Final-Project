#!/usr/bin/env python3
"""
experiments/run.py

Command-line interface to run classification experiments
on Perceptron, Scratch NN, and PyTorch NN with two hidden layers.
"""

import os
import sys

# ─── Add the project root to sys.path so imports work from here ─────────────────
# e.g. if this file is at /.../CS-440-Final-Project/experiments/run.py,
# this will insert /.../CS-440-Final-Project into sys.path.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ────────────────────────────────────────────────────────────────────────────────

import argparse
import time
import numpy as np
import pickle

# now these will resolve properly
from data_loader.loader import load_data, get_split
from perceptron.model    import Perceptron
from nn_scratch.nn       import NeuralNetScratch
from nn_pytorch.model    import PyTorchNN


def accuracy(y_true, y_pred):
    """
    Compute simple classification accuracy.
    """
    if y_true.size == 0:
        print("Warning: y_true is empty.")
    
    if y_pred.size == 0:
        print("Warning: y_pred is empty.")

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
        "--dataset",
        choices=["digits", "faces"],
        default="digits",
        help="Dataset to use (digits or faces)"
    )
    parser.add_argument(
        "--pct",
        type=int,
        required=True,
        help="Percent of reserved training data used (10-100)"
    )
    parser.add_argument(
        "--test_pct",
        type=int,
        default=20,
        help="Percent of total data reserved for testing (5-50)"
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
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the trained model from the last run"
    )
    args = parser.parse_args()

    # Load the selected dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "digits":
        (_, _), (X, y) = load_data()
    else:  # faces
        (X, y), (_, _) = load_data()
    
    n_features = X.shape[1]
    n_classes  = len(np.unique(y))
    
    # Print dataset information
    print(f"Dataset: {args.dataset}")
    print(f"Features: {n_features}, Classes: {n_classes}")
    print(f"Total samples: {X.shape[0]}")
    
    # Print the test/train splits 
    n = X.shape[0]
    n_test = int(n * args.test_pct / 100)
    n_train = int((n-n_test) * args.pct / 100)
    print('n =', n, 
          '| n_test =', n_test, '(' , n_test/n*100, 
          '%) | n_train =', n_train, '(' , n_train/n*100, 
          '%) | unused =', n-n_train-n_test, '(' , (n-n_train-n_test)/n*100, 
          '%)')
    del n, n_test, n_train

    # Store results for all runs
    times = []
    accs = []
    last_model = None

    # Run the specified number of trials
    for run in range(args.runs):
        X_train, y_train, X_test, y_test = get_split(X, y, pct=args.pct, test_pct=args.test_pct, seed=run)

        # Instantiate the chosen model
        if args.algo == "perceptron":
            model = Perceptron(lr=args.lr, epochs=args.epochs)
        elif args.algo == "scratch":
            model = NeuralNetScratch(
                input_dim   = n_features,
                hidden_dim1 = args.hid1,
                hidden_dim2 = args.hid2,
                output_dim  = n_classes,
                lr          = args.lr,
                epochs      = args.epochs
            )
        else:  # pytorch
            model = PyTorchNN(
                input_dim   = n_features,
                hidden_dim1 = args.hid1,
                hidden_dim2 = args.hid2,
                output_dim  = n_classes,
                lr          = args.lr,
                epochs      = args.epochs
            )

        # Train and measure time
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        times.append(elapsed)

        # Evaluate accuracy
        preds = model.predict(X_test)
        acc = accuracy(y_test, preds)
        accs.append(acc)

        # Print a concise summary
        print(f"Run {run+1}/{args.runs} | {args.algo} | pct={args.pct} "
              f"| time={elapsed:.3f}s | acc={acc:.3f}")
        
        # Save the last model if requested
        if run == args.runs - 1 and args.save_model:
            last_model = model

    # Print overall statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    
    print("\nOverall Statistics:")
    print(f"Algorithm: {args.algo}, Dataset: {args.dataset}, Training Data: {args.pct}%")
    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Training Time: {mean_time:.3f}s ± {std_time:.3f}s")
    
    # Save the model if requested
    if args.save_model and last_model is not None:
        models_dir = os.path.join(PROJECT_ROOT, 'saved_models')
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, f"{args.dataset}_{args.algo}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(last_model, f)

if __name__ == "__main__":
    main()
