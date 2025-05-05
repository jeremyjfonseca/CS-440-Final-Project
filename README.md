# CS-440 Final Project

Final Project for CS 440 at Rutgers University: Face and Digit Classification. Implemented by Jeremy Fonseca, Ian Phipps, and Jay Choksi.


## Overview

This project implements the three classification algorithms from scratch as instructed in the project.pdf file:

1. **Perceptron** (NumPy)
2. **Three-layer Neural Network** with two hidden layers (NumPy)
3. **Three-layer Neural Network** with two hidden layers (PyTorch)

The project  covers:

* Loading ASCII-art datasets for digits and faces (`data_loader/loader.py`)
* Splitting data into randomized training/test sets
* Training and evaluating models with timing and accuracy metrics (`experiments/run.py`)

## Repository Structure

```
CS-440-Final-Project/
├── data_loader/
│   ├── __init__.py
│   └── loader.py            # Data loading and splitting utilities
├── perceptron/
│   ├── __init__.py
│   ├── base.py              # Abstract Classifier interface
│   └── model.py             # Perceptron implementation (NumPy)
├── nn_scratch/
│   ├── __init__.py
│   └── nn.py                # from scratch neural network implementation (NumPy) with two hidden layers
├── nn_pytorch/
│   ├── __init__.py
│   └── model.py             # PyTorch neural network implementation with two hidden layers
├── experiments/
│   ├── __init__.py
│   └── run.py               # CLI to run experiments
├── Data/                     # ASCII-art dataset files (digits & faces)
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the repository**



2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   # Windows:
   .\venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify package initialization**
   Ensure each directory contains an `__init__.py` (even if empty) for Python imports:

   ```bash
   ls data_loader perceptron nn_scratch nn_pytorch experiments
   ```

## Usage

Run classification experiments via the command-line interface:

```bash
python experiments/run.py \
  --algo <perceptron|scratch|pytorch> \
  --pct 50 \
  --test_pct 20 \
  --runs 3 \
  --hid1 128 \
  --hid2 64 \
  --lr 0.05 \
  --epochs 20
```

### Arguments

* `--algo`     : Algorithm to run (`perceptron`, `scratch`, or `pytorch`).
* `--pct`      : Percentage of reserved training data used in training (integer 10–100).
* `--test_pct` : Percentage of data reserved for testing (default: 20)
* `--runs`     : Number of random trials to average (default: 1).
* `--hid1`     : Size of the **first** hidden layer for NN models (default: 100).
* `--hid2`     : Size of the **second** hidden layer for NN models (default: 50).
* `--lr`       : Learning rate for all models (default: 0.01).
* `--epochs`   : Number of training epochs for all models (default: 10).

### Example

```bash
python experiments/run.py --algo scratch --pct 50 --test_pct 15 --runs 3 --hid1 128 --hid2 64 --lr 0.05 --epochs 20
```

## File Descriptions

### `data_loader/loader.py`

* **`load_data()`**: Reads ASCII-art images and labels for faces and digits and returns two datasets: `(X_faces, y_faces), (X_digits, y_digits)`.
* **`get_split(X, y, pct, seed)`**: Randomly splits `X, y` into `pct`% training and `(100-pct)`% test sets, using `seed` for reproducibility.

### `perceptron/base.py`

* **`Classifier`**: Abstract base class defining the `fit(X, y)` and `predict(X)` interface for all models.

### `perceptron/model.py`

* **`Perceptron`**: Implements the perceptron learning algorithm with NumPy, mapping labels to ±1 internally and updating weights/bias via the classic rule.

### `nn_scratch/nn.py`

* **`NeuralNetScratch`**: Three-layer neural network (two hidden layers) built from scratch with NumPy. Uses `tanh` activations in hidden layers and `sigmoid` at output, trained with batch gradient descent.

### `nn_pytorch/model.py`

* **`Net`**: Defines the PyTorch network architecture: `Linear → ReLU → Linear → ReLU → Linear` (raw logits).
* **`PyTorchNN`**: Wraps `Net` to conform to the `Classifier` interface, using `CrossEntropyLoss` and `SGD` for training.

### `experiments/run.py`

* Injects project root into `sys.path` for proper imports.
* Parses CLI arguments for algorithm selection, data split, hidden-layer sizes, learning rate, and epochs.
* Loads data, splits, instantiates the chosen model, trains, evaluates, and prints timing & accuracy.

## Adding new models

To add a new model:

1. Create a new package directory (e.g., `my_model/`) with `__init__.py`.
2. Implement a class inheriting from `Classifier` in your module.
3. Import your class in `experiments/run.py` and extend the `--algo` choices and instantiation logic.
