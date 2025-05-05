import numpy as np

def load_data():
    """
    Load and return (X_faces, y_faces), (X_digits, y_digits).
    """
    # TODO: read raw files into NumPy arrays
    raise NotImplementedError

def get_split(X, y, pct, seed):
    """
    Given feature matrix X and labels y, return:
      X_train, y_train = random pct% subset (seeded)
      X_test,  y_test  = the complement
    """
    raise NotImplementedError
