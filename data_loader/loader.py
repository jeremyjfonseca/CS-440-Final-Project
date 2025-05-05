# data_loader/loader.py

"""
This module loads the face and digit ASCII-art datasets from disk
and provides a utility to split any dataset into randomized
training and test subsets.
"""

import os
import numpy as np

def _load_ascii_dataset(image_path, label_path, img_height):
    """
    Generic loader for ASCII-art image datasets.

    Args:
        image_path (str): path to the file containing concatenated images
        label_path (str): path to the corresponding labels (one per image)
        img_height (int): number of lines per image

    Returns:
        X (ndarray, shape [n_samples, img_height*img_width]): 
            flattened binary feature vectors (0 for ' ', 1 else)
        y (ndarray, shape [n_samples,]): integer labels
    """
    # --- load labels ---
    with open(label_path, 'r') as f:
        labels = [int(line.strip()) for line in f]
    n_images = len(labels)

    # --- load raw lines ---
    with open(image_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f]

    # basic sanity checks
    assert len(lines) == n_images * img_height, (
        f"Expected {n_images*img_height} lines in {image_path}, "
        f"got {len(lines)}"
    )

    # determine image width from first line
    img_width = len(lines[0])

    # --- parse images ---
    X = np.zeros((n_images, img_height * img_width), dtype=np.float32)
    for i in range(n_images):
        block = lines[i*img_height:(i+1)*img_height]
        # convert each character to 0/1
        flat = []
        for row in block:
            # map ' ' → 0, anything else (e.g. '+', '#') → 1
            flat.extend([0.0 if ch == ' ' else 1.0 for ch in row])
        X[i, :] = np.array(flat, dtype=np.float32)

    y = np.array(labels, dtype=np.int64)
    return X, y


def load_data():
    """
    Load both face and digit datasets, combining their
    training + validation splits into full X,y arrays.

    Returns:
        (X_faces, y_faces), (X_digits, y_digits)
    """
    # locate project root relative to this file
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_dir = os.path.join(root, 'Data')

    # --- Digits ---
    digit_dir = os.path.join(data_dir, 'digitdata')
    Xd_train, yd_train = _load_ascii_dataset(
        os.path.join(digit_dir, 'trainingimages'),
        os.path.join(digit_dir, 'traininglabels'),
        img_height=28
    )
    Xd_val, yd_val = _load_ascii_dataset(
        os.path.join(digit_dir, 'validationimages'),
        os.path.join(digit_dir, 'validationlabels'),
        img_height=28
    )
    # combine train + validation
    X_digits = np.vstack([Xd_train, Xd_val])
    y_digits = np.concatenate([yd_train, yd_val])

    # --- Faces ---
    face_dir = os.path.join(data_dir, 'facedata')
    Xf_train, yf_train = _load_ascii_dataset(
        os.path.join(face_dir, 'facedatatrain'),
        os.path.join(face_dir, 'facedatatrainlabels'),
        img_height=70
    )
    Xf_val, yf_val = _load_ascii_dataset(
        os.path.join(face_dir, 'facedatavalidation'),
        os.path.join(face_dir, 'facedatavalidationlabels'),
        img_height=70
    )
    # combine train + validation
    X_faces = np.vstack([Xf_train, Xf_val])
    y_faces = np.concatenate([yf_train, yf_val])

    return (X_faces, y_faces), (X_digits, y_digits)


def get_split(X, y, pct, test_pct, seed):
    """
    Randomly split (X, y) into a test_pct% testing set, and uses pct% of the remaining as a training set.

    Args:
        X        (ndarray): feature matrix, shape (n_samples, n_features)
        y        (ndarray): label vector, shape (n_samples,)
        pct      (int): percentage (10-100) used of reserved training data
        test_pct (int): percentage (5-100) of reserved test data
        seed     (int): RNG seed for reproducibility

    Returns:
        X_train, y_train, X_test, y_test
    """
    assert 0 < pct <= 100, "pct must be in (0,100]"
    assert 5 <= test_pct < 100, "test pct must be in [5,100)"
    np.random.seed(seed)
    n = X.shape[0]
    # shuffle indices
    idx = np.random.permutation(n)
    n_test = int(n * test_pct / 100)
    n_train = int((n-n_test) * pct / 100)
    #print('n =', n, 'n_test =', n_test, '(' , n_test/n*100, '%), n_train =', n_train, '(' , n_train/n*100, '%), unused =', n-n_train-n_test, '(' , (n-n_train-n_test)/n*100, '%)')
    train_idx = idx[:n_train]
    test_idx  = idx[(n-n_test):]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
