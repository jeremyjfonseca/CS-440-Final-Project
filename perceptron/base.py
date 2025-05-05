class Classifier:
    """
    Abstract base class for all classifiers in this project.
    Subclasses must implement fit() and predict().
    """
    def fit(self, X, y):
        """
        Train the model on features X and labels y.
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Given features X, return predicted labels.
        """
        raise NotImplementedError
