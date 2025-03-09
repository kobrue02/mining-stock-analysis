from typing import List
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import Tensor


class BaseClassifier:

    def __init__(self):
        pass

    def fit(
        self, embeddings: List | np.array | Tensor, labels: List | np.array | Tensor
    ):
        """
        Fit the classifier to the provided embeddings and labels.

        Args:
            embeddings (List | np.array | Tensor): The embeddings to train the classifier on.
            labels (List | np.array | Tensor): The labels for the embeddings.
        """
        raise NotImplementedError

    def predict(self, embeddings):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
