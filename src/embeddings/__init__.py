from abc import ABC
from typing import List

import numpy as np


class BaseEmbeddingModel(ABC):
    """
    Base class for embedding models.
    """

    def __init__(self, model_name: str):
        self.model = None
        self.tokenizer = None

    def get_embeddings(self, sentences: List[str]) -> np.array:
        """
        Get embeddings for the provided sentences.

        Args:
            sentences (List[str]): The sentences to get embeddings for.

        Returns:
            np.array: The embeddings for the provided sentences.
        """
        raise NotImplementedError

    def get_similarity(self, sentence1: str, sentence2: str) -> float:
        """
        Get the similarity between the provided sentences.

        Args:
            sentence1 (str): The first sentence.
            sentence2 (str): The second sentence.

        Returns:
            float: The similarity between the provided sentences.
        """
        raise NotImplementedError
