from abc import ABC
from typing import List

import logging
import numpy as np


class BaseEmbeddingModel(ABC):
    """
    Base class for embedding models.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

    def get_embeddings(self, sentences: List[str]) -> np.array:
        """
        Get embeddings for the provided sentences.

        Args:
            sentences (List[str]): The sentences to get embeddings for.

        Returns:
            np.array: The embeddings for the provided sentences.
        """
        raise NotImplementedError
    
    def sentences_to_batches(self, sentences: List[str], batch_size: int) -> List[List[str]]:
        """
        Convert the provided sentences to batches.

        Args:
            sentences (List[str]): The sentences to convert to batches.
            batch_size (int): The size of each batch.

        Returns:
            List[List[str]]: The sentences converted to batches.
        """
        batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        self.logger.info(f"Converted {len(sentences)} sentences to {len(batches)} batches.")
        return batches

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
