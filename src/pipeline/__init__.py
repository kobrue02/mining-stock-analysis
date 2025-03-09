"""
Module for the pipeline class and related functions.
"""

from src.classifier import train_test_split, classification_report, BaseClassifier
from src.embeddings import BaseEmbeddingModel

import logging
import os


class Pipeline:
    """
    Pipeline for training and evaluating a classifier using embeddings.
    """

    def __init__(
        self,
        data,
        classifier: BaseClassifier,
        embeddings: BaseEmbeddingModel,
    ):
        """
        Initialize the pipeline.

        Args:
            data: The data to use for training and testing.
            classifier: The classifier to use.
            embeddings: The embeddings to use.
        """
        self.data = data
        self.classifier = classifier
        self.embeddings = embeddings

        self._init_logger()
        self._assert_data()

    def _init_logger(self):
        """Initialize the logger."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.propagate = False

    def _assert_data(self):
        """Assert that the data is in the correct format."""
        assert "text" in self.data, "Data must contain a 'text' column."
        assert "label" in self.data, "Data must contain a 'label' column."
        assert len(self.data["text"]) == len(
            self.data["label"]
        ), "Data columns must be of equal length."

    def run_train_test_eval(self, save_model: bool = False):
        """
        Train and evaluate the classifier using the provided data.
        """
        train_X, test_X, train_y, test_y = train_test_split(
            self.data["text"], self.data["label"], random_state=42
        )
        train_X = self.embeddings.transform_sentences(train_X)
        test_X = self.embeddings.transform_sentences(test_X)

        self.logger.info(
            f"{len(train_X)} training samples and {len(test_X)} testing samples."
        )

        embeddings = self.embeddings.get_embeddings(train_X)
        self.classifier.fit(embeddings, train_y)
        predictions = self.classifier.predict(self.embeddings.get_embeddings(test_X))

        report = classification_report(test_y, predictions)
        print(report)

        if save_model:
            mpath = "src/models/model.pkl"
            if os.path.exists(mpath):
                os.remove(mpath)
            self.classifier.save("src/models/model.pkl")

    def run_inference(self, sentences):
        """
        Run inference on a list of sentences.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = self.embeddings.get_embeddings(sentences)
        prediction = self.classifier.predict(embeddings)
        print(prediction)
        return prediction
