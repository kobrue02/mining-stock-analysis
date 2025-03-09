from src.embeddings import BaseEmbeddingModel
from transformers import AutoModel
from tqdm import tqdm

import numpy as np


class JinaEmbeddings(BaseEmbeddingModel):
    """
    Embedding model using the Jina embeddings model.
    """

    def __init__(self):
        super().__init__("jina-embeddings-v3")
        self.model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v3", trust_remote_code=True
        )

    def get_embeddings(self, sentences):
        if len(sentences) > 1000:
            batches = self.sentences_to_batches(sentences, batch_size=1000)
        else:
            batches = [sentences]
        embeddings = []
        for batch in tqdm(batches):
            batch_embeddings = self.model.encode(batch, task="text-matching")
            embeddings.append(batch_embeddings)
        embeddings = np.concatenate(embeddings)
        return embeddings

    def get_similarity(self, sentence1, sentence2):
        embeddings = self.get_embeddings([sentence1, sentence2])
        similarity = embeddings[0] @ embeddings[1].T
        return similarity.item()

    @staticmethod
    def transform_sentence(sentence):
        """Transform a sentence to the format expected by the model."""
        return sentence

    @staticmethod
    def transform_sentences(sentences):
        """Transform a list of sentences to the format expected by the model."""
        return sentences
