from src.embeddings import BaseEmbeddingModel
from transformers import AutoModel


class JinaEmbeddings(BaseEmbeddingModel):
    """
    Embedding model using the Jina embeddings model.
    """

    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v3", trust_remote_code=True
        )

    def get_embeddings(self, sentences):
        embeddings = self.model.encode(sentences, task="text-matching")
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
