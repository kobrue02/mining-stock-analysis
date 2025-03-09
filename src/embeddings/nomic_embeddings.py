import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from src.embeddings import BaseEmbeddingModel


class NomicEmbeddings(BaseEmbeddingModel):

    def __init__(self, model_name="nomic-ai/nomic-embed-text-v2-moe"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embeddings(self, sentences):
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def get_similarity(self, sentence1, sentence2):
        embeddings = self.get_embeddings([sentence1, sentence2])
        similarity = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)
        return similarity

    @staticmethod
    def transform_sentence(sentence):
        """Transform a sentence to the format expected by the model."""
        return f"search_document: {sentence}"

    @staticmethod
    def transform_sentences(sentences):
        """Transform a list of sentences to the format expected by the model."""
        return [NomicEmbeddings.transform_sentence(sentence) for sentence in sentences]


if __name__ == "__main__":
    nomic = NomicEmbeddings()
    sentences = ["search_document: Hello!", "search_document: ¡Hola!"]
    embeddings = nomic.get_embeddings(sentences)
    similarity = nomic.get_similarity(sentences[0], sentences[1])
    print(embeddings[0].shape, similarity)
