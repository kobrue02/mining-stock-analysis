from sklearn.linear_model import LogisticRegression
from src.classifier import BaseClassifier


class LogRegClassifier(BaseClassifier):

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, embeddings, labels):
        self.model.fit(embeddings, labels)

    def predict(self, embeddings):
        return self.model.predict(embeddings)
