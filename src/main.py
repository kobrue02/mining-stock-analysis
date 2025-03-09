from src.embeddings.jina_embeddings import JinaEmbeddings
from src.embeddings.nomic_embeddings import NomicEmbeddings
from src.classifier.logreg import LogRegClassifier
from src.classifier.neural_net import CNNClassifier, CNNConfig, CNNLayers
from src.pipeline import Pipeline
from src.utils.data_loaders import (
    load_sentiment_classification_data,
    load_headline_dataset,
)

import pandas as pd


def main(size=None):
    data = load_headline_dataset(size=size)
    embeddings = JinaEmbeddings()
    conf = CNNConfig.SMALL
    layers = CNNLayers.SMALL
    classifier = CNNClassifier(config=conf, layers=layers)
    pipeline = Pipeline(data=data, embeddings=embeddings, classifier=classifier)
    pipeline.run_train_test_eval(save_model=True)

def load_model_and_infer():
    data = load_sentiment_classification_data(size=2000)
    embeddings = NomicEmbeddings()
    classifier = CNNClassifier()
    classifier.load("src/models/model.pkl")
    pipeline = Pipeline(data=data, embeddings=embeddings, classifier=classifier)
    pipeline.run_inference(["I am happy", "I am sad", "I am angry"])


if __name__ == "__main__":
    #main()
    load_model_and_infer()
