from src.embeddings.jina_embeddings import JinaEmbeddings
from src.embeddings.nomic_embeddings import NomicEmbeddings
from src.classifier.logreg import LogRegClassifier
from src.classifier.neural_net import CNNClassifier, CNNConfig, CNNLayers
from src.pipeline import Pipeline
from src.utils.data_loaders import (
    load_sentiment_classification_data,
    load_headline_dataset,
)
from src.webscraper.junior_mining_network import JMNScraper
from src.webscraper.miners import MiningWebsiteAccess

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


def vectorize_jmn_headlines():
    scraper = JMNScraper()
    df = scraper.to_df()
    jina = JinaEmbeddings()
    embeddings = jina.get_embeddings(df["headline"].tolist())
    return embeddings


def predict_mining_stock():
    scraper = MiningWebsiteAccess()
    df = scraper.to_df()
    df.rename(columns={"headline": "text", "company": "label"}, inplace=True)
    embeddings = JinaEmbeddings()
    classifier = CNNClassifier()
    pipeline = Pipeline(data=df, embeddings=embeddings, classifier=classifier)
    pipeline.run_train_test_eval(save_model=True)


def best_nr_reports():
    df = pd.read_csv("data/mining_headlines.csv")
    # remove all columns with "N/A" in the date column
    df = df[df["date"] != "N/A"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna()
    pipe = Pipeline(news_reports=df, exchange="TSX")
    nbest = pipe.get_best_two_nr_reports()
    
    nbest_df = pd.DataFrame(nbest, columns=["report_1", "report_2", "advance_percentage", "date_r1", "date_r2"])
    print(nbest_df)


if __name__ == "__main__":
    best_nr_reports()
