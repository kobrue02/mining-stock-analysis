import pandas as pd
import kagglehub
import os


def load_headline_dataset(size: int = None):
    """
    Load the news headline dataset from the rmisra/news-category-dataset
    """
    # Download latest version
    path = "rmisra/news-category-dataset"
    if not os.path.exists(path):
        path = kagglehub.dataset_download("rmisra/news-category-dataset")
    path += "/News_Category_Dataset_v3.json"
    df = pd.read_json(path, lines=True)

    df.rename(columns={"headline": "text", "category": "label"}, inplace=True)
    n_labels = df["label"].nunique()
    if size is not None:
        df = df.groupby("label").head(size // n_labels)
    return df


def load_sentiment_classification_data(size: int = 1000) -> pd.DataFrame:
    """
    Load the sentiment classification data from the dair-ai/emotion dataset
    """
    hfpath = "hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet"
    df = pd.read_parquet(hfpath)
    # get a balanced subset of the data
    df = df.groupby("label").head(size // 6)
    labels = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    df["label"] = df["label"].map(labels)
    return df
