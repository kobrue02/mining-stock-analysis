"""
Module for the pipeline class and related functions.
"""

from src.classifier import train_test_split, classification_report, BaseClassifier
from src.embeddings import BaseEmbeddingModel
from src.market_data import TimeSeriesDaily
from src.market_data.yfinance_api import YFinanceAccess

import logging
import os
import pandas as pd



class Pipeline:
    """
    Pipeline for training and evaluating a classifier using embeddings.
    """

    def __init__(
        self,
        data = None,
        classifier: BaseClassifier = None,
        embeddings: BaseEmbeddingModel = None,
        timeseries: TimeSeriesDaily = None,
        news_reports: pd.DataFrame = None,
        exchange: str = "TSX",
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
        self.timeseries = timeseries
        self.exchange = exchange
        self.news_reports = news_reports

        self.yf = YFinanceAccess()

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
        if not self.data:
            return
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
            self.data["text"],
            self.data["label"],
            random_state=42,
            stratify=self.data["label"],
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
    
    def get_best_two_nr_reports(self):
        """
        Get the two news reports with the highest stock advance percentage.
        """
        if self.news_reports is None:
            return None

        stocks = self.news_reports["company"].unique()
        report_pairs = []
        for stock in stocks:
            company_candidates = []
            stock_reports = self.news_reports[self.news_reports["company"] == stock]
            time_range = stock_reports["date"].min(), stock_reports["date"].max()
            if self.exchange == "TSX":
                stock_data = self.yf.historic_tsx_data(stock)
            elif self.exchange == "NYSE":
                stock_data = self.yf.historic_nyse_data(stock)
            else:
                raise ValueError("Exchange must be 'TSX' or 'NYSE'.")
            
            # for any combination of two news reports for this stock
            # check the stock advance percentage
            for i in range(len(stock_reports)):
                for j in range(i + 1, len(stock_reports)):
                    # date j must be after date i
                    if stock_reports.iloc[i]["date"] > stock_reports.iloc[j]["date"]:
                        break
                    report1 = stock_reports.iloc[i]
                    report2 = stock_reports.iloc[j]
                    date1 = report1["date"]
                    date2 = report2["date"]
                    self.logger.info(f"Comparing {date1} and {date2} for {stock}.")
                    diff_price, diff_percentage = stock_data.diff_between_dates(date1, date2)
                    company_candidates.append((report1["headline"], report2["headline"], diff_percentage, date1, date2))
            
            report_pairs.extend(company_candidates)
        
        # sort by stock advance percentage
        report_pairs.sort(key=lambda x: x[2], reverse=True)

        return report_pairs

