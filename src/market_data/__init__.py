import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class TimeoutError(Exception):
    pass

class TimeSeriesDaily:

    def __init__(self, symbol, data):
        self.symbol = symbol
        self.data = data
        self.df = self.to_df()

    def to_df(self):
        df = pd.DataFrame(self.data, columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"])
        # reverse order
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    
    def to_line_chart(
            self, 
            mark_bottom: bool = False, 
            mark_top: bool = False,
            add_200ma: bool = False
            ):
        self.df.plot(x="date", y="close", title=f"{self.symbol} Daily Close Prices")
        # decrease thickness of line
        plt.gca().get_lines()[0].set_linewidth(0.5)
        if mark_bottom:
            self._mark_bottom()
        if mark_top:
            self._mark_top()
        if add_200ma:
            self._add_200ma()
        plt.show()

    def _add_200ma(self):
        # add 200 day moving average
        self.df["200ma"] = self.df["close"].rolling(window=200).mean()
        # shift it by 200 days to the right
        self.df["200ma"] = self.df["200ma"].shift(-200)
        # add t0 plot with 0.5 thickness
        self.df.plot(x="date", y="200ma", color="orange", linewidth=0.75, ax=plt.gca())

    def _mark_top(self):
        # add horizontal line at top
        # if the top is right after IPO, use local top instead
        ath = self.df["close"].max()
        if self.df["close"].idxmax() == 0:
            # find local top at least 1 year after IPO
            ath = self.df["close"].iloc[365:].max()
        plt.axhline(y=ath, color="g", linestyle="--").set_linewidth(0.5)

    def _mark_bottom(self):
        # add horizontal line at bottom
        plt.axhline(y=self.df["close"].min(), color="r", linestyle="--").set_linewidth(0.5)

    def diff_between_dates(self, date1, date2):
        close1 = self.df[self.df["date"] == date1]["close"].values[0]
        close2 = self.df[self.df["date"] == date2]["close"].values[0]
        price_diff = float(round(close2 - close1, 2))
        percentage = float(round((price_diff / close1) * 100, 2))
        return price_diff, percentage
    
    def percentage_off_ath(self):
        ath = self.df["close"].max()
        current = self.df["close"].iloc[-1]
        return round((current - ath) / ath * 100, 2)
    
    def percentage_off_atl(self):
        atl = self.df["close"].min()
        current = self.df["close"].iloc[-1]
        return round((current - atl) / atl * 100, 2)
    
    def get_local_tops(self, window=365):
        raise NotImplementedError
    
    def get_local_bottoms(self, window=365):
        raise NotImplementedError
    

class MarketDataRetriever(ABC):

    def __init__(self, api_key = None):
        self.api_key = api_key

    @abstractmethod
    def historic_tsx_data(self, symbol):
        raise NotImplementedError

    @abstractmethod
    def historic_nyse_data(self, symbol):
        raise NotImplementedError