import requests
import os
import yfinance as yf

from src.market_data import TimeSeriesDaily, TimeoutError, MarketDataRetriever


class YFinanceAccess(MarketDataRetriever):
    
    def historic_nyse_data(self, symbol):
        data = yf.Ticker(symbol)
        df = data.history(period="max")
        daily_close = [(str(date.date()), close) for date, close in zip(df.index, df["Close"])]
        return TimeSeriesDaily(symbol, daily_close)
    
    def historic_tsx_data(self, symbol):
        return self.historic_nyse_data(symbol + ".TO")
    

def main():
    yf = YFinanceAccess()
    ticker = "LGD"
    _ = yf.historic_tsx_data(ticker).to_line_chart(mark_bottom=True, mark_top=True, add_200ma=True)
    print("Local tops: " + str(yf.historic_tsx_data(ticker).get_local_tops()))
    print("Local bottoms: " + str(yf.historic_tsx_data(ticker).get_local_bottoms()))


if __name__ == "__main__":
    main()
