import requests
import os

from src.market_data import TimeSeriesDaily, TimeoutError, MarketDataRetriever

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


class AlphaVantageAccess(MarketDataRetriever):

    def _response_to_timeseries(self, symbol, url):
        response = requests.get(url)
        data = response.json()
        try:
            daily_close = [
                (key, item["4. close"])
                for (key, item)
                in data["Time Series (Daily)"].items()
            ]
        except KeyError:
            raise TimeoutError(data["Information"])
        return TimeSeriesDaily(symbol, daily_close)
    
    def historic_tsx_data(self, symbol):
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}.TRT&outputsize=full&apikey={self.api_key}"
        return self._response_to_timeseries(symbol, url)
    
    def historic_nyse_data(self, symbol):
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={self.api}"
        return self._response_to_timeseries(symbol, url)



def main():
    av = AlphaVantageAccess(ALPHA_VANTAGE_API_KEY)
    ticker = "LGD"
    _ = av.historic_tsx_data(ticker).to_line_chart(mark_bottom=True, mark_top=True)
    print(av.historic_tsx_data(ticker).diff_between_dates("2022-01-04", "2022-01-05"))
    print(av.historic_tsx_data(ticker).percentage_off_ath())



if __name__ == "__main__":
    main()