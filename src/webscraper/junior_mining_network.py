import requests
import pandas as pd

from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


class JMNScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        }
        self.news_url = "https://www.juniorminingnetwork.com/junior-miner-news.html"
        self.silver_stocks_url = "https://www.juniorminingnetwork.com/mining-stocks/silver-mining-stocks.html"
        self.gold_stocks_url = (
            "https://www.juniorminingnetwork.com/mining-stocks/gold-mining-stocks.html"
        )
        self.all_stock_names = self.get_stocks_names()
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
        self.model = AutoModelForTokenClassification.from_pretrained(
            "dslim/distilbert-NER"
        )
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def get_news_headlines(self):
        response = requests.get(self.news_url, headers=self.headers)
        soup = BeautifulSoup(response.text, "html.parser")
        titles = soup.find_all("div", class_="article-title")
        return [title.text for title in titles]

    def extract_company_name(self, headline):
        # match against stock names
        for stock_name in self.all_stock_names:
            if stock_name.lower() in headline.lower():
                return stock_name

        ner = self.nlp(headline)
        company_name = ""
        for entity in ner:
            if entity["entity"] == "B-ORG":
                company_name = entity["word"]
            elif entity["entity"] == "I-ORG":
                company_name += " " + entity["word"]
        return company_name

    def get_stocks_names(self):
        """Get the names of silver and gold stocks from the Junior Mining Network website."""
        # silver
        response = requests.get(self.silver_stocks_url, headers=self.headers)
        soup = BeautifulSoup(response.text, "html.parser")
        titles = soup.find_all("span", class_="stock-title")
        silver_stocks = [title.text for title in titles]

        # gold
        response = requests.get(self.gold_stocks_url, headers=self.headers)
        soup = BeautifulSoup(response.text, "html.parser")
        titles = soup.find_all("span", class_="stock-title")
        gold_stocks = [title.text for title in titles]

        return silver_stocks + gold_stocks

    def scrape_website(self):
        headlines = self.get_news_headlines()
        return [
            (headline, self.extract_company_name(headline))
            for headline in headlines
            if "#" not in self.extract_company_name(headline)
        ]

    def to_df(self):
        data = self.scrape_website()
        return pd.DataFrame(data, columns=["headline", "company_name"])


if __name__ == "__main__":
    scraper = JMNScraper()
    data = scraper.scrape_twitter()
    print(data.head())
