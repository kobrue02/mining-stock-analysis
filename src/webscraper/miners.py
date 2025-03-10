import pandas as pd
import requests
import re
import time

from bs4 import BeautifulSoup
from datetime import datetime
from googlesearch import search
from tqdm import tqdm


def get_press_release_date(headline, sleep=0):
    query = "Junior Mining Network: " + headline
    try:
        results = search(query, num_results=1, advanced=True)
        result = next(results)
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        tqdm.write("Rate limited, sleeping for {} seconds".format(sleep))
        time.sleep(sleep)
        return ""
    description = result.description
    try:
        # date format usually "20 Feb 2025"
        date = re.search(r"\d{1,2} [A-Za-z]{3} \d{4}", description).group()
        return datetime.strptime(date, "%d %b %Y").date()
    except AttributeError:
        tqdm.write("Date not found in description: {}".format(description))
        return ""

    

class MiningWebsiteAccess:
    """Access mining websites and extract news headlines."""

    websites = {
        "ERD": "https://erdene.com/en/news/",
        "GMIN": "https://gmin.gold/news/",
        "CXB": "https://www.calibremining.com/news/default.aspx",
        "ABRA": "https://abrasilver.com/news-releases/",
        "CDE": "https://www.coeur.com/investors/news/default.aspx",
        "LGD": "https://libertygold.ca/news/",
    }

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        self.rate_limit_sleep_seconds = 2

    def _get_soup(self, url):
        response = requests.get(url, headers=self.headers)
        return BeautifulSoup(response.text, "html.parser")

    def _get_soup_lxml(self, url):
        response = requests.get(url, headers=self.headers)
        return BeautifulSoup(response.content, "lxml")

    def _get_headline_and_date(self, soup, tag, class_):
        titles = soup.find_all(tag, class_=class_)
        return [
            (
                self._process_headline_string(title.text),
                self._get_date_article_posted(title.text)
                )
            for title in titles
            ]

    def _process_headline_string(self, headline):
        return headline.strip().replace("\n", " ")
    
    def _get_date_article_posted(self, headline: str):
        date = get_press_release_date(headline, sleep=self.rate_limit_sleep_seconds)
        if date:
            return date
        else:
            return "N/A"  # implement a better way to handle this later
    
    def _return_list_object(self, ticker: str, titles: tuple) -> list:
        return [
            (title[0], ticker, title[1])
            for title in titles
        ]

    def load_abra_silver_news(self):
        soup = self._get_soup(self.websites["ABRA"])
        titles = self._get_headline_and_date(soup, "a", "title")
        return self._return_list_object("ABRA", titles)

    def load_erdene_news(self):
        years = range(2020, 2024)
        titles = []
        for year in years:
            soup = self._get_soup(self.websites["ERD"] + f"{year}/")
            titles.extend(self._get_headline_and_date(soup, "div", "uk-width-expand@s"))
        return self._return_list_object("ERD", titles)

    def load_gmin_news(self):
        soup = self._get_soup(self.websites["GMIN"])
        titles = self._get_headline_and_date(soup, "div", "data-text !mb-0")
        return self._return_list_object("GMIN", titles)

    def load_calibre_news(self):
        soup = self._get_soup_lxml(self.websites["CXB"])
        titles = self._get_headline_and_date(soup, "a", "module_headline-link")
        return self._return_list_object("CXB", titles)
    
    def load_coeur_news(self):
        soup = self._get_soup_lxml(self.websites["CDE"])
        titles = self._get_headline_and_date(soup, "a", "module_headline-link")
        return self._return_list_object("CDE", titles)
    
    def load_liberty_news(self):
        titles = []
        for year in range(2020, 2025):
            soup = self._get_soup(self.websites["LGD"] + f"{year}-news.html")
            titles.extend(self._get_headline_and_date(soup, "div", "Column isTitleColumn"))
        return self._return_list_object("LGD", titles)

    def to_df(self):
        news = []
        company_loaders = [
            self.load_gmin_news,
            self.load_calibre_news,
            self.load_coeur_news,
            self.load_abra_silver_news,
            self.load_erdene_news,
            self.load_liberty_news,
            # ... other websites
        ]
        
        # Use tqdm to create a progress bar for the company loaders
        for loader in tqdm(company_loaders, desc="Loading company news"):
            company_news = loader()
            news.extend(company_news)
        
        df = pd.DataFrame(news, columns=["headline", "company", "date"])
        return df


if __name__ == "__main__":
    m = MiningWebsiteAccess()
    df = m.to_df()
    print(df)
    print(f"Number of news headlines: {len(df)}")
    print(f"Included companies: {', '.join(df['company'].unique())}")
