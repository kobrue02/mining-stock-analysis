import pandas as pd
import requests

from bs4 import BeautifulSoup


class MiningWebsiteAccess:
    """ Access mining websites and extract news headlines. """
    websites = {
        "ERD": "https://erdene.com/en/news/",
        "GMIN": "https://gmin.gold/news/",
        "CXB": "https://www.calibremining.com/news/default.aspx",
        "ABRA": "https://abrasilver.com/news-releases/"
    }

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

    def _get_soup(self, url):
        response = requests.get(url, headers=self.headers)
        return BeautifulSoup(response.text, "html.parser")
    
    def _get_soup_lxml(self, url):
        response = requests.get(url, headers=self.headers)
        return BeautifulSoup(response.content, "lxml")

    def _extract_headlines(self, soup, tag, class_):
        titles = soup.find_all(tag, class_=class_)
        return [
            self._process_headline_string(title.text)
            for title in titles
            ]
    
    def _process_headline_string(self, headline):
        return headline.strip().replace("\n", " ")
    
    def load_abra_silver_news(self):
        soup = self._get_soup(self.websites["ABRA"])
        titles = self._extract_headlines(soup, "a", "title")
        return [
            (title, "ABRA")
            for title in titles
            ]
    
    def load_erdene_news(self):
        years = range(2020, 2024)
        titles = []
        for year in years:
            soup = self._get_soup(self.websites["ERD"] + f"{year}/")
            titles.extend(self._extract_headlines(soup, "div", "uk-width-expand@s"))
        return [
            (title, "ERD")
            for title in titles
            ]
    
    def load_gmin_news(self):
        soup = self._get_soup(self.websites["GMIN"])
        titles = self._extract_headlines(soup, "div", "data-text !mb-0")
        return [
            (title, "GMIN")
            for title in titles
            ]
    
    def load_calibre_news(self):
        soup = self._get_soup_lxml(self.websites["CXB"])
        titles = self._extract_headlines(soup, "a", "module_headline-link")
        return [
            (title, "CXB")
            for title in titles
            ]
    
    def to_df(self):
        news = self.load_gmin_news()
        news.extend(self.load_calibre_news())
        news.extend(self.load_abra_silver_news())
        news.extend(self.load_erdene_news())
        # ... other websites
        df = pd.DataFrame(news, columns=["headline", "company"])
        return df
    

if __name__ == "__main__":
    m = MiningWebsiteAccess()
    df = m.to_df()
    print(df)
    print(f"Number of news headlines: {len(df)}")
    print(f"Included companies: {", ".join(df['company'].unique())}")