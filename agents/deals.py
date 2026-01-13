from pydantic import BaseModel, Field
from typing import List, Dict, Self, Any
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import requests
import time
import ssl

### Add ssl to prevent the ssl issue for feedparser accessing.
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

feeds = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]

# You could also add: "https://www.dealnews.com/c238/Automotive/?rss=1"
# "https://www.dealnews.com/c196/Home-Garden/?rss=1"

def extract(html_snippet: str) -> str:
    """
    A utility function that uses Beautiful Soup to clean up this HTML snippet and extract useful text

    :param html_snippet: text wrapped up with HTML elements to clean up into clean text.
    """
    soup = BeautifulSoup(html_snippet, features="html.parser")
    snippet_div = soup.find("div", class_="snippet summary")

    if snippet_div:
        return snippet_div.get_text(strip=True).replace("\n", " ")

    return soup.get_text(strip=True).replace("\n", " ")


class ScrapedDeal:
    """
    A class to represent a Deal retrieved from an RSS feed
    """

    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict[str, str]):
        """
        Populate this instance based on the provided dict
        """
        self.title = entry["title"]
        self.summary = extract(entry["summary"])
        self.url = entry["links"][0]["href"]
        raw_page_content = requests.get(self.url).content ### Get text at the page level (for product details)
        soup = BeautifulSoup(raw_page_content, "html.parser")
        content = soup.find("div", class_="content-section").get_text()
        content = content.replace("\nmore", "").replace("\n", " ")

        if "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details = content
            self.features = ""
        self.truncate()

    def truncate(self):
        """
        Set a text limit to the title, details, features of the content
        :return: None
        """
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def __repr__(self):
        """
        Return a string to describe this deal
        """
        return f"<{self.title}>"

    def describe(self):
        """
        Return a longer string to describe this deal for use in calling a model
        """
        return f"Title: {self.title}\n\nDetails: {self.details.strip()}\n\nFeatures: {self.features.strip()}\n\nURL: {self.url}"

    @classmethod
    def fetch(cls, show_progress: bool = False):
        """
        Retrieve all deals from the selected RSS feeds
        """
        deals = []
        feed_iter = tqdm(feeds) if show_progress else feeds
        for feed_url in feed_iter:
            feed = feedparser.parse(
                feed_url,
                request_headers={
                    "User-Agent": "Mozilla/5.0 (compatible; DealFetcher/1.0)",
                }
            )
            for entry in feed["entries"][:10]:
                deals.append(cls(entry))
                time.sleep(0.5)

        return deals























