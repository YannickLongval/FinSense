'''
    web scraper object to extract news headlines from a news site
'''

import requests
from bs4 import BeautifulSoup

class Scraper:
    """Contructor for Scraper class
    
    Args:
        url (string): the site that the webscraper should scrape from
    """
    def __init__(self, url:str='https://ca.finance.yahoo.com/') -> None:
        self.url = url
        self.page = requests.get(self.url)
        self.soup = BeautifulSoup(self.page.content, "html.parser")
        self.section = ""

    """Get the url of the site that the scraper is scraping from

    Returns the url as a string.
    """
    def getSite(self) -> str:
        return self.url
    
    """Set the url of the site that the scraper should scrape from
    
    Args:
        url (string): the url of the new site that the scraper should scrape
    """
    def setSite(self, url:str) -> None:
        self.url = url
        self.page = requests.get(self.url)
        self.soup = BeautifulSoup(self.page.content, "html.parser")
        self.section = ""

    """Set which section of the page that the scraper should scrape from, based on the sections id

    Args:
        section (string): the id of the section the scraper should scrape from
    """
    def setSection(self, section:str) -> None:
        self.section = self.soup.find(id=section)

    """Scrape certain elements from the site
    
    Args:
        tags (list[str]): the elements that should be scraped from the section

    Returns the text inside the elements of <tags>
    """
    def scrape(self, tags:list[str]=[]) -> list[str]:
        results = []
        for tag in tags:
            for result in self.section.find_all(tag):
                results.append(result.text)
            
        return results