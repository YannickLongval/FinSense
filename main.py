'''
    using the webscraper from scraper.py, and the trained LSTM model from train.py,
    financial news articles are scraped and classified to provide positive/negative 
    news about finance.
'''
import tensorflow as tf
import numpy as np
from scraper import Scraper
import random

"""Determine which headlines are predicted to be positive/negative from the model

Args:
    headlines (list[str]): the headlines that were passed into the model
    predictions (list[str]): the predictions for the headlines from the model
    threshold (float): how much confidence the model should have to classify the sentiment

Returns a tuple[list[str], list[str]], where the first list is the positive headlines, and the second list is the negative
"""
def classify(headlines:list[str], predictions:list[str], threshold:float=0.5) -> tuple[list[str], list[str]]:
    positive = []
    negative = []
    for i in range(len(predictions)):
        if predictions[i] < -threshold:
            negative.append(headlines[i])
        if predictions[i] > threshold:
            positive.append(headlines[i])
    return positive, negative

"""Formats and prints the headlines based on their sentiment

Args:
    positive (list[str]): the headlines that were determined positive from the model
    negative (list[str]): the headlines that were determined negative from the model
    numHeadlines (int): the numbers of headlines that should be printed for each sentiment (-1 will print all available headlines)
    incPos (boolean): whether or not the positive headlines should be printed
    incNeg (boolean): whether or not the negative headlines should be printed
"""
def printHeadlines(positive:list[str], negative:list[str], numHeadlines:int=-1, incPos:bool=True, incNeg:bool=True) -> None:
    if incPos:
        print("\nPOSITIVE:")
        if numHeadlines == -1 or numHeadlines >= len(positive):
            print("\n".join(positive))
        else:
            print("\n".join(random.sample(positive, k=numHeadlines)))
    if incNeg :
        print("\nNEGATIVE:")
        if numHeadlines == -1 or numHeadlines >= len(negative):
            print("\n".join(negative))
        else:
            print("\n".join(random.sample(negative, k=numHeadlines)))


# user has option to scrape specific website
url:str = input("Input link to website you'd like to extract news articles from (if no site is given, will default to https://ca.finance.yahoo.com/: ")

if url == "":
    url = "https://ca.finance.yahoo.com/"

scraper:Scraper = Scraper(url)

# if the site is not the default, user may need to inspect the web contents to determine which section to scrape
if url != "https://ca.finance.yahoo.com/":
    section:str = input("Which section of the site would you like to scrape? Input the id of the section (no id will scrape the entire website): ")
    if section != "":
        scraper.setSection(section)
else:
    scraper.setSection("Main")

# load the trained model from train.py
model = tf.keras.models.load_model('models/finsense_NSA_finance_model')

# scrape from site, and make sentiment predictions on the scraped headlines
headlines:list[str] = scraper.scrape(["h2", "h3"])
predictions:list[str] = model.predict(np.array(headlines))

# Sort headlines and print them based on their sentiment
positive, negative = classify(headlines, predictions, 0.4)
printHeadlines(positive, negative)