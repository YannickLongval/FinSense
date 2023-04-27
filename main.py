'''
    using the webscraper from scraper.py, and the trained LSTM model from train.py,
    financial news articles are scraped and classified to provide positive/negative 
    news about finance.
'''
import tensorflow as tf
import numpy as np
from scraper import Scraper

scraper:Scraper = Scraper("https://ca.finance.yahoo.com/")
scraper.setSection("Main")

model = tf.keras.models.load_model('models/finsense_NSA_finance_model')

headlines:list[str] = scraper.scrape(["h2", "h3"])
predictions:list[str] = model.predict(np.array(headlines))

print(headlines)
print(predictions)