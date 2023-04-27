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

positive = []
negative = []
for i in range(len(predictions)):
    if predictions[i] < -0.4:
        negative.append(headlines[i])
    if predictions[i] > 0.4:
        positive.append(headlines[i])

print("POSITIVE:")
print("\n".join(positive))
print("\nNEGATIVE:")
print("\n".join(negative))