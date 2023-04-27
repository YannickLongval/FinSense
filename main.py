'''
    using the webscraper from scraper.py, and the trained LSTM model from train.py,
    financial news articles are scraped and classified to provide positive/negative 
    news about finance.
'''

import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('models/finsense_NSA_finance_model')

sample = ["We’re deeper in debt than Ottawa lets on", "Barclays posts 27 percent rise in net profit for the first quarter, beats expectations", "Meta Platforms stock soars after earnings crush expectations, expenses drop"]

predictions = model.predict(np.array(sample))

print(predictions)