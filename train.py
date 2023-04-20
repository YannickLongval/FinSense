'''
This program trains a tensorflow model to analyse the 
sentiment of financial news, and predict whether the 
article is positive, negative or neutral.
'''

import pandas as pd
import numpy as np
# from tqdm import tqdm
# from keras.preprocessing.text import Tokenizer
# tqdm.pandas(desc="progress-bar")
# from gensim.models import Doc2Vec
# from sklearn import utils
# from sklearn.model_selection import train_test_split
# from keras.preprocessing.sequence import pad_sequences
# import gensim
# from sklearn.linear_model import LogisticRegression
# from gensim.models.doc2vec import TaggedDocument
# import re
# import seaborn as sns
import matplotlib.pyplot as plt

# importing the dataset and setting the column names
df = pd.read_csv('./datasets/fin-news-sentiment.csv',delimiter=',',encoding='latin-1', names=["sentiment", "message"])


# convert sentiment to numerical values to be more easily interpreted by the model
sentiment  = {'positive': 1,'neutral': 0,'negative': -1} 

df.sentiment = [sentiment[item] for item in df.sentiment] 

print(df.head())