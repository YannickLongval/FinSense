'''
This program trains a tensorflow model to analyse the 
sentiment of financial news, and predict whether the 
article is positive or negative.
'''

import pandas as pd
import numpy as np
import tensorflow as tf
# from tqdm import tqdm
# from keras.preprocessing.text import Tokenizer
# tqdm.pandas(desc="progress-bar")
# from gensim.models import Doc2Vec
# from sklearn import utils
from sklearn.model_selection import train_test_split
# from keras.preprocessing.sequence import pad_sequences
# import gensim
# from sklearn.linear_model import LogisticRegression
# from gensim.models.doc2vec import TaggedDocument
# import re
# import seaborn as sns
import matplotlib.pyplot as plt

# this function plots the accuracy and loss for the training and testing data
def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

# importing the dataset and setting the column names
df:pd.DataFrame = pd.read_csv('./datasets/fin-news-sentiment.csv',delimiter=',',encoding='latin-1', names=["sentiment", "message"])

# convert sentiment to numerical values to be more easily interpreted by the model
sentiment:dict  = {'positive': 1,'neutral': -1,'negative': 0}

# drop any neutral entries (since we are only looking at positive/negative cases, and convert the targets into numerical values)
df.sentiment = [sentiment[item] for item in df.sentiment] 
df = df[df.sentiment != -1]

# split data into training and testing, using a 90:10 ratio
train, test = train_test_split(df, test_size=0.1, random_state=69)

# separate the target from the data. This is to create the tensorflow.data.Dataset
var_target_train, var_target_test = train.pop('sentiment'), test.pop('sentiment')

# creating the tf.data.Datasets
tf_train:tf.data.Dataset = tf.data.Dataset.from_tensor_slices((train.values, var_target_train.values))
tf_test:tf.data.Dataset = tf.data.Dataset.from_tensor_slices((test.values, var_target_test.values))

# setting buffer and batch sizes based on amount of data (not as much so sizes are smaller)
BUFFER_SIZE:int = 500
BATCH_SIZE:int = 10

# shuffle, and batch data
tf_train = tf_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
tf_test = tf_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# creating encoder to tokenize text.
VOCAB_SIZE:int = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(tf_train.map(lambda text, label: text))

# model composed of 6 layers: input, tokenization, embedding, LSTM, activation, and a dense layer to produce a single output.
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Binary Cross-entropy loss function because we're creating a binary classifier
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

# training model with training data, and validating with test data
history = model.fit(tf_train, epochs=5,
                    validation_data=tf_test,
                    validation_steps=10)

# save the model to be used in the future
model.save('models/finsense_NSA_finance_model')

# print accuracy/loss of classifier on test data
test_loss, test_acc = model.evaluate(tf_test)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# plot accuracy/loss of classifier
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
plt.show()