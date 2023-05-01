# FinSense
A sentiment analysis classifier for finanical news.

This model uses Long Short Term Memory (LSTM) networks with TensorFlow to classify the positivity of a finanical news headline. More about LSTM's can be found here: 
<br/><br/> 
[https://intellipaat.com/blog/what-is-lstm/#:~:text='%20LSTM%20stands%20for%20long%20short,especially%20in%20sequence%20prediction%20problems.](https://intellipaat.com/blog/what-is-lstm/#:~:text='%20LSTM%20stands%20for%20long%20short,especially%20in%20sequence%20prediction%20problems.)
<br/><br/>
The dataset for this project can be found here: 
<br/><br/>
[https://www.kaggle.com/code/khotijahs1/nlp-financial-news-sentiment-analysis/input?select=all-data.csv](https://www.kaggle.com/code/khotijahs1/nlp-financial-news-sentiment-analysis/input?select=all-data.csv)
<br/><br/>
Around 5000 data points are included in the dataset before preprocessing (there are a lot of 'neutral' entries, so the total number of entries used is less), where each entry passed into the model consists of a headline, along with a "positive", or "negative" classification for the headline.
## Setup
After cloning the repository, download the required libraries by running 
```
pip install -r requirements.txt
```
Then, create the model by running <b>train.py</b>. Once the model has finished training and has been saved, <b>main.py</b> can be run to scrape news headlines, and feed them through the model.
