import nltk
import numpy as np
import pandas as pd
import re
import string
from nltk.corpus import twitter_samples, stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

nltk.download('twitter_samples')
nltk.download('stopwords')

# Load dataset
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

# Preprocessing
def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?://[^\s

]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = [stemmer.stem(word) for word in tweet_tokens if word not in stopwords_english and word not in string.punctuation]
    return tweets_clean

def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs

freqs = build_freqs(train_x, train_y)

# Feature extraction
def extract_features(tweet, freqs):
    word_l = process_tweet(tweet)
    x = np.zeros((1, 3))
    x[0,0] = 1
    for word in word_l:
        x[0,1] += freqs.get((word, 1.0), 0)
        x[0,2] += freqs.get((word, 0.0), 0)
    return x

def sigmoid(z): 
    return 1 / (1 + np.exp(-z))

def gradientDescent(x, y, theta, alpha, num_iters):
    m = x.shape[0]
    for _ in range(num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)))
        theta = theta - (alpha/m) * np.dot(x.T, (h-y))
    return float(J), theta

X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)
Y = train_y.reshape((len(train_y), 1))

J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

# Testing
def predict_tweet(tweet, freqs, theta):
    x = extract_features(tweet, freqs)
    return sigmoid(np.dot(x, theta))

def test_logistic_regression(test_x, test_y, freqs, theta):
    y_hat = [1 if predict_tweet(tweet, freqs, theta) > 0.5 else 0 for tweet in test_x]
    accuracy = np.mean(np.array(y_hat) == np.squeeze(test_y))
    return accuracy

accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Final accuracy: {accuracy:.4f}")

# Example
my_tweet = "This movie was awesome and thrilling!"
pred = predict_tweet(my_tweet, freqs, theta)
print(f"Prediction for example tweet: {'Positive' if pred > 0.5 else 'Negative'}")
