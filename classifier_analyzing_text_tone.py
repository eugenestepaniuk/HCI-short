import re
import string
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def process_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'https?://\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tokens = tweet.split()
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens if word not in stop_words]

def build_freqs(tweets, labels):
    freqs = defaultdict(int)
    for label, tweet in zip(labels, tweets):
        for word in process_tweet(tweet):
            freqs[(word, label)] += 1
    return freqs

def compute_log_prior(labels):
    pos = sum(1 for l in labels if l == 1)
    neg = sum(1 for l in labels if l == 0)
    return np.log(pos / neg)

def compute_log_likelihood(freqs, train_x, train_y):
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    N_pos, N_neg = 0, 0
    for pair in freqs:
        if pair[1] == 1:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]
    loglikelihood = {}
    for word in vocab:
        freq_pos = freqs.get((word, 1), 0)
        freq_neg = freqs.get((word, 0), 0)
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)
    return loglikelihood

def naive_bayes_predict(tweet, logprior, loglikelihood):
    words = process_tweet(tweet)
    score = logprior
    for word in words:
        score += loglikelihood.get(word, 0)
    return 1 if score > 0 else 0

def test_accuracy(test_x, test_y, logprior, loglikelihood):
    correct = 0
    for x, y in zip(test_x, test_y):
        y_pred = naive_bayes_predict(x, logprior, loglikelihood)
        if y_pred == y:
            correct += 1
    return correct / len(test_y)

if __name__ == "__main__":
    train_x = ["I love this!", "This is bad", "Amazing work", "Terrible movie", "Best day ever", "Worst day ever"]
    train_y = [1, 0, 1, 0, 1, 0]

    freqs = build_freqs(train_x, train_y)
    logprior = compute_log_prior(train_y)
    loglikelihood = compute_log_likelihood(freqs, train_x, train_y)

    test_x = ["I had the best time", "That was the worst"]
    test_y = [1, 0]

    acc = test_accuracy(test_x, test_y, logprior, loglikelihood)
    print(f"Accuracy: {acc:.2f}")

    for tweet in test_x:
        print(f"Tweet: '{tweet}' => Sentiment: {naive_bayes_predict(tweet, logprior, loglikelihood)}")
