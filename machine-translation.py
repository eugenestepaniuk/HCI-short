import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
import random

# 1-2. Завантаження ембедінгів та словників

def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vec = np.array(values[1:], dtype='float32')
            embeddings[word] = vec
    return embeddings

def load_dictionary(file_path):
    pairs = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            word1, word2 = line.strip().split()
            pairs.append((word1, word2))
    return pairs

# 3. Побудова матриць X, Y 

def build_matrices(pairs, emb1, emb2):
    X, Y = [], []
    for w1, w2 in pairs:
        if w1 in emb1 and w2 in emb2:
            X.append(emb1[w1])
            Y.append(emb2[w2])
    return np.array(X), np.array(Y)

# 4. Градієнтний спуск для пошуку матриці перетворення 

def gradient_descent(X, Y, lr=0.01, epochs=1000):
    R = np.random.rand(X.shape[1], Y.shape[1])
    for i in range(epochs):
        gradient = 2 * X.T @ (X @ R - Y)
        R -= lr * gradient
    return R

# 5. Переклад слів

def translate_word(word, emb1, emb2, R):
    if word not in emb1:
        return None
    vec = emb1[word] @ R
    best_match = max(emb2.items(), key=lambda item: cosine_similarity([vec], [item[1]])[0][0])
    return best_match[0]

# 6. Оцінка точності перекладу

def evaluate(pairs, emb1, emb2, R):
    correct = 0
    total = 0
    for w1, w2 in pairs:
        trans = translate_word(w1, emb1, emb2, R)
        if trans == w2:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

# 7-8. Завантаження твітів і перетворення у вектори 

def tweet_to_vec(tweet, embedding):
    vectors = [embedding[word] for word in tweet.split() if word in embedding]
    return np.sum(vectors, axis=0) if vectors else np.zeros(len(next(iter(embedding.values()))))

# 9. LSH 

def create_lsh_hashes(vectors, num_planes=10):
    dim = vectors.shape[1]
    planes = [np.random.randn(dim) for _ in range(num_planes)]
    hashes = []
    for vec in vectors:
        hash_bits = ''.join(['1' if np.dot(vec, plane) >= 0 else '0' for plane in planes])
        hashes.append(hash_bits)
    return hashes

# 10. Пошук подібних повідомлень

def search_similar_tweets(query_vec, tweet_vectors, tweet_texts, hashes):
    query_hash = create_lsh_hashes([query_vec])[0]
    candidates = [i for i, h in enumerate(hashes) if h == query_hash]
    if not candidates:
        return "No similar tweets found."
    similarities = [cosine_similarity([query_vec], [tweet_vectors[i]])[0][0] for i in candidates]
    most_similar = candidates[np.argmax(similarities)]
    return tweet_texts[most_similar]

# Основна логіка

if __name__ == "__main__":
    # Шляхи до файлів
    emb1_path = "lang1.vec.txt"
    emb2_path = "lang2.vec.txt"
    train_dict = "l1-l2.train.txt"
    test_dict = "l1-l2.test.txt"
    tweets_file = "tweets.txt"

    # Завантаження
    emb1 = load_embeddings(emb1_path)
    emb2 = load_embeddings(emb2_path)
    train_pairs = load_dictionary(train_dict)
    test_pairs = load_dictionary(test_dict)

    # Побудова матриць
    X, Y = build_matrices(train_pairs, emb1, emb2)

    # Навчання
    R = gradient_descent(X, Y, lr=0.01, epochs=1000)

    # Оцінка
    acc = evaluate(test_pairs, emb1, emb2, R)
    print(f"Translation accuracy: {acc:.2f}")

    # Завантаження твітів
    with open(tweets_file, encoding='utf-8') as f:
        tweets = [line.strip() for line in f.readlines()]

    tweet_vectors = np.array([tweet_to_vec(tweet, emb1) for tweet in tweets])
    hashes = create_lsh_hashes(tweet_vectors)

    # Пошук подібного до першого твіту
    print("Original tweet:", tweets[0])
    result = search_similar_tweets(tweet_vectors[0], tweet_vectors, tweets, hashes)
    print("Most similar tweet:", result)
