import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Завантаження даних про столиці та країни
url_capitals = "https://raw.githubusercontent.com/lang-uk/vecs/refs/heads/master/test/test_vocabulary.txt"
data = pd.read_csv(url_capitals, delimiter='\t', header=None, skiprows=1, names=['country1', 'city1', 'country2', 'city2'])

# 2. Завантаження моделі Word2Vec
model = KeyedVectors.load_word2vec_format("ubercorpus.cased.tokenized.word2vec.300d", binary=False)

# 3. Косинусна подібність
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# 4. Евклідова відстань
def euclidean_distance(u, v):
    return np.linalg.norm(u - v)

# 5. Пошук країни для аналогії
def get_country(city1, country1, city2, word_embeddings, cosine_similarity=cosine_similarity):
    group = {city1, country1, city2}
    vec = word_embeddings[city2] - word_embeddings[city1] + word_embeddings[country1]
    similarity = -1
    country = ''
    for word in word_embeddings.index_to_key:
        if word not in group:
            cur_similarity = cosine_similarity(vec, word_embeddings[word])
            if cur_similarity > similarity:
                similarity = cur_similarity
                country = (word, similarity)
    return country

# 6. Обчислення точності
def get_accuracy(word_embeddings, data):
    num_correct = 0
    for i, row in data.iterrows():
        country1, city1, country2, city2 = row
        predicted_country2, _ = get_country(city1, country1, city2, word_embeddings)
        if predicted_country2 == country2:
            num_correct += 1
    return num_correct / len(data)

# 7. Візуалізація слів
def visualize(words, model):
    valid_words = [w for w in words if w in model.key_to_index]
    word_vectors = np.array([model[w] for w in valid_words])
    pca = PCA(n_components=2)
    components = pca.fit_transform(word_vectors)
    plt.scatter(components[:, 0], components[:, 1])
    for i, word in enumerate(valid_words):
        plt.annotate(word, xy=(components[i, 0], components[i, 1]))
    plt.show()

# Виконання
accuracy = get_accuracy(model, data)
print(f"Точність: {accuracy:.2f}")
visualize(["Україна", "Київ", "Польща", "Варшава", "Німеччина", "Берлін"], model)
