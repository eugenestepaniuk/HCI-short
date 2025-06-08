import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict
import ipywidgets as widgets
from IPython.display import display

# Підготовка корпусу
corpus = "вивчення мови це цікавий процес мова допомагає нам спілкуватися з іншими людьми".split()
window_size = 2
embedding_dim = 10
epochs = 500
learning_rate = 0.01

# Унікальні слова
vocab = list(set(corpus))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}
vocab_size = len(vocab)

# Генерація (контекст, центр)
def generate_data(corpus, window_size):
    data = []
    for i in range(window_size, len(corpus) - window_size):
        context = [corpus[j] for j in range(i - window_size, i + window_size + 1) if j != i]
        target = corpus[i]
        data.append((context, target))
    return data

data = generate_data(corpus, window_size)

# One-hot encoding
def one_hot(idx):
    vec = np.zeros(vocab_size)
    vec[idx] = 1
    return vec

# Ініціалізація ваг
W1 = np.random.randn(vocab_size, embedding_dim)
W2 = np.random.randn(embedding_dim, vocab_size)

# Softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Тренування CBOW
for epoch in range(epochs):
    loss = 0
    for context, target in data:
        context_vec = sum([one_hot(word_to_idx[w]) for w in context]) / len(context)
        h = np.dot(W1.T, context_vec)
        u = np.dot(W2.T, h)
        y_pred = softmax(u)

        target_idx = word_to_idx[target]
        e = y_pred
        e[target_idx] -= 1

        dW2 = np.outer(h, e)
        dW1 = np.outer(context_vec, np.dot(W2, e))

        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2

        loss += -np.log(y_pred[target_idx])
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Вектори слів
word_embeddings = W1

# Збереження
np.save("word_embeddings.npy", word_embeddings)
with open("vocab.txt", "w") as f:
    for w in vocab:
        f.write(w + "\n")

# Візуалізація PCA
def plot_pca(word_embeddings, vocab):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(word_embeddings)

    plt.figure(figsize=(10, 7))
    for i, word in enumerate(vocab):
        plt.scatter(reduced[i, 0], reduced[i, 1])
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]))
    plt.title("Word Embeddings PCA")
    plt.grid()
    plt.show()

plot_pca(word_embeddings, vocab)

# Інтерфейс для тестування
def display_similar_words(word, top_n=5):
    if word not in word_to_idx:
        print("Слова немає в словнику.")
        return

    vec = word_embeddings[word_to_idx[word]]
    sims = {}
    for other in vocab:
        if other == word:
            continue
        other_vec = word_embeddings[word_to_idx[other]]
        sim = np.dot(vec, other_vec) / (np.linalg.norm(vec) * np.linalg.norm(other_vec))
        sims[other] = sim

    sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"Найбільш подібні до '{word}':")
    for w, score in sorted_sims:
        print(f"{w}: {score:.4f}")

dropdown = widgets.Dropdown(options=vocab, description='Слово:')
button = widgets.Button(description='Пошук')
output = widgets.Output()

def on_button_clicked(b):
    output.clear_output()
    with output:
        display_similar_words(dropdown.value)

button.on_click(on_button_clicked)
display(dropdown, button, output)
