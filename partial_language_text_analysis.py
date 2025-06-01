import numpy as np
import nltk
from collections import defaultdict
from nltk.corpus import treebank
from sklearn.model_selection import train_test_split

nltk.download('treebank')

# 1. Завантаження даних
tagged_sentences = treebank.tagged_sents()
train_data, test_data = train_test_split(tagged_sentences, test_size=0.2, random_state=42)

# 2. Частоти переходів і емісій
transition_counts = defaultdict(lambda: defaultdict(int))
emission_counts = defaultdict(lambda: defaultdict(int))
tag_counts = defaultdict(int)
vocab = set()

for sentence in train_data:
    prev_tag = "<s>"
    tag_counts[prev_tag] += 1
    for word, tag in sentence:
        transition_counts[prev_tag][tag] += 1
        emission_counts[tag][word.lower()] += 1
        tag_counts[tag] += 1
        prev_tag = tag
        vocab.add(word.lower())

# 3. Матриця переходів
def transition_prob(tag1, tag2):
    return (transition_counts[tag1][tag2] + 1) / (tag_counts[tag1] + len(tag_counts))

# 4. Матриця емісій
def emission_prob(tag, word):
    return (emission_counts[tag][word.lower()] + 1) / (tag_counts[tag] + len(vocab))

# 5. Алгоритм Вітербі
def viterbi(words, tags):
    n = len(words)
    m = len(tags)

    V = np.zeros((m, n))
    B = np.zeros((m, n), dtype=int)

    # Ініціалізація
    for i in range(m):
        V[i][0] = transition_prob("<s>", tags[i]) * emission_prob(tags[i], words[0])

    # Рекурсія
    for t in range(1, n):
        for j in range(m):
            max_prob = 0
            best_i = 0
            for i in range(m):
                prob = V[i][t-1] * transition_prob(tags[i], tags[j]) * emission_prob(tags[j], words[t])
                if prob > max_prob:
                    max_prob = prob
                    best_i = i
            V[j][t] = max_prob
            B[j][t] = best_i

    # Зворотний прохід
    best_path = []
    last_tag = np.argmax(V[:, n-1])
    best_path.append(tags[last_tag])
    for t in range(n-1, 0, -1):
        last_tag = B[last_tag][t]
        best_path.insert(0, tags[last_tag])

    return best_path

# 6. Тестування
def evaluate(test_data):
    tags = list(tag_counts.keys())
    correct = 0
    total = 0

    for sentence in test_data:
        words = [word for word, tag in sentence]
        true_tags = [tag for word, tag in sentence]
        pred_tags = viterbi(words, tags)

        for t1, t2 in zip(true_tags, pred_tags):
            if t1 == t2:
                correct += 1
            total += 1

    return correct / total

# 7. Порівняння з NLTK
def nltk_pos_accuracy(test_data):
    correct = 0
    total = 0
    for sentence in test_data:
        words = [w for w, t in sentence]
        true_tags = [t for w, t in sentence]
        pred_tags = [t for w, t in nltk.pos_tag(words)]
        for t1, t2 in zip(true_tags, pred_tags):
            if t1 == t2:
                correct += 1
            total += 1
    return correct / total

# Виконання оцінки
print("HMM + Viterbi accuracy:", evaluate(test_data))
print("NLTK pos_tag accuracy:", nltk_pos_accuracy(test_data))
