import nltk
import re
import numpy as np
import random
from collections import Counter
import ipywidgets as widgets
from IPython.display import display, clear_output

nltk.download('punkt')

# --- Завантаження та обробка даних ---
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_to_sentences(data):
    sentences = data.split('\n')
    return [s.strip() for s in sentences if s.strip()]

def tokenize_sentences(sentences):
    return [nltk.word_tokenize(sentence.lower()) for sentence in sentences]

def count_words(tokenized_sentences):
    word_counts = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            word_counts[token] = word_counts.get(token, 0) + 1
    return word_counts

def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    word_counts = count_words(tokenized_sentences)
    return [word for word, count in word_counts.items() if count >= count_threshold]

def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    vocabulary = set(vocabulary)
    return [[token if token in vocabulary else unknown_token for token in sentence]
            for sentence in tokenized_sentences]

def preprocess_data(train_data, test_data, count_threshold):
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary)
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary)
    return train_data_replaced, test_data_replaced, vocabulary

# --- N-грамна модель ---
def count_n_grams(data, n, start_token='<s>', end_token='</s>'):
    n_grams = {}
    for sentence in data:
        sentence = [start_token] * (n - 1) + sentence + [end_token]
        for i in range(len(sentence) - n + 1):
            n_gram = tuple(sentence[i:i + n])
            n_grams[n_gram] = n_grams.get(n_gram, 0) + 1
    return n_grams

def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    previous_n_gram = tuple(previous_n_gram)
    denominator = n_gram_counts.get(previous_n_gram, 0) + k * vocabulary_size
    numerator = n_plus1_gram_counts.get(previous_n_gram + (word,), 0) + k
    return numerator / denominator

def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
    vocabulary = list(vocabulary) + ['</s>', '<unk>']
    vocabulary_size = len(vocabulary)
    return {word: estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k)
            for word in vocabulary}

def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    n = len(next(iter(n_gram_counts)))
    sentence = ['<s>'] * n + sentence + ['</s>']
    N = len(sentence)
    product_pi = 1.0
    for t in range(n, N):
        n_gram = sentence[t - n:t]
        word = sentence[t]
        prob = estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k)
        product_pi *= 1 / prob
    return product_pi ** (1 / N)

# --- Інтерфейс користувача ---
def create_ui(train_data_processed, vocabulary):
    n_gram_counts_list = [count_n_grams(train_data_processed, n) for n in range(1, 6)]

    text_input = widgets.Text(value='', placeholder='Введіть текст...', description='Текст:', layout=widgets.Layout(width='500px'))
    prefix_input = widgets.Text(value='', placeholder='Початок слова...', description='Префікс:', layout=widgets.Layout(width='500px'))
    n_gram_dropdown = widgets.Dropdown(options=[(f'{i}-грама', i) for i in range(1, 6)], value=2, description='N-грама:')
    k_slider = widgets.FloatSlider(value=1.0, min=0.01, max=5.0, step=0.01, description='K:', orientation='horizontal', readout_format='.2f')
    suggestion_output = widgets.Output()

    def update_suggestions(_):
        with suggestion_output:
            clear_output()
            text = text_input.value.strip()
            if not text:
                print("Введіть текст.")
                return
            tokens = text.lower().split()
            n = n_gram_dropdown.value
            k = k_slider.value
            start_with = prefix_input.value if prefix_input.value else None
            if n == 1:
                previous = []
            else:
                previous = tokens[-(n - 1):] if len(tokens) >= n - 1 else ['<s>'] * (n - 1 - len(tokens)) + tokens
            probs = estimate_probabilities(previous, n_gram_counts_list[n - 2], n_gram_counts_list[n - 1], vocabulary, k)
            if start_with:
                probs = {w: p for w, p in probs.items() if w.startswith(start_with) and w not in ['<s>', '</s>', '<unk>']}
            else:
                probs = {w: p for w, p in probs.items() if w not in ['<s>', '</s>', '<unk>']}
            top = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            if not top:
                print("Немає пропозицій.")
                return
            print(f"Пропозиції для '{text}':")
            for word, prob in top:
                btn = widgets.Button(description=f"{word} ({prob:.4f})", tooltip=f"Ймовірність: {prob:.4f}")
                btn.on_click(lambda b, w=word: setattr(text_input, 'value', text + ' ' + w))
                display(btn)

    suggest_button = widgets.Button(description='Запропонувати')
    suggest_button.on_click(update_suggestions)

    display(text_input, prefix_input, widgets.HBox([n_gram_dropdown, k_slider]), suggest_button, suggestion_output)

# --- Запуск системи ---
def run_autocomplete_system(file_path):
    print("Завантаження даних...")
    data = load_data(file_path)
    print("Токенізація...")
    sentences = split_to_sentences(data)
    tokenized = tokenize_sentences(sentences)
    random.seed(42)
    random.shuffle(tokenized)
    train_size = int(len(tokenized) * 0.8)
    train_data = tokenized[:train_size]
    test_data = tokenized[train_size:]
    train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data, test_data, count_threshold=2)
    print("Створення інтерфейсу...")
    create_ui(train_data_processed, vocabulary)
