import re
import numpy as np
from collections import Counter
import pandas as pd

# 1. Обробка тексту та створення словника частотності
def process_data(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        text = f.read().lower()
    words = re.findall(r'\w+', text)
    return words

def get_count(word_list):
    return Counter(word_list)

def get_probs(word_count_dict):
    total = sum(word_count_dict.values())
    return {word: count / total for word, count in word_count_dict.items()}

# 2. Функції редагування
def delete_letter(word):
    return [word[:i] + word[i+1:] for i in range(len(word))]

def switch_letter(word):
    return [word[:i] + word[i+1] + word[i] + word[i+2:] for i in range(len(word) - 1)]

def replace_letter(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return [word[:i] + c + word[i+1:] for i in range(len(word)) for c in letters if word[i] != c]

def insert_letter(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return [word[:i] + c + word[i:] for i in range(len(word) + 1) for c in letters]

# 3. Генерація кандидатів
def edit_one_letter(word):
    return set(delete_letter(word) + switch_letter(word) + replace_letter(word) + insert_letter(word))

def edit_two_letters(word):
    edits = edit_one_letter(word)
    return set(e2 for e1 in edits for e2 in edit_one_letter(e1))

# 4. Функція корекції
def get_corrections(word, probs, vocab, n=2):
    candidates = (
        {word} if word in vocab else
        edit_one_letter(word) & vocab or
        edit_two_letters(word) & vocab or
        {word}
    )
    candidates_probs = {w: probs.get(w, 0) for w in candidates}
    return sorted(candidates_probs.items(), key=lambda x: x[1], reverse=True)[:n]

# 5. Алгоритм мінімальної відстані редагування
def min_edit_distance(source, target, ins_cost=1, del_cost=1, rep_cost=2):
    m, n = len(source), len(target)
    D = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        D[i][0] = i * del_cost
    for j in range(n + 1):
        D[0][j] = j * ins_cost
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if source[i - 1] == target[j - 1] else rep_cost
            D[i][j] = min(
                D[i - 1][j] + del_cost,
                D[i][j - 1] + ins_cost,
                D[i - 1][j - 1] + cost
            )
    return D, D[m][n]

# 6. Основна програма
if __name__ == "__main__":
    # Завантаження корпусу
    word_list = process_data("shakespeare.txt")
    vocab = set(word_list)
    word_count_dict = get_count(word_list)
    probs = get_probs(word_count_dict)

    # Тестування автокорекції
    test_words = ["hllo", "mony", "comming", "happpy", "xylophonne"]
    for word in test_words:
        suggestions = get_corrections(word, probs, vocab, n=3)
        print(f"Корекції для '{word}':")
        for i, (cor, p) in enumerate(suggestions):
            print(f"  {i+1}. {cor} (ймовірність: {p:.6f})")
        print()

    # Тестування min_edit_distance
    test_pairs = [("intention", "execution"), ("play", "stay"), ("sunday", "saturday")]
    for source, target in test_pairs:
        D, dist = min_edit_distance(source, target)
        print(f"Відстань між '{source}' і '{target}': {dist}")
        print(pd.DataFrame(D, index=["#"] + list(source), columns=["#"] + list(target)))
        print()
