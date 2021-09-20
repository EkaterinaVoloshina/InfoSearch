import os
import pandas as pd
import numpy as np
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
from string import digits, ascii_lowercase, punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

morph = MorphAnalyzer()
tokenizer = WordPunctTokenizer()
stop = set(stopwords.words("russian"))
vectorizer = TfidfVectorizer()


def preprocessing(text):
    """Убирает стоп-слова, цифры и слова на латинице,
    возвращает леммы"""
    t = tokenizer.tokenize(text.lower())
    lemmas = [morph.parse(word)[0].normal_form for word in t
              if word not in punctuation and word not in stop and not set(word).intersection(digits)
              and not set(word).intersection(ascii_lowercase)]
    return lemmas


def build_corpus(path):
    """Собирает леммы из файлов с текстами"""
    corpus = []
    titles = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            dir = os.path.join(root, dir)
            for r, d, file in os.walk(dir):
                for f in file:
                    titles.append(f.replace('.ru.txt', ''))
                    filename = os.path.join(r, f)
                    with open(filename, encoding='utf-8') as f:
                        text = f.read()
                    lemmas = preprocessing(text)
                    corpus.append(lemmas)
    return corpus, titles


def indexation(corpus):
    """Возвращает матрицу Document-Term"""
    X = vectorizer.fit_transform([' '.join(i) for i in corpus])
    names = vectorizer.get_feature_names()
    df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
    return X.toarray()


def query_indexation(query):
    """Преобразовывает запрос в вектор"""
    return vectorizer.transform([' '.join(query)]).toarray()


def count_cos(query, corpus):
    """Считает косинусную близость"""
    return cosine_similarity(query, corpus)[0]


def find_docs(query, corpus, names):
    """Выполняет поиск"""
    lemmas = preprocessing(query)
    if lemmas != []:
        query_index = query_indexation(lemmas)
        cos = count_cos(query_index, corpus)
        ind = np.argsort(cos)
        return np.array(names)[ind][::-1]
    else:
        return ['В Вашем запросе только цифры, пунктуация или латиница. Попробуйте еще раз!']


def main():
    corpus, titles = build_corpus('./friends-data')
    matrix = indexation(corpus)
    query = input('Введите свой запрос: ')
    while query != '':
        docs = find_docs(query, matrix, titles)
        print('Ищем документы по запросу...')
        print(*docs[:20], sep='\n')
        print('Хотите отправить новый запрос? Если нет, нажмите Enter')
        query = input('Введите свой запрос: ')


if __name__ == "__main__":
    main()
