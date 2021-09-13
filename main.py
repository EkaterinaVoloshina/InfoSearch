import os
import pandas as pd
import numpy as np
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
from string import punctuation, digits, ascii_lowercase
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


def preprocessing(text, morph, tokenizer, stop):
    """Убирает стоп-слова, цифры и слова на латинице,
    возвращает леммы"""
    text = tokenizer.tokenize(text.lower())
    lemmas = [morph.parse(word)[0].normal_form for word in text
              if word not in punctuation and word not in stop
              and not set(word).intersection(digits)
              and not set(word).intersection(ascii_lowercase)]
    return lemmas


def build_corpus(path, morph, tokenizer, stop):
    """Собирает леммы из файлов с текстами"""
    corpus = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            root_dir = os.path.join(root, dir)
            for r, d, file in os.walk(root_dir):
                for f in file:
                    filename = os.path.join(r, f)
                    with open(filename, encoding='utf-8') as t:
                        text = t.read()
                    lemmas = preprocessing(text, morph, tokenizer, stop)
                    corpus.append(lemmas)
    return corpus


def indexation(vectorizer, corpus):
    """Возвращает матрицу Term-Document"""
    X = vectorizer.fit_transform([' '.join(i) for i in corpus])
    df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
    return df


def find_answers(df):
    """Ищет самое частое слово, самое редкое слово,
    слова, которые встречаются во всех текстах и
    имя самого популярного главного героя"""
    characters = {}
    X = np.asarray(df.to_numpy().sum(axis=0)).ravel()
    names = df.columns.to_numpy()
    characters['моника'] = np.sum(X[(names == 'моника') | (names == 'мон')])
    characters['чендлер'] = np.sum(X[(names == 'чендлер') | (names == 'чэндлер') | (names == 'чэндлер')])
    characters['фиби'] = np.sum(X[(names == 'фиби') | (names == 'фибс')])
    characters['росс'] = X[names == 'росс'][0]
    characters['рэйчел'] = np.sum(X[(names == 'рэйчел') | (names == 'рейч')])
    characters['джоуи'] = np.sum(X[(names == 'джоуи') | (names == 'джои') | (names == 'джо')])

    print('Самое частотное слово:', ' '.join(names[np.where(X == X.max())]))

    print('Самое редкое слово:', names[np.where(X == X.min())][0])

    words = df.loc[:, (df != 0).all(axis=0) == True].columns.to_list()
    print('Слова, которые встречаются во всех документах:', ', '.join(words))
    char = max(characters, key=characters.get)
    print('Самый популярный главный герой:', char.title())


def main():
    morph = MorphAnalyzer()
    tokenizer = WordPunctTokenizer()
    stop = set(stopwords.words("russian"))
    vectorizer = CountVectorizer(analyzer='word')
    corpus = build_corpus('./friends-data', morph, tokenizer, stop)
    df = indexation(vectorizer, corpus)
    find_answers(df)


if __name__ == "__main__":
    main()
