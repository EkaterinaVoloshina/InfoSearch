import os
import json
import numpy as np
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
from string import digits, ascii_lowercase, punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from scipy import sparse

morph = MorphAnalyzer()
tokenizer = WordPunctTokenizer()
stop = set(stopwords.words("russian"))
count_vectorizer = CountVectorizer()
tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')


def preprocessing(text):
    """Убирает стоп-слова, цифры и слова на латинице,
    возвращает леммы"""
    t = tokenizer.tokenize(text.lower())
    lemmas = [morph.parse(word)[0].normal_form for word in t
              if word not in punctuation and word not in stop and not set(word).intersection(digits)
              and not set(word).intersection(ascii_lowercase)]
    return ' '.join(lemmas)


def build_corpus(path):
    """Собирает леммы из файлов с текстами"""
    texts = []
    lemmas = []
    with open(path, 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]
    
    for line in corpus:
        answers = json.loads(line)['answers']
        if answers:
            values = np.array(map(int, [i['author_rating']['value'] for i in answers if i != '']))
            answer = answers[np.argmax(values)]['text']
            texts.append(answer)
            lemmas.append(preprocessing(answer))
    return texts, lemmas


def indexation(corpus, k=2, b=0.75):
    """Возвращает матрицу Document-Term"""
    x_count = count_vectorizer.fit_transform(corpus)
    x_tf = tf_vectorizer.fit_transform(corpus)  # tf matrix
    x_idf = tfidf_vectorizer.fit_transform(corpus)
    idf = tfidf_vectorizer.idf_  # idf matrix
    len_d = x_count.sum(axis=1)
    avdl = len_d.mean()
    lengths = k * (1 - b + b * len_d / avdl)
    matrix = sparse.lil_matrix(x_tf.shape)
    for i, j in zip(*x_tf.nonzero()):
        matrix[i, j] = (x_tf[i, j] * (k + 1) * idf[j])/(x_tf[i, j] + lengths[i])
    return matrix.tocsr()


def query_indexation(query):
    """Преобразовывает запрос в вектор"""
    return count_vectorizer.transform([query])


def count_bm25(query, corpus):
    """Считает близость по BM25"""
    return corpus.dot(query.T)


def find_docs(query, corpus, answers):
    """Выполняет поиск"""
    lemmas = preprocessing(query)
    if lemmas:
        query_index = query_indexation(lemmas)
        bm25 = count_bm25(query_index, corpus)
        ind = np.argsort(bm25.toarray(), axis=0)
        return np.array(answers)[ind][::-1].squeeze()
    else:
        return ['В Вашем запросе только цифры, пунктуация или латиница. Попробуйте еще раз!']


def main():
    corpus, lemmas = build_corpus('./questions_about_love.jsonl')
    matrix = indexation(lemmas)
    query = input('Введите свой запрос: ')
    while query != '':
        docs = find_docs(query, matrix, corpus)
        print('Ищем документы по запросу...')
        print(*docs[:20], sep='\n')
        print('Хотите отправить новый запрос? Если нет, нажмите Enter')
        query = input('Введите свой запрос: ')


if __name__ == '__main__':
    main()
