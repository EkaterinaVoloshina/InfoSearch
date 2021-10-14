import json
import numpy as np
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
from string import digits, ascii_lowercase, punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import tarfile
from gensim import models
from scipy import sparse

morph = MorphAnalyzer()
tokenizer = WordPunctTokenizer()
stop = set(stopwords.words("russian"))
#  tar = tarfile.open('araneum_none_fasttextcbow_300_5_2018.tgz', "r:gz")
#  tar.extractall()
#  tar.close()

fasttext = models.KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
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
    questions = []
    with open(path, 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]
    
    for line in corpus:
        string = json.loads(line)
        question = string['question']
        answers = string['answers']
        if answers:
            values = np.array(map(int, [i['author_rating']['value'] for i in answers if i != '']))
            answer = answers[np.argmax(values)]['text']
            texts.append(answer)
            lemmas.append(preprocessing(answer))
            questions.append(preprocessing(question))
    return texts, lemmas, questions


def sentence_embedding(sentence):
    """
    Складывает вектора токенов строки sentence
    """
    words = sentence.split()
    vectors = np.zeros(300)
    for i in words:
        if i in fasttext:
            vectors += fasttext[i]
    return vectors/len(vectors)


def indexation(corpus, k=2, b=0.75):
    """Возвращает матрицу Document-Term"""
    bow_corpus = count_vectorizer.fit_transform(corpus)
    tf_corpus = tf_vectorizer.fit_transform(corpus)
    tfidf_corpus = tfidf_vectorizer.fit_transform(corpus)
    bert_corpus = np.load('bert.npy')
    ft_corpus = np.zeros((len(corpus), 300))
    for i, item in enumerate(corpus):
        ft_corpus[i] = sentence_embedding(item)
        
    idf = tfidf_vectorizer.idf_  # idf matrix
    len_d = bow_corpus.sum(axis=1)
    avdl = len_d.mean()
    lengths = k * (1 - b + b * len_d / avdl)
    bm_corpus = sparse.lil_matrix(tf_corpus.shape)
    for i, j in zip(*tf_corpus.nonzero()):
        bm_corpus[i, j] = (tf_corpus[i, j] * (k + 1) * idf[j])/(tf_corpus[i, j] + lengths[i])
    return bow_corpus, tfidf_corpus, ft_corpus, bm_corpus.tocsr(), bert_corpus

                                                           
def query_indexation(corpus, k=2, b=0.75):
    """Возвращает матрицу Document-Term"""
    bow_corpus = count_vectorizer.transform(corpus)
    tfidf_corpus = tfidf_vectorizer.transform(corpus)
    bert_corpus = np.load('bert_questions.npy')
    ft_corpus = np.zeros((len(corpus), 300))
    for i, item in enumerate(corpus):
        ft_corpus[i] = sentence_embedding(item)
    return bow_corpus, tfidf_corpus, ft_corpus, bert_corpus                                                           
                                                    

def count_cos(query, corpus):
    """Считает косинусную близость"""
    return cosine_similarity(query, corpus)


def calculate_metrics(questions, answers):
    """Считает метрики по запросам"""
    metrics = 0
    sim = count_cos(questions, answers)
    for i, string in enumerate(sim):
        index = np.argsort(string, axis=0)[::-1]
        if i in index[:5]:
            metrics += 1
    return metrics/sim.shape[0]


def main():
    corpus, lemmas, questions = build_corpus('./questions_about_love.jsonl')
    print('Векторизую корпус...')
    bow_corpus, tfidf_corpus, ft_corpus, bm_corpus, bert_corpus = indexation(lemmas)
    print('Векторизую запросы...')
    bow_questions, tfidf_questions, ft_questions, bert_answers = query_indexation(questions)
    print('Считаю метрики...')
    bow_metrics = calculate_metrics(bow_corpus, bow_questions)
    print('Метрика для Bag of Words:', bow_metrics)
    tfidf_metrics = calculate_metrics(tfidf_corpus, tfidf_questions)
    print('Метрика для TF-IDF:', tfidf_metrics)
    ft_metrics = calculate_metrics(ft_corpus, ft_questions)
    print('Метрика для FastText:', ft_metrics)
    bm_metrics = calculate_metrics(bm_corpus, bow_questions)
    print('Метрика для BM25:', bm_metrics)
    bert_metrics = calculate_metrics(bert_corpus, bert_answers)
    print('Метрика для BERT:', bert_metrics)


if __name__ == '__main__':
    main()
