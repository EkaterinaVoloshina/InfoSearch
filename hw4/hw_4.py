import json
import numpy as np
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
from string import digits, ascii_lowercase, punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import torch
import tarfile
from gensim import models
import transformers

morph = MorphAnalyzer()
wpt_tokenizer = WordPunctTokenizer()
stop = set(stopwords.words("russian"))
#tar = tarfile.open('araneum_none_fasttextcbow_300_5_2018.tgz', "r:gz")
#tar.extractall()
#tar.close()

fasttext = models.KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocessing(text):
    """Убирает стоп-слова, цифры и слова на латинице,
    возвращает леммы"""
    t = wpt_tokenizer.tokenize(text.lower())
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


def get_emb(sent):
    model.to(device)
    with torch.no_grad():
      enc = tokenizer(sent, padding=True, truncation=True, return_tensors='pt')
      enc.to(device)
      output = model(**enc, return_dict=True)
    return output.last_hidden_state[:,0,:].squeeze(0).cpu().detach().numpy()


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


def indexation(corpus):
    """Возвращает матрицу Document-Term"""
    bert_X = np.load('bert.npy')
    X = np.zeros((len(corpus), 300))
    for i, item in enumerate(corpus):
        X[i] = sentence_embedding(item)
    return bert_X, X


def query_indexation(query, model='fasttext'):
    """Преобразовывает запрос в вектор"""
    if model == 'bert':
        return get_emb(query)
    else:
        return sentence_embedding(query)


def count_cos(query, corpus):
    """Считает косинусную близость"""
    return cosine_similarity(query, corpus)[0]


def find_docs(query, corpus, answers, model='fasttext'):
    """Выполняет поиск"""
    lemmas = preprocessing(query)
    if lemmas:
        if model == 'bert':
            query_index = query_indexation(query, 'bert')
        else:
            query_index = query_indexation(lemmas)
        sim = count_cos(np.expand_dims(query_index, axis=0), corpus)
        ind = np.argsort(sim, axis=0)
        print(ind.shape)
        return np.array(answers)[ind][::-1].squeeze()
    else:
        return ['В Вашем запросе только цифры, пунктуация или латиница. Попробуйте еще раз!']


def main():
    corpus, lemmas = build_corpus('questions_about_love.jsonl')
    bert_matrix, matrix = indexation(lemmas)
    query = input('Введите свой запрос: ')
    while query != '':
        model = input('По умолчанию используется FastText. Хотите использовать BERT? y/n: ')
        if model.lower() == 'y':
            docs = find_docs(query, bert_matrix, corpus, model='bert')
        else:
            docs = find_docs(query, matrix, corpus)
        print('Ищем документы по запросу...')
        print(*docs[:5], sep='\n')
        print('Хотите отправить новый запрос? Если нет, нажмите Enter')
        query = input('Введите свой запрос: ')


if __name__ == '__main__':
    main()