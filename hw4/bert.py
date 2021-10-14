import json
import pandas as pd
import numpy as np
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
from string import digits, ascii_lowercase, punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm.notebook import tqdm
import nltk
import pickle as pkl
nltk.download('stopwords')

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
device = 'cuda:0'


def build_corpus(path):
    """Собирает леммы из файлов с текстами"""
    texts = []
    questions = []
    with open('questions_about_love.jsonl', 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]
    
    for line in corpus:
        string = json.loads(line)
        question = string['question']
        answers = string['answers']
        if answers != []:
            values = np.array(map(int, [i['author_rating']['value'] for i in answers if i !='']))
            answer = answers[np.argmax(values)]['text']
            texts.append(answer)
            questions.append(question)
    return texts, questions

def get_emb(sent):
    with torch.no_grad():
      enc = tokenizer(sent, padding=True, truncation=True, return_tensors='pt')
      enc.to(device)
      output = model(**enc, return_dict=True)
    return output.last_hidden_state[:,0,:].squeeze(0).cpu().detach().numpy()

def indexation(corpus):
    """Возвращает матрицу Document-Term"""
    model.to(device)
    X = np.zeros((len(corpus), 312))
    for i, sent in tqdm(enumerate(corpus)):
        X[i] = get_emb(sent)
    return X


def main():
    corpus, questions = build_corpus('questions_about_love.jsonl')
    matrix = indexation(corpus)
    np.save('bert', matrix, allow_pickle=True)
    question_matrix = indexation(questions)
    np.save('bert_questions', question_matrix, allow_pickle=True)


if __name__ == '__main__':
    main()