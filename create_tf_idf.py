import numpy as np
import nltk
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

ROOT = "./Database"
ROOT_OP = "Tf-Idf"

DOCUMENTS = []


def tokenize(s):
    s = s.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(s)
    return tokens


def gen_tf_idf(s):
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words="english")
    vectorizer = tfidf.fit_transform(DOCUMENTS)
    return vectorizer


# for directory in os.listdir(ROOT):
#     if os.path.isdir(ROOT + "/" + directory):
#         for nested_directory in os.listdir(ROOT + "/" + directory):
#             if nested_directory == "Lyrics":
#                 for files in os.listdir('')

for root, dirs, files in os.walk(ROOT):
    if "Lyrics/" in root:
        print(root, dirs, files)
