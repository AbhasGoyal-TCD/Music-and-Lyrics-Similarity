from tracemalloc import stop
import numpy as np
import nltk
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

ROOT = "./Database"
ROOT_OP = "Tf-Idf"

DOCUMENTS_LYRICS = {}
DOCUMENTS_ABC = {}


def tokenize(s, lemmatization=True):
    tokens = nltk.word_tokenize(s)
    if lemmatization:
        lemmas = []
        lemmatizer = WordNetLemmatizer()
        for item in tokens:
            lemmas.append(lemmatizer.lemmatize(item))
        return lemmas
    return tokens


def gen_tf_idf(s, stopwords):
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=stopwords)
    tfidf_vector = tfidf.fit_transform(s)
    return tfidf, tfidf_vector


def read_file(path):
    with open(path) as f:
        lines = f.readlines()
    return "".join(lines)


def populate_documents(root, files, dir_name):
    for file in files:
        lines = read_file(root + "/" + file)
        file = file.replace(".txt", "").replace(".abc", "")
        if dir_name == "Lyrics":
            lines = lines.translate(str.maketrans("", "", string.punctuation))
            lines = lines.translate(str.maketrans("", "", string.digits))

            # Removes Non-ASCII letters from the string
            encoded_lines = lines.encode("ascii", "ignore")
            lines = encoded_lines.decode()
            DOCUMENTS_LYRICS[file] = lines

        elif dir_name == "ABC":
            DOCUMENTS_ABC[file] = lines


def get_cosine_sim_mat(documents, stopwords):
    # Generates TF-IDF values and then a cosine similarity matrix
    tfidf, tfidf_vector = gen_tf_idf(documents.values(), stopwords)
    df = pd.DataFrame(
        tfidf_vector.toarray(),
        index=documents.keys(),
        columns=tfidf.get_feature_names_out(),
    )

    cosine_sim = cosine_similarity(tfidf_vector, tfidf_vector)
    cosine_sim_df = pd.DataFrame(
        cosine_sim, index=documents.keys(), columns=documents.keys()
    )

    return df, cosine_sim, cosine_sim_df


def sanity_check(df_lyrics, cosine_sim_lyrics_df, song1, song2):
    # A check function that verifies all the values are in sync
    val1 = round(
        cosine_similarity(
            df_lyrics.loc[song1].to_numpy().reshape(1, -1),
            df_lyrics.loc[song2].to_numpy().reshape(1, -1),
        )[0][0],
        4,
    )

    val2 = round(cosine_sim_lyrics_df.loc[song1, song2], 4)
    print(val1, val2)
    assert val1 == val2


def main():
    for root, dirs, files in os.walk(ROOT):
        if "Lyrics/" in root:
            populate_documents(root, files, "Lyrics")
        elif "ABC/" in root:
            populate_documents(root, files, "ABC")

    df_lyrics, cosine_sim_lyrics, cosine_sim_lyrics_df = get_cosine_sim_mat(
        DOCUMENTS_LYRICS, stopwords="english"
    )
    df_abc, cosine_sim_abc, cosine_sim_abc_df = get_cosine_sim_mat(
        DOCUMENTS_ABC, stopwords=None
    )

    sanity_check(df_lyrics, cosine_sim_lyrics_df, "BookOfDays", "Blueberry Hill")
    sanity_check(df_abc, cosine_sim_abc_df, "BookOfDays", "What A Wonderful World")


if __name__ == "__main__":
    main()
