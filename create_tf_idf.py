from sqlite3 import DatabaseError
import numpy as np
import nltk
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import itertools

ROOT = "./Database"

DOCUMENTS_LYRICS = {}
DOCUMENTS_ABC = {}
DATASET = {}

LYRICS = "Lyrics/"
ABC = "ABC/"

AGGREGATE_ARTIST = {}


def populate_dataset_dict():
    for directory in os.listdir(ROOT):
        if os.path.isdir(ROOT + "/" + directory):
            DATASET[directory] = {}
            nested_path = ROOT + "/" + directory + "/Lyrics/"
            for nested_dir in os.listdir(ROOT + "/" + directory + "/Lyrics/"):
                if os.path.isdir(nested_path + nested_dir):
                    DATASET[directory][nested_dir] = [
                        x.replace(".txt", "")
                        for x in os.listdir(nested_path + nested_dir)
                    ]


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
        if dir_name == LYRICS:
            lines = lines.translate(str.maketrans("", "", string.punctuation))
            lines = lines.translate(str.maketrans("", "", string.digits))

            # Removes Non-ASCII letters from the string
            encoded_lines = lines.encode("ascii", "ignore")
            lines = encoded_lines.decode()
            DOCUMENTS_LYRICS[file] = lines

        elif dir_name == ABC:
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


def sanity_check(df, cosine_sim_df, song1, song2):
    # A check function that verifies all the values are in sync
    val1 = round(
        cosine_similarity(
            df.loc[song1].to_numpy().reshape(1, -1),
            df.loc[song2].to_numpy().reshape(1, -1),
        )[0][0],
        4,
    )

    val2 = round(cosine_sim_df.loc[song1, song2], 4)
    assert val1 == val2


def generate_pairs_of_artist():
    li = []
    for genre in DATASET:
        li.extend(DATASET[genre].keys())
    combinations = itertools.combinations(li, 2)
    return [x for x in combinations]


def main():
    populate_dataset_dict()

    for root, dirs, files in os.walk(ROOT):
        if LYRICS in root:
            populate_documents(root, files, LYRICS)
        elif ABC in root:
            populate_documents(root, files, ABC)

    df_lyrics, cosine_sim_lyrics, cosine_sim_lyrics_df = get_cosine_sim_mat(
        DOCUMENTS_LYRICS, stopwords="english"
    )
    # df_abc, cosine_sim_abc, cosine_sim_abc_df = get_cosine_sim_mat(
    #     DOCUMENTS_ABC, stopwords=None
    # )

    sanity_check(df_lyrics, cosine_sim_lyrics_df, "BookOfDays", "Blueberry Hill")
    # sanity_check(df_abc, cosine_sim_abc_df, "BookOfDays", "What A Wonderful World")

    for genre in DATASET:
        for artist in DATASET[genre]:
            AGGREGATE_ARTIST[artist] = {}
            temp_lyrics_df = cosine_sim_lyrics_df.loc[DATASET[genre][artist]][
                DATASET[genre][artist]
            ]
            temp_lyrics_df_mat = temp_lyrics_df.to_numpy()
            AGGREGATE_ARTIST[artist]["Lyrics"] = np.linalg.det(temp_lyrics_df_mat)

            # Uncomment when ABC files are present
            # temp_abc_df = cosine_sim_abc_df.loc[DATASET[genre][artist]][
            #     DATASET[genre][artist]
            # ]
            # temp_abc_df_mat = temp_abc_df.to_numpy()
            # AGGREGATE_ARTIST[artist]["ABC"] = np.linalg.det(temp_abc_df_mat)

    pairs_artists = generate_pairs_of_artist()


if __name__ == "__main__":
    main()
