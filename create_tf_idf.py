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
from scipy.stats import mannwhitneyu

ROOT = "./Database"

DOCUMENTS_LYRICS = {}
DOCUMENTS_ABC = {}
DATASET = {}

LYRICS = "Lyrics/"
ABC = "ABC_processed/"

AGGREGATE_ARTIST = {}
AGGREGATE_PAIR_ARTISTS = {}


def populate_dataset_dict():
    for directory in os.listdir(ROOT):
        if os.path.isdir(ROOT + "/" + directory):
            DATASET[directory] = {}
            nested_path = ROOT + "/" + directory + "/Lyrics/"
            for nested_dir in os.listdir(ROOT + "/" + directory + "/Lyrics/"):
                if os.path.isdir(nested_path + nested_dir):
                    DATASET[directory][nested_dir] = [
                        x.replace(".txt", "").replace(" ", "").lower()
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
        file = file.replace(".txt", "").replace(".abc", "").lower().replace(" ", "")
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
    print("Documents len", len(documents.values()))
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
    print(len(li))
    combinations = list(itertools.permutations(li, 2))
    # combinations = [(a, b) for idx, a in enumerate(li) for b in li[idx + 1 :]]
    print("Length of combinations: ", len(combinations))
    return combinations


def get_songs_from_artist(artist):
    for genre in DATASET:
        for a in DATASET[genre]:
            if a == artist:
                return DATASET[genre][a]
    return None


def get_genre_from_artist(artist):
    for genre in DATASET:
        for a in DATASET[genre]:
            if a == artist:
                return genre
    return None


def get_absolute_differences_mean(mat1, mat2):
    return abs(mat1 - mat2).mean()


def mann_whitney_rank(mat1, mat2):
    U1, p = mannwhitneyu(mat1.flatten(), mat2.flatten())
    return p


def main():
    populate_dataset_dict()

    for root, dirs, files in os.walk(ROOT):
        if LYRICS in root:
            populate_documents(root, files, LYRICS)
        elif ABC in root:
            populate_documents(root, files, ABC)

    # pprint(DATASET)

    df_lyrics, cosine_sim_lyrics, cosine_sim_lyrics_df = get_cosine_sim_mat(
        DOCUMENTS_LYRICS, stopwords=None
    )
    df_abc, cosine_sim_abc, cosine_sim_abc_df = get_cosine_sim_mat(
        DOCUMENTS_ABC, stopwords=None
    )

    # sanity_check(df_lyrics, cosine_sim_lyrics_df, "BookOfDays", "BlueberryHill")
    # sanity_check(df_abc, cosine_sim_abc_df, "BookOfDays", "WhatAWonderfulWorld")

    for genre in DATASET:
        for artist in DATASET[genre]:
            AGGREGATE_ARTIST[artist] = {}
            temp_lyrics_df = cosine_sim_lyrics_df.loc[DATASET[genre][artist]][
                DATASET[genre][artist]
            ]
            temp_lyrics_df_mat = temp_lyrics_df.to_numpy()
            # AGGREGATE_ARTIST[artist]["Lyrics"] = np.linalg.det(temp_lyrics_df_mat)
            # AGGREGATE_ARTIST[artist]["Lyrics"] = temp_lyrics_df_mat.mean()

            # Uncomment when ABC files are present
            temp_abc_df = cosine_sim_abc_df.loc[DATASET[genre][artist]][
                DATASET[genre][artist]
            ]
            temp_abc_df_mat = temp_abc_df.to_numpy()
            # AGGREGATE_ARTIST[artist]["ABC"] = np.linalg.det(temp_abc_df_mat)
            # AGGREGATE_ARTIST[artist]["ABC"] = temp_abc_df_mat.mean()

            AGGREGATE_ARTIST[artist] = round(
                1 - get_absolute_differences_mean(temp_lyrics_df_mat, temp_abc_df_mat),
                3,
            )

    pairs_artists = generate_pairs_of_artist()
    for pair in pairs_artists:
        AGGREGATE_PAIR_ARTISTS[pair] = {}
        songs1 = get_songs_from_artist(pair[0])
        songs2 = get_songs_from_artist(pair[1])
        temp_lyrics_df = cosine_sim_lyrics_df.loc[songs1][songs2]
        temp_lyrics_df_mat = temp_lyrics_df.to_numpy()
        # try:
        #     AGGREGATE_PAIR_ARTISTS[pair]["Lyrics"] = np.linalg.det(temp_lyrics_df_mat)
        # except:
        #     print(pair)

        # Uncomment when ABC files are present
        temp_abc_df = cosine_sim_abc_df.loc[songs1][songs2]
        temp_abc_df_mat = temp_abc_df.to_numpy()
        # AGGREGATE_PAIR_ARTISTS[pair]['ABC'] = np.linalg.det(temp_abc_df_mat)

        AGGREGATE_PAIR_ARTISTS[pair] = round(
            1 - get_absolute_differences_mean(temp_lyrics_df_mat, temp_abc_df_mat), 3
        )

    pprint(AGGREGATE_ARTIST)
    pprint(AGGREGATE_PAIR_ARTISTS)


if __name__ == "__main__":
    main()
