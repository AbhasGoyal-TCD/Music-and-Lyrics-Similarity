from sqlite3 import DatabaseError
from webbrowser import get
from cv2 import rotate
import numpy as np
import nltk
import string
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import itertools
from scipy.stats import mannwhitneyu
import seaborn as sns

# import matplotlib.pyplot as plt

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


def get_all_artists():
    artists = []
    for genre in DATASET:
        artists.extend(DATASET[genre].keys())
    return artists


def generate_pairs_of_artist():
    artists = get_all_artists()
    combinations = list(itertools.permutations(artists, 2))
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


def get_colors_from_artist(artists):
    genres = DATASET.keys()
    colors = [
        "red",
        "blue",
        "green",
        "black",
        "cyan",
        "yellow",
        "orange",
        "brown",
        "violet",
        "purple",
    ]

    op = []
    for g, c in zip(genres, colors):
        DATASET[g]["color"] = c
    for a in artists:
        genre = get_genre_from_artist(a)
        op.append(DATASET[genre]["color"])

    return op


def main():
    populate_dataset_dict()

    for root, dirs, files in os.walk(ROOT):
        if LYRICS in root:
            populate_documents(root, files, LYRICS)
        elif ABC in root:
            populate_documents(root, files, ABC)

    df_lyrics, cosine_sim_lyrics, cosine_sim_lyrics_df = get_cosine_sim_mat(
        DOCUMENTS_LYRICS, stopwords=None
    )
    df_abc, cosine_sim_abc, cosine_sim_abc_df = get_cosine_sim_mat(
        DOCUMENTS_ABC, stopwords=None
    )

    for genre in DATASET:
        for artist in DATASET[genre]:
            AGGREGATE_ARTIST[artist] = {}
            temp_lyrics_df = cosine_sim_lyrics_df.loc[DATASET[genre][artist]][
                DATASET[genre][artist]
            ]
            temp_lyrics_df_mat = temp_lyrics_df.to_numpy()

            temp_abc_df = cosine_sim_abc_df.loc[DATASET[genre][artist]][
                DATASET[genre][artist]
            ]
            temp_abc_df_mat = temp_abc_df.to_numpy()

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

        temp_abc_df = cosine_sim_abc_df.loc[songs1][songs2]
        temp_abc_df_mat = temp_abc_df.to_numpy()

        AGGREGATE_PAIR_ARTISTS[pair] = round(
            1 - get_absolute_differences_mean(temp_lyrics_df_mat, temp_abc_df_mat), 3
        )

    # pprint(AGGREGATE_ARTIST)
    # pprint(AGGREGATE_PAIR_ARTISTS)

    artists = get_all_artists()
    pair_artists = [[None for a in artists] for a in artists]

    for i, a1 in enumerate(artists):
        for j, a2 in enumerate(artists):
            if a1 == a2:
                pair_artists[i][j] = AGGREGATE_ARTIST[a1]
            else:
                pair_artists[i][j] = AGGREGATE_PAIR_ARTISTS[(a1, a2)]

    aggregate_pair_artist_df = pd.DataFrame(
        pair_artists, columns=artists, index=artists
    )

    # pprint(aggregate_pair_artist_df)
    plt.rcParams["figure.figsize"] = (14, 10)
    plt.rcParams["xtick.labelsize"] = 7

    # heatmap = sns.heatmap(aggregate_pair_artist_df, cmap="YlGnBu")
    # fig = heatmap.get_figure()
    # fig.savefig("heatmap.png")

    li = []
    for i in range(len(artists)):
        li.append(pair_artists[i][i])
    colors = get_colors_from_artist(artists)
    plt.bar(artists, li, color=colors)
    plt.xticks(rotation=90)
    plt.tight_layout()
    genre_colors = [DATASET[g]["color"] for g in DATASET]
    genres = list(DATASET.keys())
    # legend_c = [plt.Rectangle((0, 0), 1, 1, color=DATASET[g]["color"]) for g in DATASET]
    #  = [DATASET[g]["color"] for g in DATASET]
    plt.legend(
        handles=[
            mpatches.Patch(color=genre_colors[i], label=genres[i])
            for i in range(len(genres))
        ]
    )
    plt.savefig("similarity_within_artist.png")

    # print(len(DATASET.keys()))
    # print(artists)
    # print(get_colors_from_artist(artists))


if __name__ == "__main__":
    main()
