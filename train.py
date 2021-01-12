# Python libraries
import datetime as dt
import re
import pickle
from tqdm import tqdm
import os
import sys
import time
import logging
import random
import json
from collections import Counter
import argparse

# Data Science modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from market_sentiment.nlp_utils import *
from market_sentiment.data_utils import *
from market_sentiment.visualize import visualize_frequent_words, generate_word_cloud

sns.set()
plt.style.use("ggplot")

# Import Scikit-learn moduels
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser(description="Sentiment Analysis Trainer")
parser.add_argument("--data-file",
                    help="text file for dataset",
                    default="data/FinancialPhraseBank-v1.0/Sentences_75Agree.txt")
parser.add_argument("--sentiment-words-file",
                    help="csv with sentiment word list",
                    default="resources/LoughranMcDonald_SentimentWordLists_2018.csv")
parser.add_argument("--output-dir",
                    metavar="DIR",
                    default="./output",
                    help="output directory for model")
parser.add_argument("--seed",
                    type=int,
                    default=42,
                    help="random seed value")
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")


# Set Seaborn Style
sns.set(style="white", palette="deep")


def train_log_reg(filename, sentiment_words_file, seed=42):
    train_df = load_finphrase(filename)

    # Samples
    pd.set_option("display.max_colwidth", -1)
    logging.debug(train_df.sample(n=20, random_state=seed))


    # Encode the label
    le = LabelEncoder()
    le.fit(train_df["label"])
    train_df["label"] = le.transform(train_df["label"])
    logging.debug(list(le.classes_))
    logging.debug(train_df["label"])

    corpus = create_corpus(train_df)
    # visualize_frequent_words(corpus, stop_words)
    # generate_word_cloud(corpus, stop_words)

    # Load sentiment data
    sentiment_df = pd.read_csv(sentiment_words_file)

    # Make all words lower case
    sentiment_df["word"] = sentiment_df["word"].str.lower()
    sentiments = sentiment_df["sentiment"].unique()
    sentiment_df.groupby(by=["sentiment"]).count()

    sentiment_dict = {
        sentiment: sentiment_df.loc[sentiment_df["sentiment"] == sentiment][
            "word"
        ].values.tolist()
        for sentiment in sentiments
    }


    columns = [
        "tone_score",
        "word_count",
        "n_pos_words",
        "n_neg_words",
        "pos_words",
        "neg_words",
    ]

    # Analyze tone for original text dataframe
    print(train_df.shape)
    tone_lmdict = [
        tone_count_with_negation_check(sentiment_dict, x.lower())
        for x in tqdm(train_df["sentence"], total=train_df.shape[0])
    ]
    tone_lmdict_df = pd.DataFrame(tone_lmdict, columns=columns)
    train_tone_df = pd.concat([train_df, tone_lmdict_df.reindex(train_df.index)], axis=1)
    train_tone_df.head()

    # Show corelations to next_decision
    plt.figure(figsize=(10, 6))
    corr_columns = ["label", "n_pos_words", "n_neg_words"]
    sns.heatmap(
        train_tone_df[corr_columns].astype(float).corr(),
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
    )
    # plt.show()

    # X and Y data used
    Y_data = train_tone_df["label"]
    X_data = train_tone_df[["tone_score", "n_pos_words", "n_neg_words"]]

    # Train test split (Shuffle=False will make the test data for the most recent ones)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X_data.values, Y_data.values, test_size=0.2, shuffle=True
    )

    # Tokenize
    tokenized, tokenized_text, bow, vocab, id2vocab, token_ids = tokenize_df(
        train_tone_df, col="sentence", lemma=True, stopwords=True, tokenizer="NLTK"
    )
    sns.distplot([len(x) for x in tokenized_text])

    # X and Y data used
    Y_data = train_tone_df["label"]
    X_data = tokenized_text

    # Train test split (Shuffle=False will make the test data for the most recent ones)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X_data, Y_data.values, test_size=0.2, shuffle=True
    )


    pipeline = Pipeline(
        [("vec", TfidfVectorizer(analyzer="word")), ("clf", LogisticRegression())]
    )

    pipeline.fit(X_train, Y_train)

    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    # Define metrics
    # Here, use F1 Macro to evaluate the model.
    def metric(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        return acc, f1


    acc, f1 = metric(Y_train, pred_train)
    logging.info("Training - acc: %.8f, f1: %.8f" % (acc, f1))
    acc, f1 = metric(Y_test, pred_test)
    logging.info("Test - acc: %.8f, f1: %.8f" % (acc, f1))
    return pipeline


def main():
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    os.makedirs(args.output_dir, exist_ok=True)

    # Set Random Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    filename = args.data_file
    sentiment_words_file = args.sentiment_words_file
    model_pipeline = train_log_reg(filename, sentiment_words_file, args.seed)

    model_name = 'log_reg_model.pkl'
    with open(os.path.join(args.output_dir, model_name), 'wb') as f:
        pickle.dump(model_pipeline, f)


if __name__ == "__main__":
    main()
