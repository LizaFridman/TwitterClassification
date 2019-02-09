import csv
import random
import math
import operator
import pprint
import re
import string

from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder


# KNN ----------------------------------------------------------
def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


# -------------------------------------------------------------

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    # r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    # r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']


def tokenize(s):
    s = re.sub(r'[^\x00-\x7f]*', r'', s)
    return tokens_re.findall(s)


def preprocess(s, lowercase=True):
    token_list = tokenize(s)
    if lowercase:
        token_list = [t if emoticon_re.search(t) else t.lower() for t in token_list]
    return token_list


def process_twitter_dataset(csv_path, csv_name):
    with open(csv_path + csv_name, 'r', encoding="utf8", errors='ignore') as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        headers = next(reader)
        text_idx = headers.index('text')
        gender_idx = headers.index('gender')

        tweet_texts = []
        terms_male = []
        terms_female = []
        terms_brand = []

        for line in reader:
            tweet_texts.append(line[text_idx])
            tokens = preprocess(line[text_idx], lowercase=True)
            for token in tokens:
                # token = token.lower()
                if (token not in stop) and (not token.isnumeric()):
                    # if  (token.startswith('#') and token not in stop):
                    if line[gender_idx] == 'male':
                        terms_male.append(token)
                    else:
                        if line[gender_idx] == 'female':
                            terms_female.append(token)
                        else:
                            if line[gender_idx] == 'brand':
                                terms_brand.append(token)

        return tweet_texts, terms_male, terms_female, terms_brand


def gender_term_freq(terms_male, terms_female, terms_brand):
    count_male = Counter()
    count_male.update(terms_male)

    count_female = Counter()
    count_female.update(terms_female)

    count_brand = Counter()
    count_brand.update(terms_brand)

    return count_male, count_female, count_brand


# A function for computing lexical diversity
def lexical_diversity(tokens):
    return len(set(tokens))/len(tokens)


# A function for computing the average number of words per tweet
def average_words(statuses):
    total_words = sum([len(s.split()) for s in statuses])
    return total_words/len(statuses)

#-------------------------------------------------------------


def encode_class_labels(train_rows, test_rows, df):
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(df.ix[train_rows, "gender"])
    y_test = encoder.transform(df.ix[test_rows, "gender"])

    return y_train, y_test, encoder.classes_


def compute_text_feats(vectorizer, rows, df):
    return vectorizer.transform(df.ix[rows, "text_norm"])


def compute_text_desc_feats(vectorizer, rows, df):
    train_text = df.ix[rows, :]["text_norm"]
    train_desc = df.ix[rows, :]["description_norm"]

    return vectorizer.transform(train_text.str.cat(train_desc, sep=' '))


def normalize_text(text):
    # Remove non-ASCII chars.
    text = re.sub('[^\x00-\x7F]+', ' ', text)

    # Remove URLs
    text = re.sub('https?:\/\/.*[\r\n]*', ' ', text)

    # Remove special chars.
    text = re.sub('[?!+%{}:;.,"\'()\[\]_]', '', text)

    # Remove double spaces.
    text = re.sub('\s+', ' ', text)

    return text


def extract_feats_from_text(df, train_rows, test_rows):
    df["text_norm"] = [normalize_text(text) for text in df["text"]]
    df["description_norm"] = [normalize_text(text) for text in df["description"].fillna("")]

    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(df.ix[train_rows, :]["text_norm"])

    X_train = compute_text_feats(vectorizer, train_rows, df)
    X_test = compute_text_feats(vectorizer, test_rows, df)

    return X_train, X_test


def extract_feats_from_text_and_desc(df, train_rows, test_rows):
    df["text_norm"] = [normalize_text(text) for text in df["text"]]
    df["description_norm"] = [normalize_text(text) for text in df["description"].fillna("")]

    vectorizer = CountVectorizer()
    train_text = df.ix[train_rows, :]["text_norm"]
    train_desc = df.ix[train_rows, :]["description_norm"]
    vectorizer = vectorizer.fit(train_text.str.cat(train_desc, sep=' '))

    X_train = compute_text_desc_feats(vectorizer, train_rows, df)
    X_test = compute_text_desc_feats(vectorizer, test_rows, df)

    return X_train, X_test


def extract_tweet_count_feats(df, train_rows, test_rows):
    feats = df[["retweet_count", "tweet_count", "fav_number"]]

    train_feats = feats.ix[train_rows, :]
    test_feats = feats.ix[test_rows, :]

    scaler = StandardScaler().fit(train_feats)

    return scaler.transform(train_feats), scaler.transform(test_feats)


def extract_tfidf_from_text_and_desc(df, train_rows, test_rows):
    tfidf = TfidfVectorizer(strip_accents="unicode")
    df["text_norm"] = [normalize_text(text) for text in df["text"]]
    df["description_norm"] = [normalize_text(text) for text in df["description"].fillna("")]

    train_text = df.ix[train_rows, :]["text_norm"]
    train_desc = df.ix[train_rows, :]["description_norm"]
    tfidf = tfidf.fit(train_text.str.cat(train_desc, sep=' '))

    X_train = compute_text_desc_feats(tfidf, train_rows, df)
    X_test = compute_text_desc_feats(tfidf, test_rows, df)

    return X_train, X_test
