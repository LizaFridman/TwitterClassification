import pandas as pd

import random
import pprint
# from nltk.corpus import stopwords
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

import TweetMachineLearning as tweet_ml

csv_path = "C:\\GitHub\\TwitterClassification\\"
csv_name = 'gender-classifier-DFE-791531.csv'


def print_terms(count_male, count_female, count_brand):
    pp = pprint.PrettyPrinter()
    print("Term Frequency per Gender as taken from Twitter Dataset:")
    print("Male most common:", sum(count_male.values()))
    pp.pprint(count_male.most_common(20))
    print("")

    print("Female most common:", sum(count_female.values()))
    pp.pprint(count_female.most_common(20))
    print("")

    print("Brand most common:", sum(count_brand.values()))
    pp.pprint(count_brand.most_common(20))


def print_results(y_true, y, data_set_name, class_names):
    print(data_set_name)
    print(classification_report(y, y_true, target_names=class_names))
    print("Accuracy: {}".format(accuracy_score(y, y_true)))
    print("==================================================================")
    print()


def report_results(grid_search, y_train, X_train, y_test, X_test, class_names):
    print("Best params: ", grid_search.best_params_)
    print_results(grid_search.predict(X_train), y_train, "Train", class_names)
    print_results(grid_search.predict(X_test), y_test, "Test", class_names)


def main():
    # tweet_texts, terms_male, terms_female, terms_brand = tweet_ml.process_twitter_dataset(csv_path=csv_path, csv_name=csv_name)
    # count_male, count_female, count_brand = tweet_ml.gender_term_freq(terms_male, terms_female, terms_brand)
    # print_terms(count_male, count_female, count_brand)

    df = pd.read_csv(csv_path + csv_name, encoding='latin1')
    chosen_rows = df[df["gender"].isin(["male", "female", "brand"]) & (df["gender:confidence"] > 0.99)].index.tolist()
    n_samples = len(chosen_rows)
    random.shuffle(chosen_rows)
    test_size = round(n_samples*0.2)
    test_rows = chosen_rows[:test_size]

    #val_rows = chosen_rows[test_size:2*test_size]
    train_rows = chosen_rows[2*test_size:]
    #print("Train Rows:")
    #print(train_rows)
    #print("Test Rows:")
    #print(test_rows)

    #print("Encoding Labels...")
    y_train, y_test, class_names = tweet_ml.encode_class_labels(train_rows, test_rows, df)

    JOBS = 2
    PARAMS = [{'max_depth': [150, 160, 170, 180, 190, 200, 220],
               'n_estimators': [120, 140, 160, 180, 200, 220, 240, 260],
               'average': ['micro']}]

    print("TIDF")

    X_train, X_test = tweet_ml.extract_tfidf_from_text_and_desc(df, train_rows, test_rows)

    tweet_feats_train, tweet_feats_test = tweet_ml.extract_tweet_count_feats(df, train_rows, test_rows)

    # Merge tweets feats and TF-IDF.
    X_train = hstack((X_train, tweet_feats_train))
    X_test = hstack((X_test, tweet_feats_test))

    grid_search = GridSearchCV(RandomForestClassifier(), PARAMS, n_jobs=JOBS, verbose=5, cv=4,
                               scoring="f1")
    grid_search.fit(X_train, y_train)
    report_results(grid_search, y_train, X_train, y_test, X_test, class_names)

    print("Count Vectorizer")

    X_train, X_test = tweet_ml.extract_feats_from_text_and_desc(df, train_rows, test_rows)

    # Merge tweets feats and TF-IDF.
    X_train = hstack((X_train, tweet_feats_train))
    X_test = hstack((X_test, tweet_feats_test))

    grid_search = GridSearchCV(RandomForestClassifier(), PARAMS, n_jobs=JOBS, verbose=5, cv=4,
                               scoring="f1")
    grid_search.fit(X_train, y_train)
    report_results(grid_search, y_train, X_train, y_test, X_test, class_names)

    print("Extracting Features...")
    X_train, X_test = tweet_ml.extract_feats_from_text(df, train_rows, test_rows)

    print("Multinomial Naive Bayes calculation...")
    nb = MultinomialNB()
    nb = nb.fit(X_train, y_train)
    print(classification_report(y_test, nb.predict(X_test), target_names=class_names))
    accuracy_score(y_test, nb.predict(X_test))

    # X_train, X_test = tweet_ml.extract_feats_from_text_and_desc(df, train_rows, test_rows)
    # X_train, X_test = tweet_ml.extract_tfidf_from_text_and_desc(df, train_rows, test_rows)


if __name__ == "__main__":
    main()
