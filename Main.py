import pandas as pd
import random
import pprint
# from nltk.corpus import stopwords
from sklearn.metrics import classification_report, accuracy_score
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
    print("Train Rows:")
    print(train_rows)
    print("Test Rows:")
    print(test_rows)

    print("Encoding Labels...")
    y_train, y_test, class_names = tweet_ml.encode_class_labels(train_rows, test_rows, df)
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
