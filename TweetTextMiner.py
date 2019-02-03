import csv
import pprint
import re
import string

from collections import Counter
from nltk.corpus import stopwords

csv_path = "C:\\GitHub\\TwitterClassification\\"
csv_name = 'gender-classifier-DFE-791531.csv'

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
    # r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    # r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

punctuation = list(string.punctuation)
stop = stopwords.words('english')  + punctuation + ['rt', 'via']


def tokenize(s):
    s = re.sub(r'[^\x00-\x7f]*', r'', s)
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    token_list = tokenize(s)
    if lowercase:
        token_list = [t if emoticon_re.search(t) else t.lower() for t in token_list]
    return token_list


def term_freq():
    with open(csv_path + csv_name, 'r', encoding="utf8", errors='ignore') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        headers = next(reader)
        text_idx = headers.index('text')
        gender_idx = headers.index('gender')
        terms_male = []
        terms_female = []
        terms_brand = []

        for line in reader:
            tokens = preprocess(line[text_idx])
            for token in tokens:
                if token not in stop:
                    # if  (token.startswith('#') and token not in stop):
                    if line[gender_idx] == 'male':
                        terms_male.append(token)
                    else:
                        if line[gender_idx] == 'female':
                            terms_female.append(token)
                        else:
                            if line[gender_idx] == 'brand':
                                terms_brand.append(token)

        # Results...
        count_male = Counter()
        count_male.update(terms_male)
        pp = pprint.PrettyPrinter()

        print("Male most common:", sum(count_male.values()))
        pp.pprint(count_male.most_common(20))
        print("")

        count_female = Counter()
        count_female.update(terms_female)
        print("Female most common:", sum(count_female.values()))
        pp.pprint(count_female.most_common(20))
        print("")

        count_brand = Counter()
        count_brand.update(terms_brand)
        print("Brand most common:", sum(count_brand.values()))
        pp.pprint(count_brand.most_common(20))
