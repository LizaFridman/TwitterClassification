from nltk.corpus import stopwords
from TwitterDataCollector import TwitterDataCollector

print("Hello World! Here are my stopwords:\n")
print(stopwords.words('english'))
dr = TwitterDataCollector("Name", "Path")
dr.who_am_I()