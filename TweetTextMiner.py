import csv

csv_path = "C:\\GitHub\\TwitterClassification\\"
csv_name = 'gender-classifier-DFE-791531.csv'

with open(csv_path+csv_name, 'r', encoding="utf8", errors='ignore') as csv_file:
    reader = csv.reader(csv_file,delimiter=",")
    headers=next(reader)
    text_idx=headers.index('text')
    gender_idx=headers.index('gender')
    tweet=next(reader)

    print(headers)
    print(tweet)