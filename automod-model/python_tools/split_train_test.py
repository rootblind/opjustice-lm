import csv
import random
from math import floor

category_messages = {
    "OK": [],
    "NOT OK": [],
}

columns = ["OK", "NOT OK"]

with open('./automod-model/side-model/data.csv', 'r', newline='', encoding='utf-8') as read_file:
    reader = csv.DictReader(read_file)
    for row in reader:
        for label in columns:
            if int(row[label]) == 1:
                category_messages[label].append(row)
                break

for labels in columns:
    random.shuffle(category_messages[labels])

split_index = {}

for label in category_messages:
    split_index[label] = floor(len(category_messages[label]) * 0.2)

with open('./automod-model/side-model/train.csv', 'a', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['Message', "OK", "NOT OK"])
    writer.writeheader()
    for label in columns:
        writer.writerows(category_messages[label][split_index[label]:len(category_messages[label])])

with open('./automod-model/side-model/test.csv', 'a', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['Message', "OK", "NOT OK"])
    writer.writeheader()
    for label in columns:
        writer.writerows(category_messages[label][:split_index[label]])