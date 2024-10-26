import csv
import random
from sklearn.model_selection import train_test_split
from math import floor

category_messages = {
    "OK": [],
    "Aggro": [],
    "Violence": [],
    'Sexual': [],
    "Hateful": []
}

columns = ["OK", "Aggro", "Violence", "Sexual", "Hateful"]

with open('./automod-model/train.csv', 'r', newline='', encoding='utf-8') as read_file:
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

with open('./automod-model/split-train.csv', 'a', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['Message', "OK", "Aggro", "Violence", "Sexual", "Hateful"])
    writer.writeheader()
    for label in columns:
        writer.writerows(category_messages[label][split_index[label]:len(category_messages[label])])

with open('./automod-model/split-test.csv', 'a', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['Message', "OK", "Aggro", "Violence", "Sexual", "Hateful"])
    writer.writeheader()
    for label in columns:
        writer.writerows(category_messages[label][:split_index[label]])