import csv
import random

classes = ['Message','OK','Aggro','Violence','Sexual','Hateful']
data = []

with open('./automod-model/train.csv', 'r', newline='', encoding='utf-8') as read_file:
    reader = csv.DictReader(read_file)
    for row in reader:
        new_row = {
            'Message': row['Message'],
            'OK': row['OK'],
            'Aggro': row['Aggro'],
            'Violence': row['Violence'],
            'Sexual': row['Sexual'],
            'Hateful': row['Hateful']
        }
        if row['Insult'] == 1:
            new_row['Aggro'] = 1
        data.append(new_row)

random.shuffle(data)

with open('./automod-model/converted-train.csv', 'a', newline='', encoding='utf-8') as write_file:
    writer = csv.DictWriter(write_file, fieldnames=classes)
    writer.writeheader()
    writer.writerows(data)

print('Executed')