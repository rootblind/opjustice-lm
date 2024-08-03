import csv
import os

# the dataset was scored based on a float system
# this script converts that into a boolean system

classes = ['Message','OK','Insult','Violence','Sexual','Hateful','Flirt','Spam','Aggro']
data = []


with open('./automod-model/test_sample.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for c in classes[1:]:
            row[c] = float(row[c])
        data.append(row)

for d in data:
    for c in classes[2:]:
        if d['OK'] >= 0.7:
            d[c] = 0
        if d[c] > d['OK']:
            d['OK'] = 0
        if d[c] >= 0.6:
            d[c] = 1
            d['OK'] = 0
        if d['OK'] >= 0.5:
            d[c] = 0
            d['OK'] = 1
        d[c] = int(d[c])
    d['OK'] = int(d['OK'])


def write():
    if not os.path.exists('./automod-model/test.csv'):
        with open('./automod-model/test.csv', 'w') as f:
            pass
    with open('./automod-model/test.csv','w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=classes)
        writer.writeheader()
        for d in data:
            writer.writerow(d)
write()

print('Executed')
