from datasets import load_dataset
import csv
import pandas as pd
import random

train = pd.read_csv('./automod-model/train.csv')
train = train.sample(frac=1).reset_index(drop=True)
ok_count = 0

rows_to_write = []
converted_file = './automod-model/side-model/train.csv'

for index, row in train.iterrows():
    if row['OK'] == 1 and ok_count <= 1000:
        ok_count = ok_count + 1
        rows_to_write.append(
            {'Message': row['Message'], 'OK': 1, 'NOT OK': 0}
        )
    elif row['OK'] == 0:
        rows_to_write.append(
            {'Message': row['Message'], 'OK':0, 'NOT OK': 1}
        )

random.shuffle(rows_to_write)

with open(converted_file, 'a', newline='', encoding='utf-8') as dataset:
    writer = csv.DictWriter(dataset, fieldnames=['Message', 'OK', 'NOT OK'])
    writer.writeheader()
    writer.writerows(rows_to_write)
    
print('Done!')
    
    