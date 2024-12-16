import pandas as pd

df = pd.read_csv('./automod-model/data_dumps/data.csv')

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('./automod-model/data_dumps/data1.csv', index=False, encoding='utf-8')