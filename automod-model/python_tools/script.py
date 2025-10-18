import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from optoolkit import DataToolkit

dtk = DataToolkit()
df = pd.read_csv("automod-model/data_dumps/fw_flag_data.csv", encoding="utf-8")
train, test = dtk.split_train_test(df, df.columns[1:])

train.to_csv("automod-model/data_compile/train.csv", index=False, encoding="utf-8")
test.to_csv("automod-model/data_compile/test.csv", index=False, encoding="utf-8")