# this source file is used just to run scripts, like a work sheet
import sys, os
import regex as re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optoolkit import DataToolkit
import pandas as pd
from deep_translator import GoogleTranslator

dtk = DataToolkit()

dfs = [
    pd.read_csv("./automod-model/data_compile/train.csv"),
    pd.read_csv("./automod-model/data_compile/test.csv"),
]

df_test = pd.concat(dfs, ignore_index=True)

df_test = dtk.remove_duplicates(df_test, df_test.columns)
df_test.to_csv("./automod-model/data_compile/data.csv", index=False, encoding='utf-8')