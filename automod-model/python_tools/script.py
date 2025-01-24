# this source file is used just to run scripts, like a work sheet
import sys, os
import regex as re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optoolkit import DataToolkit
import pandas as pd

df = pd.read_csv('./automod-model/data_dumps/flag_data.csv')

datatk = DataToolkit()
patterns = [
            re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            re.compile(r'<:(\d+):>'),
            re.compile(r'[^a-zA-Z -]'),
        ]
df["Message"] = df["Message"].apply(lambda text: datatk.filter_text(text, patterns))

df.to_csv('./automod-model/data_compile/flag.csv', index=False, encoding='utf-8')