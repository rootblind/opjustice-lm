import pandas as pd
import regex as re
import numpy as np

def label_keyword(df, keyword, label):
    pattern = rf'{re.escape(keyword)}\w*'
    df.loc[df["Message"].str.lower().str.contains(pattern), label] = 1
    return df

def label_keywords(df, keyword_disct):
    for label in keyword_disct:
        for keyword in keyword_disct[label]:
            label_keyword(df, keyword, label)
    return df


if __name__ == '__main__':
    df = pd.read_csv('./automod-model/data_dumps/roof_to_op.csv')
    
    keyword_dict = {
        "Violence": ["spanzur", "impusc"],
        "Sexual": ["pul", "pzd", "pizda", "tzatze", "tate", "fut", "sugi"],
        "Hateful": ["tigan", "bozgor", "poponar", "handicap", "retard"]
    }
    df = label_keywords(df, keyword_dict)
    

    print("Executed")
    df.to_csv('./automod-model/data_dumps/roof_to_op-1.csv', index=False, encoding='utf-8')