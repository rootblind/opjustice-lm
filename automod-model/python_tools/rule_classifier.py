import regex as re
import pandas as pd

from optoolkit import DataRegexClassifier, RegexClassifier

if __name__ == "__main__":
    sexual_words = ["pula", "pizda", "tate", "anus", "gaoaza", "penis", "vagin", "sex", "sexy", "rape", "viol", "fut", "futute",
                    "violez", "masturbez", "laba", "muie", "puli", "pizdi", "muist", "blowjob", "fuck", "dick", "pussy", "pzda", "porno"
                    ]
    hateful_words = [
        "handicapat", "retardat", "tigan", "bozgor", "nigger", "negrotei", "poponar", "faggot"
    ]
    violent_words = [
        "te sparg", "sinucid", "omor", "injunghi", "beregata", "pistol", "funia", "executat",
        "omoara", "arunca", "impusc"
    ]
    sexual_classifier = RegexClassifier(triggerWords=sexual_words, max_typo=0)
    hateful_classifier = RegexClassifier(triggerWords=hateful_words)
    violent_classifier = RegexClassifier(triggerWords=violent_words)

    df = pd.read_csv('./automod-model/data_dumps/lolro_curated_unlabeled.csv')
    df = df[:1000]

    sexual_dfc = DataRegexClassifier(sexual_classifier, "Message", "Sexual")
    df = sexual_dfc.classify_data(df)
    df["Violence"] = df.apply(
        lambda row: 1 if violent_classifier.regex_classifier(row["Message"]) and (row["Violence"] != 1) else row["Violence"],
        axis=1
    )
    df["Hateful"] = df.apply(
        lambda row: 1 if hateful_classifier.regex_classifier(row["Message"]) and (row["Hateful"] != 1) else row["Hateful"],
        axis=1
    )
    
    df = df[df[["OK", "Aggro", "Violence", "Sexual", "Hateful"]].sum(axis=1) > 0]
    print(df.head())
