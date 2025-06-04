import regex as re
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optoolkit import DataRegexClassifier, RegexClassifier

if __name__ == "__main__":
    sexual_words = [
        "pizda", "gaoaza", "penis", "vagin", "futu(-| )?(te|ti|le)", "orgasm", "ejacul", "erecti", "clitoris", "dildo",
        "viol(ez|at)", "masturb", "muie", "puli", "pizdi", "muist", "blowjob", "dick", "pussy", "pzda", "porno",
        "futai", "futere", "futu(-| )?ti", "excita", "dezbrac", "handjob", "deepthroat", "dau rape", "boobjob",
        "labar", "labagiu", "child( )?porn", "cum( )?shot", "o sug(i|eti|e)?", "whore", "slut", "sperm"
    ]
    hateful_words = [
        "handicapat", "retard", "tigan", "bozgor", "n(i)+?(g)+(e)+?(r)+?", "negrotei", "poponar", "fa(g)+(ot)?", "nigro",
        "tigan", "ungur[a-z]+? imputit", "kike", "rusnac", "chink", "homalau", "d(i|y)ke",
        "autist", "pajeet", "exterminarea rasei [a-z]+", "rasa [a-z]+ trebuie sa moara", "gas chamber", "nigg",
        "trann(ie|y)", "esti masina de spalat", "(mars|du(-| )?te) (in|la|fa) (bucatarie|cratita|mancare|cartofi|senvis|sandwich|spalat)",
        "white power", "sieg heil", "gas jews", "gazez evrei"
    ]
    violent_words = [
        "(te|va) sparg", "sinucid", "(te|va|ma) omor", "injunghi", "arunca(-| )?te", "(te|va) bat",
        "dau bataie", "ia glont", "viol(ez|at)", "hang it", "kys", "kill yourself", "kys yourself", "hang yourself",
        "taie(-| )?(ma|te)", "spanzur", "impusca(-| )?(te|ma)", "ma impusc", "decapit", "strangu",
        "ineaca", "sugruma", "mutila", "iti rup (gatul|mainile)", "iti sparg fata", "te bag in pamant", "te dau cu capul", "iti scot matele",
        "ia(-| )?ti viata", "exterminarea rasei", "rasa [a-z]+ trebuie sa moara", "nu meriti sa (traiesti|existi|respiri)",
        "meriti sa mori", "doar mori", "omoara(-| )?te", "dau rape", "t(e|i)(-| )?as rupe", "sa te vad mort", "sa te calce (un|o )?(masina|tren|autobuz|tramvai)",
        "(mars|sa te vad|merg) in mormant"
    ]
    aggro_words = [
        "prostule", "proasto", "idio(tule|ato)", "retardat(ule|o)", "jego(sule|aso)", "gunoiule", "esti (un )?gunoi",
        "copile", "scarbo", "ratat(ule|o)", "esti (un |o )?ratat(a)?", "lingaule", "fraier(e|o|ilor)",
        "esti pro(st|asta)", "sunteti prosti", "sunteti ratati", "esti retard(at)?", "cine te(-| )?a intrebat",
        "du(-| )?te dracu", "esti un caine", "pisatule", "zdreanto", "esti (o )?zdreanta", "curvo", "esti curva",
        "est(i|e) e(-| )?girl", "est(i|e) e(-| )?grill", "(esti|iei) cuck", "e(sti)? (un )?incel", "esti (un )?simp",
        "esti (un )?soyboy", "nu meriti sa traiesti", "nu meriti sa existi", "o sug(i|eti|e)?", "(^|[^a-zA-Z])mars([^a-zA-Z]|$)",
        "ma-ta"
    ]

    sexual_classifier = RegexClassifier(triggerWords=sexual_words, max_typo=0)
    hateful_classifier = RegexClassifier(triggerWords=hateful_words)
    violent_classifier = RegexClassifier(triggerWords=violent_words)
    aggro_classifier = RegexClassifier(triggerWords=aggro_words)

    df = pd.read_csv('./automod-model/data_dumps/lolro_curated_unlabeled.csv')
    df = df[:1000]

    sexual_dfc = DataRegexClassifier(sexual_classifier, "Message", "Sexual")
    hateful_dfc = DataRegexClassifier(hateful_classifier, "Message", "Hateful")
    violent_dfc = DataRegexClassifier(violent_classifier, "Message", "Violence")
    aggro_dfc = DataRegexClassifier(aggro_classifier, "Message", "Aggro")
    
    df["OK"] = 0
    df = aggro_dfc.classify_data(df)
    df = violent_dfc.classify_data(df)
    df = sexual_dfc.classify_data(df)
    df = hateful_dfc.classify_data(df)
    
    df = df[df[["OK", "Aggro", "Violence", "Sexual", "Hateful"]].sum(axis=1) > 0]
    print(df.head())
