# this source file is used just to run scripts, like a work sheet
import sys, os
import regex as re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optoolkit import DataToolkit, Model, DatasetLoader, RegexClassifier
import pandas as pd
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer

sexual_words = [
        "pizda", "gaoaza", "penis", "vagin", "futu(-| )?(te|ti|le)", "orgasm", "ejacul", "erecti", "clitoris", "dildo",
        "viol(ez|at)", "masturb", "muie", "puli", "pizdi", "muist", "blowjob", "dick", "pussy", "pzda", "porno",
        "futai", "futere", "futu(-| )?ti", "excita", "dezbrac", "handjob", "deepthroat", "dau rape", "boobjob",
        "labar", "labagiu", "child( )?porn", "cum( )?shot", "o sug(i|eti|e)?", "sperm"
    ]
hateful_words = [
        "handicapat", "retard", "tigan", "bozgor", "n(i)+?(g)+(e)+?(r)+?", "negrotei", "poponar", "fa(g)+(ot)?", "nigro",
        "ungur[a-z]+? imputit", "kike", "rusnac", "chink", "homalau", "d(i|y)ke",
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

def regex_labeling(text):
    labels = []
    if aggro_classifier.regex_classifier(text):
        labels.append("Aggro")

    if violent_classifier.regex_classifier(text):
        labels.append("Violence")

    if sexual_classifier.regex_classifier(text):
        labels.append("Sexual")

    if hateful_classifier.regex_classifier(text):
        labels.append("Hateful")

    return labels

dataset = DatasetLoader(["Message"], "Message")

tokenizer = AutoTokenizer.from_pretrained("./automod-model/model_versions/v3")

model = Model(model_name='./automod-model/model_versions/v3', 
                                   num_labels=len(dataset.labels), 
                                   id2label=dataset.id2label, 
                                   label2id=dataset.label2id)

df = pd.read_csv("./automod-model/data_dumps/lolro_curated_unlabeled.csv")

df = df[7_000:10_000]

labeled = {"Message": []}

for l in dataset.labels:
    labeled[l] = []

for _, row in df.iterrows():
    if len(row["Message"]) > 511:
        continue
    model_labels = model.label_text(row["Message"], tokenizer, dataset.labels, 0.75)
    regex_labels = regex_labeling(row["Message"])

    scores = {
        "OK": 0,
        "Aggro": 0,
        "Violence": 0,
        "Sexual": 0,
        "Hateful": 0
    }

    for label in regex_labels:
        scores[label] = 1

    if len(regex_labels) == 0 and (len(model_labels) == 0 or "OK" in model_labels):
        scores["OK"] = 1
    else:
        for label in model_labels:
            if label != "OK":
                scores[label] = 1

    for score in scores:
        labeled[score].append(scores[score])

    labeled["Message"].append(row["Message"])

df = pd.DataFrame(data=labeled)

df.to_csv("./automod-model/data_dumps/autolabeled_lolro.csv", index=False, encoding='utf-8')
