from deep_translator import GoogleTranslator
import os
import torch
import csv
import numpy as np
import regex as re
import pandas as pd
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

class ToxicityDataset:
    def __init__(self, dataset_path='./automod-model/dataset', dataset_name='rootblind/opjustice-dataset'):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.dataset = self.load_dataset()
        self.labels = self.get_labels()
        self.id2label, self.label2id = self.create_label_mappings()

    def load_dataset(self):
        if os.path.exists(self.dataset_path):
            dataset = load_from_disk(self.dataset_path)
        else:
            os.mkdir(self.dataset_path)
            dataset = load_dataset(self.dataset_name)
            dataset.save_to_disk(self.dataset_path)
        return dataset

    def get_labels(self):
        return [label for label in self.dataset['train'].features.keys() if label not in 'Message']

    def create_label_mappings(self):
        id2label = {idx: label for idx, label in enumerate(self.labels)}
        label2id = {label: idx for idx, label in enumerate(self.labels)}
        return id2label, label2id

    def preprocess_data(self, examples, tokenizer):
        text = examples["Message"]
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=400)
        labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels}
        labels_matrix = np.zeros((len(text), len(self.labels)))

        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()
        return encoding

    def encode_dataset(self, tokenizer):
        encoded_dataset = self.dataset.map(lambda x: self.preprocess_data(x, tokenizer), batched=True, remove_columns=self.dataset['train'].column_names)
        encoded_dataset.set_format("torch")
        return encoded_dataset


class ToxicityModel:
    def __init__(self, model_name, num_labels, id2label, label2id, device=None):
        self.model_name = model_name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(num_labels, id2label, label2id)

    def load_model(self, num_labels, id2label, label2id):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, 
                                                                   problem_type="multi_label_classification", 
                                                                   num_labels=num_labels,
                                                                   id2label=id2label,
                                                                   label2id=label2id).to(self.device)
        return model

    def predict(self, text, tokenizer):
        encoding = tokenizer(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
        return outputs.logits
    
    def label_text(self, text, tokenizer):
        logits = self.predict(text, tokenizer)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu()) 
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        predictions[np.argmax(probs)] = 1
        predicted_labels = [toxicity_dataset.id2label[idx] for idx, label in enumerate(predictions) if label]
        if 'OK' in predicted_labels:
            predicted_labels = ['OK']

        return predicted_labels

index = 1
class DataTranslator:
    def __init__(self, source, dest):
        self.source = source
        self.dest = dest
        self.translator = GoogleTranslator(source=self.source, target=self.dest)

    def translate_text(self, text):
        global index
        print(f'Translated {text[:20]}...[{index}]')
        index = index + 1
        translator = self.translator
        return translator.translate(text)
    
    def translate_data(self, data, text_header):
        # takes ~5m per 1000 messages or 3-4 messages per second
        data[text_header] = data[text_header].apply(lambda text: self.translate_text(text))
        return data

    def remove_duplicates(self, data, header):
        return data.drop_duplicates(subset=header, inplace=True)
    
    def remove_labels(self, data, header, labels):
        new_dt = {}

        for column in header:
            new_dt[column] = []

        for _, row in data.iterrows():
            if row[header[1]] == labels:
                continue
            for column in header:
                new_dt[column].append(row[column])
        return pd.DataFrame(data=new_dt)
    
    def focus_labels(self, data, header, labels):
        new_dt = {}

        for column in header:
            new_dt[column] = []

        for _, row in data.iterrows():
            if row[header[1]] in labels:
                for column in header:
                    new_dt[column].append(row[column])
        return pd.DataFrame(data=new_dt)

    def filter_data(self, data, header):
        alphabet_pattern = re.compile(r'^[a-zA-Z]')
        allowed_pattern = re.compile(r'[^a-zA-Z0-9 ]')
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

        filtered_data = {}

        for column in header:
            filtered_data[column] = []

        def filter(message):
            message = message.lower()
            if len(message) < 3:
                return message
            message = message.replace('\n', ' ').replace('\r', ' ')
            message = message.replace('ă', 'a')
            message = message.replace('î', 'i')
            message = message.replace('ș', 's')
            message = message.replace('ț', 't')
            message = message.replace('â', 'a')
            message = re.sub(url_pattern, '', message)
            message = re.sub(allowed_pattern, '', message)
            message = message.lstrip()
            return message
        
        for _, row in data.iterrows():
            text = filter(row[header[0]])
            if ' ' not in text or len(text) > 512:
                continue

            if(len(text) > 2 and alphabet_pattern.search(text)):
                row[header[0]] = text
                for column in header:
                    filtered_data[column].append(row[column])
        return pd.DataFrame(data=filtered_data)
    
class ConvertDataset:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def equivalent_labeling(self, df, header):
        model = self.model
        tokenizer = self.tokenizer
        new_columns = ["OK", "Aggro", "Violence", "Sexual", "Hateful"]
        new_df = {}
        
        new_df["Message"] = []
        for column in new_columns:
            new_df[column] = []
        
        convert_labels = {
            "threat" : "Violence",
            "insult": "Aggro",
            "identity_hate": "Hateful",
        }
        for _, row in df.iterrows():
            score = {}
            for c in new_columns:
                score[c] = 0

            for h in header[1:]:
                if row[h] == 1 and h == 'obscene':
                    logits = model.predict(row['comment_text'], tokenizer)
                    sigmoid = torch.nn.Sigmoid()
                    probs = sigmoid(logits.squeeze().cpu()) 
                    predictions = np.zeros(probs.shape)
                    argmax_idx = np.argmax(probs[2:]) + 2
                    predictions[argmax_idx] = 1
                    predicted_labels = [toxicity_dataset.id2label[idx] for idx, label in enumerate(predictions) if label]
                    for label in predicted_labels:
                        score[label] = 1
                elif row[h] == 1 and h != 'obscene':
                    score[convert_labels[h]] = 1

            new_df["Message"].append(row['comment_text'])
            for c in new_columns:
                new_df[c].append(score[c])

        return pd.DataFrame(data=new_df)


if __name__ == '__main__':
    # Load dataset
    start_time = time.time()
    toxicity_dataset = ToxicityDataset()
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('./automod-model/model_versions/automod-model-training-10')
    toxicity_model = ToxicityModel(model_name='./automod-model/model_versions/automod-model-training-10', 
                                   num_labels=len(toxicity_dataset.labels), 
                                   id2label=toxicity_dataset.id2label, 
                                   label2id=toxicity_dataset.label2id)

    data_convertor = ConvertDataset(toxicity_model, tokenizer)
    dt = pd.read_csv('./automod-model/data_augmenting/toxic_jigsaw.csv')
    data_translator = DataTranslator('en', 'ro')

    dt.drop(columns=["id", "toxic", "severe_toxic"], inplace=True)
    headers = dt.columns.tolist()
    dt = dt[dt[headers[1:]].sum(axis=1) > 0]
    dt_filtered = data_translator.filter_data(dt, headers)
    data_translator.remove_duplicates(dt_filtered, 'comment_text')
    dt_filtered = dt_filtered.sample(frac=1).reset_index(drop=True)
    dt_filtered = dt_filtered[:30]

    # parallel translations
    print(f'Translating {len(dt_filtered)} messages...')
    dt_filtered = data_translator.translate_data(dt_filtered, 'comment_text')


    dt_filtered = data_translator.filter_data(dt_filtered, headers)

    dt_final = data_convertor.equivalent_labeling(dt_filtered, headers)
    print(dt_filtered.head())
    dt_final.to_csv('./automod-model/data_dumps/toxic_jigsaw_to_op.csv', index=False, encoding='utf-8')
    end_time = time.time()
    print(f'Execution ended in {(end_time - start_time):.2f} seconds.')
    ### toxic_jigsaw.csv | youtoxic_english.csv