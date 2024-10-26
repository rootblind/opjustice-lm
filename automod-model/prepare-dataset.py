import os
import torch
import csv
import numpy as np
import regex as re
import pandas as pd
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
        encoding = tokenizer(text, return_tensors="pt")
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

# prepares the unlabeled data by filtering and curating it
class UnlabeledData:
    def __init__(self, filename, header="Message"):
        self.filename = filename
        self.header = header
        self.raw_data = self.csv_to_data()
        self.data = self.filter_data(self.raw_data, self.header)
        self.df = self.remove_duplicates(self.data, self.header)

    def csv_to_data(self):
        data = []
        with open(self.filename, 'r', newline='', encoding='utf-8') as read_file:
            reader = csv.DictReader(read_file)
            for row in reader:
                data.append(row)
        return data
    def filter_data(self, data, header):
        alphabet_pattern = re.compile(r'^[a-zA-Z]')
        allowed_pattern = re.compile(r'[^a-zA-Z0-9 ]')
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

        filtered_data = []

        def filter(message):
            if len(message) < 3:
                return message
            message = message.replace('+rep', '')
            message = message.replace('-rep', '')
            message = message.replace('\n', ' ').replace('\r', ' ')
            message = re.sub(url_pattern, '', message)
            message = re.sub(allowed_pattern, '', message)
            message = message.lstrip()
            return message
        
        for row in data:
            text = filter(row[header])
            if ' ' not in text:
                continue

            if(len(text) > 2 and alphabet_pattern.search(text)):
                row[header] = text
                filtered_data.append(row)
        return filtered_data
    
    def remove_duplicates(self, data, header):
        pd_data = {header: []}

        for row in data:
            pd_data[header].append(row[header])
        
        df = pd.DataFrame(data=pd_data)
        df.drop_duplicates(subset=header, inplace=True)
        return df
    
class DatasetClassifier:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def label_dataset(self, dataset):
        scores = {
                "OK": [],
                "Aggro": [],
                "Violence": [],
                "Sexual": [],
                "Hateful": []
            }
        for _, row in dataset.iterrows():
            labels = self.model.label_text(row['Message'], tokenizer)
            for key in scores:
                if key in labels:
                    scores[key].append(1)
                else:
                    scores[key].append(0)
        for key in scores:
            dataset[key] = scores[key]
        return dataset
    
    def dataset_to_csv(self, dataset, filename):
        dataset.to_csv(filename, index=False, encoding='utf-8')

class DatasetFocusLabel:
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def focus_dataset(self):
        dataset = self.dataset
        labels = self.labels
        focused_pd = {
            "Message": [],
            "OK": [],
            "Aggro": [],
            "Violence": [],
            "Sexual": [],
            "Hateful": []
        }

        for _, row in dataset.iterrows():
            text = row["Message"]
            for label in labels:
                if row[label] == 1:
                    focused_pd["Message"].append(text)
                    focused_pd["OK"].append(int(row["OK"]))
                    focused_pd["Aggro"].append(int(row["Aggro"]))
                    focused_pd["Violence"].append(int(row["Violence"]))
                    focused_pd["Sexual"].append(int(row["Sexual"]))
                    focused_pd["Hateful"].append(int(row["Hateful"]))
                    break
        return pd.DataFrame(data=focused_pd)



if __name__ == "__main__":
    # Load dataset
    toxicity_dataset = ToxicityDataset()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('./automod-model/model_versions/automod-model-training-10')
    toxicity_model = ToxicityModel(model_name='./automod-model/model_versions/automod-model-training-10', 
                                   num_labels=len(toxicity_dataset.labels), 
                                   id2label=toxicity_dataset.id2label, 
                                   label2id=toxicity_dataset.label2id)
    
    unlabeled_data = UnlabeledData('./automod-model/train-unlabeled.csv')

    df = pd.read_csv('./automod-model/partial-dataset.csv')
    
    df_focus = DatasetFocusLabel(df, ["Aggro", "Violence", "Sexual", "Hateful"]).focus_dataset()

    classifier = DatasetClassifier(toxicity_model, tokenizer)
    classifier.dataset_to_csv(df_focus, './automod-model/focus-notok.csv')