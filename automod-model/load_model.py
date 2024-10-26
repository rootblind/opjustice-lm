"""
Loading the model for testing inputs and analyze the outputs

Most of the code below is taken from model.py so I won't recomment it
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from datasets import load_dataset, load_from_disk
import os


# loading the model and methods for the training loop and testing

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
    

if __name__ == "__main__":
    # Load dataset
    toxicity_dataset = ToxicityDataset()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('./automod-model/model_versions/automod-model-training-10')
    toxicity_model = ToxicityModel(model_name='./automod-model/model_versions/automod-model-training-10', 
                                   num_labels=len(toxicity_dataset.labels), 
                                   id2label=toxicity_dataset.id2label, 
                                   label2id=toxicity_dataset.label2id)
    
    print(toxicity_model.label_text("test in pizda si cristosii matii", tokenizer))