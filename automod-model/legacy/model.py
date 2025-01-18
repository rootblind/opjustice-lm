"""
Multi-label text classification model.

The goal of this model is to accurately score messages in romanian language based on toxicity labels ['OK','Insult','Violence','Sexual','Hateful','Flirt','Spam','Aggro']
Where OK is a message that has no toxicity and the others as the name implies.
The model is trained on a dataset based on discord messages from a few servers and it is oriented towards the moderation levels of League of Legends Romania discord server

opjustice-lm is based on readerbench/RoBERT-base language model from the huggingface hub.

The code below is based off of the notebook example from google colabs
https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=mEkAQleMMT0k
"""

import os
import torch
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
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
            dataset = load_dataset(self.dataset_name, data_files={"data": "data.csv", "train": "train.csv", "test": "test.csv"})
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


class ToxicityTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, output_dir, batch_size=16, metric_name="f1", epochs=8):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.metric_name = metric_name
        self.epochs = epochs
        self.trainer = self.initialize_trainer()

    def multi_label_metrics(self, predictions, labels, threshold=0.5):
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        metrics = {'f1': f1_micro_average, 'roc_auc': roc_auc, 'accuracy': accuracy}
        return metrics

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        return self.multi_label_metrics(preds, p.label_ids)

    def initialize_trainer(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            load_best_model_at_end=True,
            metric_for_best_model=self.metric_name,
            # experimenting with
            save_steps=500,
            weight_decay=0.01,
            logging_dir='./automod-model/logs',
            logging_steps=10,
            lr_scheduler_type='linear',
            warmup_steps=500
        )

        return Trainer(
            self.model,
            training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

    def train(self):
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        self.trainer.train()

    def evaluate(self):
        return self.trainer.evaluate()

    def save_model(self):
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        self.trainer.save_model(self.output_dir)


if __name__ == "__main__":
    start_time = time.time()
    # Load dataset
    toxicity_dataset = ToxicityDataset()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('readerbench/RoBERT-small')
    toxicity_model = ToxicityModel(model_name='readerbench/RoBERT-small', 
                                   num_labels=len(toxicity_dataset.labels), 
                                   id2label=toxicity_dataset.id2label, 
                                   label2id=toxicity_dataset.label2id)
    
    # Encode dataset
    encoded_dataset = toxicity_dataset.encode_dataset(tokenizer)

    # Initialize trainer
    trainer = ToxicityTrainer(model=toxicity_model.model, 
                              tokenizer=tokenizer, 
                              train_dataset=encoded_dataset['train'], 
                              eval_dataset=encoded_dataset['test'], 
                              output_dir='./automod-model/model_versions/v1-test',
                              batch_size=1,
                              epochs=1
                              )

    # Train and evaluate model
    trainer.train()
    trainer.evaluate()
    trainer.save_model()

    # Example inference
    text = "mai taci in rasa ta"
    logits = toxicity_model.predict(text, tokenizer)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.6)] = 1
    predictions[np.argmax(probs)] = 1
    predicted_labels = [toxicity_dataset.id2label[idx] for idx, label in enumerate(predictions) if label]
    
    print(predicted_labels)
    print(predictions)
    print(probs)
    end_time = time.time()
    print(f'Execution time: {(end_time - start_time):.2f} seconds.')