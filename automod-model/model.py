"""
Multi-label text classification model.

The goal of this model is to accurately score messages in romanian language based on toxicity labels ['OK','Insult','Violence','Sexual','Hateful','Flirt','Spam','Aggro']
Where OK is a message that has no toxicity and the others as the name implies.
The model is trained on a dataset based on discord messages from a few servers and it is oriented towards the moderation levels of League of Legends Romania discord server

opjustice-lm is based on readerbench/RoBERT-base language model from the huggingface hub.

The code below is based off of the notebook example from google colabs
https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=mEkAQleMMT0k
"""

#importing the necessary libraries

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
import os

# I like the way hugging face manages datasets, so i run this code first where basically 
# if the dataset directory exists, then it loads the dataset from that directory (there is no checking if the directory has the right contents)
# if not, the directory is created, the dataset is loaded from my repo from huggingface and it is stored locally in the newly created directory
if os.path.exists('./automod-model/dataset'):
    dataset = load_from_disk('./automod-model/dataset')
else:
    os.mkdir('./automod-model/dataset')
    dataset = load_dataset('rootblind/opjustice-dataset')
    dataset.save_to_disk('./automod-model/dataset')

# storing the labels into a list
labels = [label for label in dataset['train'].features.keys() if label not in 'Message']
# creating id to label and label to id dictionaries
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}


model_name = './automod-model/model_versions/automod-model-training-6' #'readerbench/RoBERT-base'

# loading the tokenizer, device and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id).to(device) # moving the model to cuda if exists


def preprocess_data(examples):
  # take a batch of texts
  text = examples["Message"]
  # encode them with padding to the maximum length
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding

# processing the dataset and formatting it for pytorch
encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
encoded_dataset.set_format("torch")

# preparing variables
batch_size = 8 # i have a RTX 2050 on a Lenovo laptop, it has 4GB so I am ranging between 8-16 batches
metric_name = "f1"
output_dir = './automod-model/model_versions/automod-model-training-7.1'

args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

#forward pass
input_ids_todevice = encoded_dataset['train']['input_ids'][0].unsqueeze(0).to(device) # making sure all tensors are on the same device
labels_todevice = encoded_dataset['train'][0]['labels'].unsqueeze(0).to(device) # making sure all tensors are on the same device
outputs = model(input_ids=input_ids_todevice, labels=labels_todevice) # loading the model for inference

for name, param in model.named_parameters():
    if not param.is_contiguous():
        param.data = param.data.contiguous()

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

trainer.save_model(output_dir)

# Inference
# Sending a text as input, preprocessing it for the model to analyze and output
text = "il iubiti la cat il pomeniti, taceti odata in plm ca e annoying si nu mai plangeti dupa el ca are ban de o sapt, va zice si blind oribii drq"
encoding = tokenizer(text, return_tensors="pt")
encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

with torch.no_grad():
    outputs = model(**encoding)

logits = outputs.logits
# apply sigmoid + threshold
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)

# predictions that are above 50% accurate are scored with 1
predictions[np.where(probs >= 0.5)] = 1

# In early stages, the model can be unsure of the labels, scoring everything under 0.5, therefore the most "confident" answer will be taken
# normally the answer is either OK or not OK and other labels
predictions[np.argmax(probs)] = 1
# turn predicted id's into actual label names
predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label]
print(predicted_labels)
print(predictions)
print(probs)
