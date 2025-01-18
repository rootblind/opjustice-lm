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

if os.path.exists('./automod-model/side-model/dataset'):
    dataset = load_from_disk('./automod-model/side-model/dataset')
else:
    os.mkdir('./automod-model/side-model/dataset')
    dataset = load_dataset('rootblind/opjustice_side-model-dataset')
    dataset.save_to_disk('./automod-model/side-model/dataset')
# Load the tokenizer
model_dir = './automod-model/side-model/model_versions/v1-fold-1'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the labels and create id2label and label2id mappings
labels = [label for label in dataset['train'].features.keys() if label != 'Message']
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

# Preprocessing function to encode the data
def preprocess_data(examples):
    # Encode the texts
    text = examples["Message"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=400)
    # Add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()
    return encoding

# Preprocess the dataset
encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
encoded_dataset.set_format("torch")
# Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(model_dir,
                                                           local_files_only=True,
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id).to(device)

# Set model to evaluation mode
model.eval()

# Define the text for inference


# Tokenize the text
def send_input(text):
    encoding = tokenizer(text, return_tensors="pt")
    encoding = {k: v.to(model.device) for k, v in encoding.items()}

    # Get model outputs
    with torch.no_grad():
        outputs = model(**encoding)

    # Process the outputs
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    # relu
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)

    predictions[np.where(probs >= 0.5)] = 1
    predictions[np.argmax(probs)] = 1

    # Convert predicted ids to actual label names
    predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]

    # Print the predicted labels
    print(text[:50] + '...', predicted_labels)
    print(predictions)
    print(probs)
    return predicted_labels

print(send_input('sa imi bag pula in ma-ta'))