"""
Just a little script to load the model and send an unlabeled csv as input and output the labeled and filtered version.
I combined the python_tools script to make the process smoother
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from datasets import load_dataset, load_from_disk
import os
import csv
import regex as re
from transformers import pipeline

if os.path.exists('./automod-model/dataset'):
    dataset = load_from_disk('./automod-model/dataset')
else:
    os.mkdir('./automod-model/dataset')
    dataset = load_dataset('rootblind/opjustice-dataset')
    dataset.save_to_disk('./automod-model/dataset')
# Load the tokenizer
model_dir = './automod-model/model_versions/automod-model-training-9'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Define the labels and create id2label and label2id mappings
labels = [label for label in dataset['train'].features.keys() if label != 'Message']
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

helper_model = AutoModelForSequenceClassification.from_pretrained('readerbench/ro-offense').to(device)
helper_tokenizer = AutoTokenizer.from_pretrained('readerbench/ro-offense')

# Preprocessing function to encode the data
def preprocess_data(examples):
    # Encode the texts
    text = examples["Message"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
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
def send_input_rooffense(text):
    #pipe = pipeline('text-classification', model='readerbench/ro-offense', device=device)
    labels = {
        0: 'OK',
        1: 'Profanity',
        2: 'Insult',
        3: 'Abuse'
    }
    #return labels[pipe(text)[0]['label']]
    encoding = helper_tokenizer(text, return_tensors="pt")
    encoding = {k: v.to(helper_model.device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = helper_model(**encoding)
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)

    predictions[np.argmax(probs)] = 1
    predicted_label = [labels[idx] for idx, label in enumerate(predictions) if label == 1.0]
    return predicted_label[0]

# Tokenize the text and sending it as input to the model
def send_input(text, print_text=True, tensor_output=False):
    encoding = tokenizer(text, return_tensors="pt")
    encoding = {k: v.to(model.device) for k, v in encoding.items()}

    # Get model outputs
    with torch.no_grad():
        outputs = model(**encoding)

    # Process the outputs
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)

    predictions[np.where(probs >= 0.5)] = 1
    predictions[np.argmax(probs)] = 1

    # Convert predicted ids to actual label names
    predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    if 'OK' in predicted_labels:
        predicted_labels = ['OK']

    # Print the predicted labels
    if print_text:
        print(text[:50] + '...', predicted_labels)
        print(predictions)
        print(probs)
    if tensor_output:
        return probs.tolist()
    else:
        return predicted_labels


def label_dataset(source):
    # source is the dataset as a list
    labeled_data = []
    for message in source:
        scores = {
            'OK': 0,
            'Insult': 0,
            'Violence': 0,
            'Sexual': 0,
            'Hateful': 0,
            'Flirt': 0,
            'Spam': 0,
            'Aggro': 0
        }
        labels = send_input(message, False)
        for label in labels:
            scores[label] = 1
        output_row = {'Message': message}
        output_row.update(scores)
        labeled_data.append(output_row)
    print('Labeling dataset was executed.')
    return labeled_data

def label_dataset_v2(source): # this labeling method uses readerbench/ro-offense to help correct our model
    # source is the dataset as a list
    labeled_data = []
    for message in source:
        scores = {
            'OK': 0,
            'Insult': 0,
            'Violence': 0,
            'Sexual': 0,
            'Hateful': 0,
            'Flirt': 0,
            'Spam': 0,
            'Aggro': 0
        }
        labels = send_input(message, False)
        helper_tag = send_input_rooffense(message) # sending the input to the helper model as well
        if helper_tag == 'OK':
            labels = ['OK']

        if helper_tag == 'Abuse' and 'OK' in labels:
            labels.remove('OK')
            probs_list = send_input(message, False, True)
            abuse_list = probs_list[2:5]
            abuse_tags = ['Violence', 'Sexual', 'Hateful']
            abuse_tag = abuse_tags[abuse_list.index(max(abuse_list))]
            labels.append(abuse_tag)


        for label in labels:
            scores[label] = 1
        output_row = {'Message': message}
        output_row.update(scores)
        labeled_data.append(output_row)
    print('Labeling dataset was executed.')
    return labeled_data

def filter_dataset(source):
    # source is the list of rows
    alphabet_pattern = re.compile(r'^[a-zA-Z]')
    allowed_pattern = re.compile(r'[^a-zA-Z0-9 -]')
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    messages = []
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
    
    for row in source:
        message = row['Message']
        message = filter(message)
        if ' ' not in message:
            continue
        if len(message) > 2 and alphabet_pattern.search(message) and allowed_pattern.fullmatch(message):
            messages.append(message)
    messages = list(set(messages))
    print('Filter executed.')

    return messages

def write_csv(source, destination):
    # source is the list of rows
    columns = ['Message','OK','Insult','Violence','Sexual','Hateful','Flirt','Spam','Aggro']
    with open(destination, "w", newline='', encoding='utf-8') as dest_file:
        writer = csv.DictWriter(dest_file, fieldnames=columns)
        writer.writeheader()
        for row in source:
            writer.writerow(row)

    print('Writing executed.')

def append_csv(source, destination):
    # source is the list of rows
    columns = ['Message','OK','Insult','Violence','Sexual','Hateful','Flirt','Spam','Aggro']
    with open(destination, "a", newline='', encoding='utf-8') as dest_file:
        writer = csv.DictWriter(dest_file, fieldnames=columns)
        for row in source:
            writer.writerow(row)

    print('Appending executed.')

def load_source(filename):
    data = []
    with open(filename, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def executor(source, destination):
    reader = load_source(source) # reading the unlabeled data
    data = filter_dataset(reader) # filter the data to be formatted for processing
    labeled_data = label_dataset(data) # send the rows (the raw text) to the model to be scored and write the text with its results in csv format
    #append_csv(labeled_data, destination) # alternative to append the dataset directly
    write_csv(labeled_data, destination) # alternative to create the file instead; I prefer this approach because I can review the dataset before appending
    # especially useful to correct the dataset for the next training loop
    print('Executor finished.')

### DO NOT FORGET to add the Message column to the unlabeled dataset
executor('./automod-model/train-unlabeled.csv', './automod-model/partial-train.csv')
