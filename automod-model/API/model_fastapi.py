from fastapi import FastAPI, HTTPException,Request
# from pydantic import BaseModel
import uvicorn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset, load_from_disk
import os


# loading the model and methods for the training loop and testing

if os.path.exists('./automod-model/dataset'):
    dataset = load_from_disk('./automod-model/dataset')
else:
    os.mkdir('./automod-model/dataset')
    dataset = load_dataset('rootblind/opjustice-dataset')
    dataset.save_to_disk('./automod-model/dataset')

app = FastAPI()

model_dir = './automod-model/model_versions/v1'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels = [label for label in dataset['train'].features.keys() if label != 'Message']
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}


model = AutoModelForSequenceClassification.from_pretrained(model_dir,
                                                           local_files_only=True,
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id).to(device)

model.eval()

def process_input(text):
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
    if 'OK' in predicted_labels:
        predicted_labels = ['OK']
    # Print the predicted labels
    return {"labels":predicted_labels}


    

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/hello")
def read_root():
    return {"Hello": "Hello"}

@app.post("/classify")
async def classify(request: Request):
    data = await request.json()
    if not 'text' in data:
        raise HTTPException(status_code=400, detail="Text input required")
    
    response = process_input(data['text'])
    return response

if __name__ == "__main__":
    uvicorn.run("model_fastapi:app", host='127.0.0.1', port=8080, reload=True)