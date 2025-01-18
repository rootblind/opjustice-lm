"""
Multi-label text classification model.

The goal of this model is to accurately score messages in romanian language based on toxicity labels ['OK','Insult','Violence','Sexual','Hateful','Flirt','Spam','Aggro']
Where OK is a message that has no toxicity and the others as the name implies.
The model is trained on a dataset based on discord messages from a few servers and it is oriented towards the moderation levels of League of Legends Romania discord server

opjustice-lm is based on readerbench/RoBERT-base language model from the huggingface hub.

The code below is based off of the notebook example from google colabs
https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=mEkAQleMMT0k
"""
import torch
import numpy as np
from transformers import AutoTokenizer
import time
from optoolkit import Model, DatasetLoader, ModelTrainer

if __name__ == "__main__":
    start_time = time.time()
    # Load dataset
    dataset = DatasetLoader(["Message"], "Message")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('./automod-model/model_versions/v1')
    model = Model(model_name='readerbench/RoBERT-small', 
                                   num_labels=len(dataset.labels), 
                                   id2label=dataset.id2label, 
                                   label2id=dataset.label2id)
    
    # Encode dataset
    encoded_dataset = dataset.encode_dataset(tokenizer)

    # Initialize trainer
    trainer = ModelTrainer(model=model.model, 
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
    model = Model(trainer.output_dir, len(dataset.labels), dataset.id2label, dataset.label2id)
    # Example inference
    text = "mai taci in rasa ta"
    predicted_labels = model.label_text(text, tokenizer, dataset.labels)
    logits = model.predict(text, tokenizer)
    print(predicted_labels)
    print(logits)
    end_time = time.time()
    print(f'Execution time: {(end_time - start_time):.2f} seconds.')