"""
Loading the model for testing inputs and analyze the outputs
"""

from transformers import AutoTokenizer
from optoolkit import Model, DatasetLoader

if __name__ == "__main__":
    # Load dataset
    dataset = DatasetLoader(["Message"], "Message")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('./automod-model/model_versions/v4')
    model = Model(model_name='./automod-model/model_versions/v4', 
                                   num_labels=len(dataset.labels), 
                                   id2label=dataset.id2label, 
                                   label2id=dataset.label2id)
    
    text = 'esti un terminat si un trist'
    print(model.label_text(text, tokenizer, dataset.labels))
    print(model.predict(text, tokenizer))