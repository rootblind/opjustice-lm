import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optoolkit import CNNTransformerClassifier, CNNModelTrainer, DatasetLoader
from transformers import AutoTokenizer
import torch

dataset = DatasetLoader(["Message"], "Message")

name = "readerbench/RoBERT-small"

tokenizer = AutoTokenizer.from_pretrained(name)
encoded_dataset = dataset.encode_dataset(tokenizer)

model = CNNTransformerClassifier(name, len(dataset.labels), dataset.id2label, dataset.label2id,
                                 device=torch.device("cuda")
                                 )

trainer = CNNModelTrainer(
    model, tokenizer, encoded_dataset["train"], encoded_dataset["test"],
    "automod-model/model_versions/v4-cnn",
    32, "f1", 8

)

trainer.train()
trainer.evaluate()
trainer.save_model()

model = CNNTransformerClassifier(
    trainer.output_dir, len(dataset.labels), dataset.id2label, dataset.label2id, torch.device("cuda")
)

text = "mai taci in rasa ma-tii"
labels = model.label_text(text, tokenizer)
print(labels)