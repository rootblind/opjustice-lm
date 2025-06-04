from optoolkit import CNNTransformerClassifier, CNNModelTrainer, DatasetLoader
from transformers import AutoTokenizer, AutoConfig
import torch
import os
from safetensors.torch import load_file
os.environ["WANDB_DISABLED"] = "true"

dataset = DatasetLoader(["Message"], "Message")

name = "readerbench/RoBERT-small"

tokenizer = AutoTokenizer.from_pretrained(name)
config = AutoConfig.from_pretrained(name,
                                    _name_or_path=name,
                                    num_labels=5
                                    )

encoded_dataset = dataset.encode_dataset(tokenizer)

model = CNNTransformerClassifier(config)

trainer = CNNModelTrainer(
    model, tokenizer, encoded_dataset["train"], encoded_dataset["test"],
    "automod-model/model_versions/v4-cnn",
    32, "f1", 8

)

trainer.train()
trainer.evaluate()
trainer.save_model()

model = CNNTransformerClassifier(config).to(torch.device("cuda"))
state_dict = load_file(f"{name}/model.safetensors")
model.load_state_dict(state_dict)
model.eval()
text = "esti urat si un prost omoarate"
labels = model.label_text(text, tokenizer)
print(labels)