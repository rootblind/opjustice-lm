import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "readerbench/RoGPT2-base"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|start_turn|>", "<|end_turn|>", "<|separator|>", "<|padding|>"]
})
tokenizer.add_special_tokens({'pad_token': "<|padding|>"})
model.resize_token_embeddings(len(tokenizer))
model.to(device)

file_path = "chatbot-ai/dataset/data_tokenized.txt"


with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

texts = text_data.splitlines()


def encode_function(examples):
    encodings = tokenizer(examples['text'], return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
    encodings["labels"] = encodings["input_ids"].detach().clone()

    return encodings

from datasets import Dataset
dataset = Dataset.from_dict({"text": texts})


tokenized_datasets = dataset.map(encode_function, batched=True, remove_columns=["text"])


train_size = int(0.9 * len(tokenized_datasets))
train_dataset = tokenized_datasets.select(range(train_size))
val_dataset = tokenized_datasets.select(range(train_size, len(tokenized_datasets)))


training_args = TrainingArguments(
    output_dir="./chatbot-ai/versions/v1",
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./chatbot-ai/logs",
    logging_steps=500,
    eval_strategy="epoch",
    disable_tqdm=False,
    learning_rate=1e-4,
    #load_best_model_at_end=True,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_steps=500

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 8. Start Fine-Tuning
trainer.train()

# 9. Save the Model After Training
model.save_pretrained("./chatbot-ai/versions/v1/fine_tuned_gpt2")
tokenizer.save_pretrained("./chatbot-ai/versions/v1/fine_tuned_gpt2")
