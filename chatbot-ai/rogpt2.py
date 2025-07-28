import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Pre-trained GPT-2 Model and Tokenizer
model_name = "readerbench/RoGPT2-base"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|start_turn|>", "<|end_turn|>", "<|separator|>", "<|padding|>"]
})
tokenizer.add_special_tokens({'pad_token': "<|padding|>"})
model.resize_token_embeddings(len(tokenizer))
model.to(device)
# 2. Add Special Tokens if Needed (Optional)
# If you need to add custom tokens like start/end markers, do it here
# tokenizer.add_special_tokens({"additional_special_tokens": ["<|start_turn|>", "<|end_turn|>"]})
# model.resize_token_embeddings(len(tokenizer))

# 3. Load Your Text Data (Example: from local file)
# Assuming you have a plain text file (data.txt)
file_path = "chatbot-ai/dataset/data_tokenized.txt"

# Load your text data
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# Split text into lines (or use other logic based on your data)
texts = text_data.splitlines()

# 4. Tokenize the Data
# Tokenize each line of text into the appropriate format for GPT-2
def encode_function(examples):
    encodings = tokenizer(examples['text'], return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
    encodings["labels"] = encodings["input_ids"].detach().clone()

    return encodings

# Convert to Dataset object
from datasets import Dataset
dataset = Dataset.from_dict({"text": texts})

# Tokenize the dataset
tokenized_datasets = dataset.map(encode_function, batched=True, remove_columns=["text"])

# 5. Split Dataset into Train and Validation Sets
train_size = int(0.9 * len(tokenized_datasets))
train_dataset = tokenized_datasets.select(range(train_size))
val_dataset = tokenized_datasets.select(range(train_size, len(tokenized_datasets)))

# 6. Set Training Arguments
training_args = TrainingArguments(
    output_dir="./chatbot-ai/versions/v1",          # Output directory
    overwrite_output_dir=True,       # Overwrite the output directory if it exists
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=4,   # Batch size for training
    per_device_eval_batch_size=8,    # Batch size for evaluation
    save_steps=10_000,               # Save the model every 10,000 steps
    save_total_limit=2,              # Only keep the last 2 models
    logging_dir="./chatbot-ai/logs",            # Directory for logs
    logging_steps=500,               # Log every 500 steps
    eval_strategy="epoch",     # Evaluate once every epoch
    disable_tqdm=False,
)

# 7. Initialize the Trainer
trainer = Trainer(
    model=model,                         # The model to be trained
    args=training_args,                  # Training arguments
    train_dataset=train_dataset,         # Training dataset
    eval_dataset=val_dataset,            # Validation dataset
)

# 8. Start Fine-Tuning
trainer.train()

# 9. Save the Model After Training
model.save_pretrained("./chatbot-ai/versions/v1/fine_tuned_gpt2")
tokenizer.save_pretrained("./chatbot-ai/versions/v1/fine_tuned_gpt2")
