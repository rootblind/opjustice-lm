from optoolkit import DataProcessor, BaseTransformerModel, DatasetLoader, BaseModelTrainer, get_lora_model
import torch
from minbpe import BasicTokenizer, RegexTokenizer
import json

def add_lora_layers(model: BaseTransformerModel, device: torch.device, rank: int = 4, alpha: int = 8):
    model = get_lora_model(
        model=model,
        lora_config={
            "rank": rank,
            "alpha": alpha
        },
        device=device
    )

    return model

tokenizer = BasicTokenizer()
tokenizer.load("chatbot-ai/tokenizer/tokenizer.model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size=256
n_embd=256
n_head=4
n_layer=4
dropout=0.1
batch_size = 32
lora_rank = 4
lora_alpha = 16
seq_file = "chatbot-ai/dataset/sequence-500k.txt"

model = BaseTransformerModel(
    tokenizer=tokenizer,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
    device=device,
    ignore_index=tokenizer.special_tokens["<|padding|>"]
).to(device)

model = add_lora_layers(model, device, lora_rank, lora_alpha) # uncomment to train with lora layers
# Check model device
print(f"Model is on: {next(model.parameters()).device}")

dataset = DatasetLoader(
    dataset_file=seq_file,
    tokenizer=tokenizer,
    block_size=block_size,
    batch_size=batch_size,
    device=device
)

train, test = dataset.get_dataloaders()

# Check batch device (optional, add inside training loop if needed)
sample_batch = next(iter(train))
print(f"Sample batch is on: {sample_batch[0].device}, {sample_batch[1].device}")

trainer = BaseModelTrainer(
    model=model,
    train_loader=train,
    val_loader=test,
    file_dir="chatbot-ai/versions/prototype",
    max_iters=4,
    use_scheduler=True
)

trainer.train()