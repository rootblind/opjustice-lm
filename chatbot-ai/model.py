from optoolkit import DataProcessor, BaseTransformerModel, DatasetLoader, FineTuning
import torch
from minbpe import BasicTokenizer, RegexTokenizer
import time

tokenizer = RegexTokenizer()
tokenizer.load("chatbot-ai/tokenizer/large/tokenizer.model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size=512
n_embd=512
n_head=4
n_layer=2
dropout=0.2
seq_file = "chatbot-ai/dataset/sequence.txt"
batch_size = 32
split = 0.95
learning_rate = 1e-4
eval_interval = 5

version = "chatbot-ai/versions/prototype"

dp = DataProcessor()

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

# Check model device
print(f"Model is on: {next(model.parameters()).device}")

finetuner = FineTuning(
    data_file="chatbot-ai/dataset/conversations.json",
    tokenizer=tokenizer,
    model=model,
    model_path="chatbot-ai/versions/prototype/checkpoint_0.pth",
    file_dir="chatbot-ai/versions/proto_finetuned",
    split=split,
    block_size=block_size,
    batch_size=batch_size,
    device=device,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
    learning_rate=learning_rate,
    max_iters=2,
    eval_interval=eval_interval,
    use_lora=True,
    use_scheduler=True
)
print(f"Tokenizer vocab size: {len(tokenizer.vocab) + len(tokenizer.special_tokens)}")
print(f"Model vocab size: {model.get_vocab_size()}")
finetuner.train()