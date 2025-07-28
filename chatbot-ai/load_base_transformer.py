from optoolkit import DataProcessor, BaseTransformerModel, DatasetLoader, get_lora_model
import torch
from minbpe import BasicTokenizer
import time


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
tokenizer.load("chatbot-ai/tokenizer/large/tokenizer.model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size=512
n_embd=512
n_head=4
n_layer=2
dropout=0.2
seq_file = "chatbot-ai/dataset/sequence.txt"
batch_size = 32

version = "chatbot-ai/versions/prototype"

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

model = add_lora_layers(model, device) # uncomment to train with lora layers

checkpoint = torch.load(f"{version}/checkpoint_0.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

input_tokens = tokenizer.encode("urmatoarea zi ")
input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device=device)
with torch.no_grad():
    output = model.generate(input_tokens=input_tokens, max_new_tokens=100)

print(tokenizer.decode(output[0].tolist()))