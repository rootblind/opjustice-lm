from optoolkit import DataProcessor, BaseTransformerModel, DatasetLoader, get_lora_model
import torch
from minbpe import BasicTokenizer, RegexTokenizer
import time

tokens = {
    "start": "<|start_turn|>",
    "end": "<|end_turn|>",
    "separator": "<|separator|>",
    "eos": "<|endoftext|>"
}
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

def get_input_tokens(turns: list[dict]) -> list[int]:
    formatted_input = ""
    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        formatted_input += f"{tokens['start']}{role}{tokens['separator']}{content}{tokens['end']}"

    formatted_input += f"{tokens['start']}response{tokens['separator']}"

    input_tokens = tokenizer.encode(formatted_input, allowed_special="all")
    input_tokens = torch.tensor(input_tokens, dtype=torch.long)
    input_tokens = input_tokens.unsqueeze(0).to(device)
    return input_tokens

def get_generated_message(input_tokens: list[int]) -> str:
    model_answer = ""
    model.eval()
    while True:
        try:
            output_tokens = model.advanced_generation(
                input_tokens=input_tokens, max_new_tokens=1, temperature=.9, top_k=50, top_p=None)
            last_generated_token = output_tokens[0, -1].item()
            if last_generated_token == tokenizer.special_tokens["<|endoftext|>"]:
                break

            if last_generated_token == tokenizer.special_tokens["<|end_turn|>"]:
                break

            input_tokens = torch.cat(
                (input_tokens, output_tokens[:, -1:]), dim=1)
            model_answer += tokenizer.decode([last_generated_token])

            if len(output_tokens[0]) > block_size:
                break
        except Exception:
            continue

    return model_answer

if __name__ == "__main__":

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

    version = "chatbot-ai/versions/proto_finetuned"

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
    model = add_lora_layers(model, device) # uncomment if lore is used in training
    checkpoint = torch.load(f"{version}/checkpoint_1.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    text_input = "nu am tastat gresit"
    
    turns = [
        {
            "role": "user",
            "content": text_input
        },
    ]

    input_tokens = get_input_tokens(turns=turns)
    model_answer = get_generated_message(input_tokens=input_tokens)

    turns.append({
        "role": "response",
        "content": model_answer
    })

    # turns = turns[:-2] # Uncomment this if you want to retry the generation
    for turn in turns:
        role = turn["role"]
        if role == "user":
            print(f"You: {turn['content']}")
        elif role == "response":
            print(f"Response: {turn['content']}")