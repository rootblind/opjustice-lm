from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_path = "./chatbot-ai/versions/v1/fine_tuned_gpt2"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

start_token = "<|start_turn|>"
end_token = "<|end_turn|>"
separator_token = "<|separator|>"
eos_token = "<|endoftext|>"


def generate_response(user_input):
    prompt = f"{start_token}user{separator_token}{user_input}{end_token}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
    
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=150,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        eos_token_id=tokenizer.encode(end_token)[0],  # Stop at <|end_turn|>
        pad_token_id=tokenizer.pad_token_id,
        top_k=50,
    )


    response = tokenizer.decode(output[0], skip_special_tokens=True)

    print(response)
    response = response.split(f"{start_token}response{separator_token}")[1].split(end_token)[0].strip()

    return response

user_input = "henlo senpai"
response = generate_response(user_input)
print(f"Response: {response}")
