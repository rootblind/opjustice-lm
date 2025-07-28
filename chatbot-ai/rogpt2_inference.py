from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model_path = "./chatbot-ai/versions/v1/fine_tuned_gpt2"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Define special tokens
start_token = "<|start_turn|>"
end_token = "<|end_turn|>"
separator_token = "<|separator|>"
eos_token = "<|endoftext|>"


# Function to generate a response
def generate_response(user_input):
    # Prepare the prompt
    prompt = f"{start_token}user{separator_token}{user_input}{end_token}"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
    
    # Generate the response with max_new_tokens instead of max_length
    output = model.generate(
        inputs["input_ids"],  # Input token IDs
        max_new_tokens=150,    # Limit the number of new tokens to generate
        num_beams=5,           # Beam search for better generation
        no_repeat_ngram_size=2,  # Prevent repeating n-grams
        early_stopping=True,   # Stop generation if EOS token is encountered
        eos_token_id=tokenizer.encode(eos_token)[0],  # Stop at <|endoftext|>
        pad_token_id=tokenizer.pad_token_id,   # Ensure padding is handled
        top_k=50,         # Limits the number of highest probability tokens for sampling
    )

    # Decode and return the generated text
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the response after the user input
    print(response)
    response = response.split(f"{start_token}response{separator_token}")[1].split(end_token)[0].strip()

    return response

# Example of how to use the function
user_input = "henlo senpai"
response = generate_response(user_input)
print(f"Response: {response}")
