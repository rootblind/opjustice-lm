from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from transformers import BitsAndBytesConfig
from peft import LoraConfig, PeftModel, TaskType
import torch

# Load your fine-tuned model
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" #"chatbot-ai/versions/fine_tuned_model/checkpoint-200"
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
).to(torch.device("cuda"))

lora_path = "chatbot-ai/versions/v1/checkpoint-400"
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

# Function to generate a response
def chat_with_model(user_input, max_length=150, temperature=0.7, top_p=0.9):
    prompt = f"User: {user_input}\nAI:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=len(inputs["input_ids"][0]) + max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the model's response
    if response.startswith(prompt):
        response = response[len(prompt):]
        
    if "User:" in response:
        response = response.split("User:")[0]
    return response.split("AI:")[-1].strip()

# Example usage
max_history = 10
history = [] 
while True:
    user_text = input("You: ")
    if user_text.lower() in ["quit", "exit"]:
        break
    convo = ""
    
    convo = "".join(history[-max_history:])
    user_input = convo + f"User: {user_text}\n"
    
    reply = chat_with_model(user_input)
    history.append(f"User: {user_text}\nAI: {reply}\n")

    print("AI:", reply)
