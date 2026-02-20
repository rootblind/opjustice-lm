import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
import torch
# 1. Load and preprocess your dataset
with open("chatbot-ai/dataset/conversations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def format_conversations(data):
    max_history=10
    examples = []
    for convo in data:
        history = []
        for turn in convo["turns"]:
            # Build context from previous turns
            prompt_text = ""
            for h in history[-max_history:]:
                prompt_text += f"User: {h['user']}\nAI: {h['response']}\n"
            prompt_text += f"User: {turn['user']}\nAI: {turn['response']}\n"
            examples.append(prompt_text)
            # Add current turn to history
            history.append(turn)
    return examples

texts = format_conversations(data)
dataset = Dataset.from_dict({"text": texts})

# 2. Load tokenizer and model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" #"readerbench/RoGPT2-base" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
).to(torch.device("cuda"))
model = prepare_model_for_kbit_training(model)

# Tokenize
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# 3. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
#model = PeftModel.from_pretrained(model, "chatbot-ai/versions/v1/checkpoint-200")
model.print_trainable_parameters()


# 4. Training arguments
training_args = TrainingArguments(
    output_dir="chatbot-ai/versions/v1",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,  # Keep small for 2050 4GB VRAM
    gradient_accumulation_steps=8,  # Simulate larger batch size
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=2,
    logging_steps=10,
    save_steps=200,
    optim="paged_adamw_8bit"
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

model.gradient_checkpointing_disable()
# 6. Train
trainer.train()

# 7. Save
trainer.save_model("chatbot-ai/versions/fine_tuned_model")
