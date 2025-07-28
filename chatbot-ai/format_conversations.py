import json
from transformers import GPT2Tokenizer

# Tokens used in conversation formatting
tokens = {
    "start": "<|start_turn|>",
    "end": "<|end_turn|>",
    "separator": "<|separator|>",
    "eos": "<|endoftext|>"
}

# Initialize the tokenizer (assume it's GPT-2 in this case)
model_name = "readerbench/RoGPT2-base"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"additional_special_tokens": ["<|start_turn|>", "<|end_turn|>", "<|separator|>", "<|endoftext|>"], "pad_token": "<|padding|>"})

# Load the conversation data
with open("chatbot-ai/dataset/conversations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Block size for token limits (1024 tokens in this case)
block_size = 1024

def format_message(message: dict) -> str:
    """Formats each message to include special tokens."""
    return f"{tokens['start']}{message['role']}{tokens['separator']}{message['content']}{tokens['end']}"

def format_dataset():
    text = ""
    current_conversation = []
    current_length = 0
    convo_size = 500

    # Iterate through the conversations
    for conversation in data:  # Limit to the first 100 conversations
        current_conversation = []
        current_length = 0
        for message in conversation["turns"]:
            # Format user and response messages
            user_message = format_message({
                "role": "user",
                "content": message["user"]
            })

            response_message = format_message({
                "role": "response",
                "content": message["response"]
            })

            # Tokenize the current user-response pair
            user_response_tokens = tokenizer.encode(user_message + response_message)

            # Check if the current conversation exceeds the block size
            if current_length + len(user_response_tokens) + 1 >= block_size:
                # If it exceeds, we need to start a new chunk
                if current_conversation:
                    # Add current conversation chunk to the text and start a new one
                    text += ''.join(current_conversation) + tokens['eos'] + "\n"
                    current_conversation = []  # Reset for the next chunk
                    current_length = 0  # Reset token length
                    convo_size -= 1

                if len(user_response_tokens) < block_size:
                    current_conversation.append(user_message + response_message)
                    current_length += len(user_response_tokens)
            else:
                current_conversation.append(user_message + response_message)
                current_length += len(user_response_tokens)

        # Add any remaining conversation after finishing the conversation
        if current_conversation:
            text += ''.join(current_conversation) + tokens['eos'] + "\n"
            convo_size -= 1

        if convo_size < 0:
            break
        

    # Save the formatted dataset to a file
    with open("chatbot-ai/dataset/data_tokenized.txt", "w", encoding="utf-8") as f:
        f.write(text)

# Call the function to format and save the dataset
format_dataset()
