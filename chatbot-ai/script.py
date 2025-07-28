from optoolkit import DataProcessor, BaseTransformerModel, DatasetLoader, FineTuning
import torch
from minbpe import BasicTokenizer, RegexTokenizer
import time
import torch.nn as nn

tokenizer = RegexTokenizer()
#tokenizer.load("chatbot-ai/tokenizer/small/tokenizer.model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size=256
n_embd=512
n_head=8
n_layer=4
dropout=0.2
seq_file = "chatbot-ai/dataset/sequence.txt"
batch_size = 64
split = 0.95
learning_rate = 1e-4
eval_interval = 5

version = "chatbot-ai/versions/prototype"

dp = DataProcessor()
import re

with open("chatbot-ai/dataset/sequence.txt", "r", encoding="utf-8") as f:
    file = f.read()
    file = re.sub(r'\s+', " ", file).strip()
    with open("chatbot-ai/dataset/sequence-500k.txt", "w", encoding="utf-8") as g:
        g.write(file[-500_000:])