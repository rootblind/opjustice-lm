import csv
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('readerbench/ro-offense')

encoded = tokenizer('"Do not meddle in the affairs of wizards, for they are subtle and quick to anger')

print(tokenizer.decode(encoded['input_ids']))