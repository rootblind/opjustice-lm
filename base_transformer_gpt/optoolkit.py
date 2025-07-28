# resources used from https://github.com/ImadSaddik/Train_Your_Language_Model_Course/

# ml libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


# auxiliary
import regex as re
import json
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from datetime import timedelta
from tqdm import tqdm

# local
from minbpe import BasicTokenizer, RegexTokenizer
import copy

class DatasetLoader:
    def __init__(self, dataset_file, tokenizer, block_size: int = 256, batch_size: int = 64, device: torch.device = None,
                 split_size: float = 0.9, num_workers: int = 0
                 ):
        self.dataset_file = dataset_file
        self.data_sequence = self.load()

        self.tokenizer = tokenizer
        self.num_workers = num_workers

        self.encoded_sequence = self.encode_sequence()

        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.split_size = split_size

        self.train_data, self.val_data = self.split_train_val()

    def load(self) -> str:
        with open(self.dataset_file, "r") as f:
            text_sequence = f.read()
        return text_sequence
    
    def encode_sequence(self) -> list:
        return self.tokenizer.encode(self.data_sequence)
    
    def split_train_val(self):
        data = torch.tensor(self.encoded_sequence, dtype=torch.long)
        split_index = int(self.split_size * len(data))
        train_data = data[:split_index]
        val_data = data[split_index:]

        return train_data, val_data

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = TextDataset(self.train_data.to(self.device), self.block_size)
        val_dataset = TextDataset(self.val_data.to(self.device), self.block_size)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        return train_loader, val_loader
    
    def test_batches(self):
        train, _ = self.get_dataloaders()

        x, y = next(iter(train))

        print(x.shape, y.shape)

class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size
        if len(self.data) < block_size:
            raise ValueError(f"Data length {len(self.data)} is smaller than block_size {block_size}.")

    def __len__(self) -> int:
        return max(len(self.data) - self.block_size, 0)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index : index + self.block_size]
        y = self.data[index + 1 : index + self.block_size + 1]
        return x, y
    
    

class DataProcessor:
    def __init__(self, patterns = [
            re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            re.compile(r'<[^>]*>'),
            re.compile(r'[^0-9a-zA-Z -]'),
        ]):

        self.tokenizer = BasicTokenizer()

        self.patterns = patterns

    def to_labeled_tokens(self, file_json, result_path):
        with open(file_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        fine_tuning = []
        for convo in data:
            for turn in convo["turns"]:
                user = turn["user"].strip().replace("\n", " ")
                response = turn["response"].strip().replace("\n", " ")
                fine_tuning.append(f"<|startoftext|>user<|separator|>{user}<|endoftext|>")
                fine_tuning.append(f"<|startoftext|>response<|separator|>{response}<|endoftext|>")
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(fine_tuning, f, ensure_ascii=False, indent=2)
        
            

    def to_plain_txt(self, file_json, file_txt):
        with open(file_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(file_txt, "w", encoding="utf-8") as out:
            for convo in data:
                for turn in convo["turns"]:
                    user = turn["user"].strip().replace("\n", " ")
                    bot = turn["response"].strip().replace("\n", " ")
                    out.write(f"{user}\n{bot}\n")

    def to_sequence(self, file_txt, sequence_file):
        filtered_lines = []
        with open(file_txt, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                filtered_line = self.filter_text(line, self.patterns)

                if len(filtered_line) > 4:
                    filtered_lines.append(filtered_line)
        text_sequence = " ".join(filtered_lines)

        with open(sequence_file, "w", encoding='utf-8') as f:
            f.write(text_sequence)

    def concat_sequences(self, file1, file2, result):
        with open(file1, 'r', encoding='utf-8') as f1, \
            open(file2, "r", encoding="utf-8") as f2, \
            open(result, "w", encoding="utf-8") as fout:

            fout.write(f1.read())
            fout.write(" ")
            fout.write(f2.read())

    def generate_tokenizer(self, sequence_file, tokenizer_out, vocab_size=1024):
        with open(sequence_file, "r", encoding="utf-8") as f:
            text_sequence = f.read()

        self.tokenizer.train(text_sequence, vocab_size=vocab_size)

        max_vocab_id = list(self.tokenizer.vocab.keys())[-1]
        self.tokenizer.special_tokens = {
            "<|start_turn|>": max_vocab_id + 1,
            "<|end_turn|>": max_vocab_id + 2,
            "<|separator|>": max_vocab_id + 3,
            "<|endoftext|>": max_vocab_id + 4,
            "<|unk|>": max_vocab_id + 5,
            "<|padding|>": max_vocab_id + 6,
        }

        self.tokenizer.save(file_prefix=f"{tokenizer_out}/tokenizer")

    def filter_text(self, text, patterns=None):
        """
                Filters the text input given by making it lowercase, swapping diactritics to their counterparts and removes
            the patterns.

            - text: the text to be filtered
            - patterns: the patterns to be removed

            - returns: the filtered text
        """
        text = text.lower()
        if len(text) < 3:
            return text
        #text = text.replace('\n', ' ').replace('\r', ' ')
        text = text.replace('ă', 'a')
        text = text.replace('î', 'i')
        text = text.replace('ș', 's')
        text = text.replace('ț', 't')
        text = text.replace('â', 'a')

        if patterns:
            for pattern in patterns:
                text = re.sub(pattern, '', text)
        
        text = text.lstrip()
        return text
    
    def whatsapp_sequence(self, file_path: str, dump_path: str):
        encryption_message = "Mesajele și apelurile sunt criptate integral. Doar persoanele din această conversație le pot citi, asculta sau distribui. Află mai multe."
        media_pattern = "<Media omitted>"
        email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        edited_message = "<Acest mesaj a fost editat>"
        deleted_message = "Acest mesaj a fost șters"
        null_message = "null"
        created_group_message = "created group"
        added_you_to_group_message = "added you"
        tagging_pattern = r'@[\w]+'
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Apply filters to remove unwanted lines
        filtered_lines = []
        for line in lines:
            if (
                encryption_message not in line and
                deleted_message not in line and
                null_message != line.split(" ")[-1] and
                media_pattern not in line and
                created_group_message not in line and
                added_you_to_group_message not in line and
                not re.search(email_pattern, line) and
                not re.search(url_pattern, line)
            ):
                line = line.replace(edited_message, "").strip()
                line = re.sub(tagging_pattern, "", line).strip()
                filtered_lines.append(line)

        messages = []
        for line in filtered_lines:
            match = re.match(r'^\d{2}\.\d{2}\.\d{4}, \d{2}:\d{2}(?: [^\s]+)? - [^:]+: (.+)$', line)
            if match:
                messages.append(self.filter_text(match.group(1), self.patterns))

        print(messages)
        text_sequence = " ".join(messages)

        with open(dump_path, "w", encoding="utf-8") as f:
            f.write(text_sequence)
    
    def whatsapp_json(self, file_path: str, result_path: str):
        encryption_message = "Mesajele și apelurile sunt criptate integral. Doar persoanele din această conversație le pot citi, asculta sau distribui. Află mai multe."
        media_pattern = "<Media omitted>"
        email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        edited_message = "<Acest mesaj a fost editat>"
        deleted_message = "Acest mesaj a fost șters"
        null_message = "null"
        created_group_message = "created group"
        added_you_to_group_message = "added you"
        tagging_pattern = r'@[\w]+'

        filtered_lines = []

        with open(file_path, "r", encoding='utf-8') as f:
            lines = f.readlines()

        
        for line in lines:
            if (
                    encryption_message not in line and
                    deleted_message not in line and
                    null_message not in line and
                    media_pattern not in line and
                    created_group_message not in line and
                    added_you_to_group_message not in line and
                    not re.search(email_pattern, line) and
                    not re.search(url_pattern, line)
            ):
                line = line.replace(edited_message, "").strip()
                line = re.sub(tagging_pattern, "", line).strip()
                filtered_lines.append(line)
        
        messages = []
        for line in filtered_lines:
            messageMatch = re.search(r'^\d{2}\.\d{2}\.\d{4}, \d{2}:\d{2}(?: [^\s]+)? - [^:]+: (.+)$', line)
            
            if messageMatch:
                message = self.filter_text(messageMatch.group(1), self.patterns)
                if len(message) < 2:
                    continue
            else:
                continue

            dateTimeMatch = re.match(r'^(\d{2}\.\d{2}\.\d{4})', line)

            if dateTimeMatch:
                dateTime = self.filter_text(dateTimeMatch.group(1), self.patterns)
            else:
                continue

            senderMatch = re.search(r' - ([^:]+):', line)
            if senderMatch:
                sender = self.filter_text(senderMatch.group(1).strip(), self.patterns)
            else:
                continue

            msgObj = {
                "sender": sender,
                "message": message,
                "datetime": dateTime
            }

            messages.append(msgObj)

        # making sure the first message starts with the user-person
        while True:
            if messages[0]["sender"] == 'polymorph':
                del messages[0]
            else:
                break

        lastMessage = messages[0]
        conversationData = []

        for msg in messages[1:]:
            if lastMessage["sender"] == msg["sender"]:
                lastMessage["message"] += f" {msg["message"]}"
            
            if msg["message"] == messages[-1]["message"] or lastMessage["sender"] != msg["sender"]:
                conversationData.append(lastMessage)
                lastMessage = msg

        paired_data = []

        for i in range(0, len(conversationData), 2):
            if i < len(conversationData) - 1:
                if conversationData[i + 1]["datetime"] == conversationData[i]["datetime"]:
                    # considering that messages within the same date time are part of a single conversation
                    paired_data.append(
                        {
                            "user": conversationData[i]["message"],
                            "response": conversationData[i + 1]["message"],
                            "timestamp": conversationData[i + 1]["datetime"]
                        }
                    )
        
        dataset = []
        lastTimestamp = None

        for pair in paired_data:
            if lastTimestamp == None or pair["timestamp"] != lastTimestamp:
                dataset.append(
                    {
                        "conversation_id": len(dataset),
                        "turns": [pair]
                    }
                )
            elif pair["timestamp"] == lastTimestamp:
                dataset[-1]["turns"].append(pair)

            lastTimestamp = pair["timestamp"]

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

    def concat_json(self, dir: str, result_path: str):
        index = 0
        dataset = []
        with open(result_path, "w", encoding="utf-8") as writer:
            for file in os.listdir(dir):
                if file.endswith('.json') and file not in result_path:
                    file_path = os.path.join(dir, file)
                    print(file)
                    with open(file_path, "r", encoding='utf-8') as f:
                        data = json.load(f)
                        for conversation in data:
                            conversation["conversation_id"] = index
                            index += 1

                        dataset.extend(data)
            json.dump(dataset, writer, indent=2)

        





# layer classes

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        _, T, _ = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = weights @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd: int, dropout: float, num_heads: int, head_size: int, block_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out
    
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, dropout: float, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int) -> None:
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_embd, dropout, n_head, head_size, block_size)
        self.feed_forward = FeedForward(dropout, n_embd)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class BaseTransformerModel(nn.Module):
    def __init__(self, tokenizer, block_size = 256, n_embd = 384, n_head = 6, n_layer = 6,
                 dropout = 0.2, device = None, manual_seed = None, ignore_index = -100):
        
        super().__init__()
        self.tokenizer = tokenizer

        self.vocab_size = self.get_vocab_size()

        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.ignore_index = ignore_index
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if manual_seed:
            torch.manual_seed(manual_seed)

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head, dropout=dropout, block_size=block_size) for _ in range(n_layer)])
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.final_linear_layer = nn.Linear(n_embd, self.vocab_size)

        self.apply(self._init_weights)


    def get_vocab_size(self) -> int:
        vocab = self.tokenizer.vocab
        special_tokens = self.tokenizer.special_tokens

        return len(vocab) + len(special_tokens)
        
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            input_tokens: Tensor of token indices of shape (batch_size, sequence_length)
            targets: Optional tensor of target token indices of same shape as input_tokens

        Returns:
            Tuple of (logits, loss) where logits has shape (batch_size, sequence_length, vocab_size)
            and loss is optional cross-entropy loss if targets are provided
        """

        B, T = input_tokens.shape
        # 1. Check token indices
        if (input_tokens >= self.vocab_size).any():
            invalid = input_tokens >= self.vocab_size
            print(f"Invalid tokens found: {input_tokens[invalid]}")
            print(f"Max allowed index: {self.vocab_size - 1}")
            raise ValueError("Token indices exceed vocabulary size")

        # 2. Check sequence length
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")
        token_embedding = self.token_embedding_table(input_tokens)  # (B,T,C)
        positional_embedding = self.position_embedding_table(
            torch.arange(T, device=self.device))  # (T,C)
        
        assert token_embedding.device == positional_embedding.device, \
        f"Device mismatch: {token_embedding.device} vs {positional_embedding.device}"

        x = token_embedding + positional_embedding  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.final_layer_norm(x)  # (B,T,C)
        logits = self.final_linear_layer(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)

        return logits, loss
    
    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
                Generate new tokens given a context.

                Args:>ns: Starting token indices of shape (batch_size, sequence_length)
                        max_new_tokens: Number of new tokens to generate

                Returns:
                        Tensor of token indices of shape (batch_size, sequence_length + max_new_tokens)
                """

        # input_tokens is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop input_tokens to the last block_size tokens
            cropped_input = input_tokens[:, -self.block_size:]
            # get the predictions
            logits, _ = self(cropped_input)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            input_tokens = torch.cat(
                (input_tokens, idx_next), dim=1)  # (B, T+1)
        return input_tokens
    
    def advanced_generation(
        self,
        input_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generates new tokens from the model.

        Args:
            input_tokens: The initial input tokens.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: Controls randomness (higher = more random).
            top_k: Limits generation to the top-k most likely tokens.
            top_p: Limits generation to tokens with cumulative probability <= top_p.

        Returns:
            The generated tokens.
        """
        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -self.block_size:]
            logits, _ = self(cropped_input)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[...,
                                         1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(logits).scatter_(
                    1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            idx_next = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)

        return input_tokens
    
    def count_parameters(self) -> float:
        """Returns the number of parameters in millions (M)."""
        return sum(p.numel() for p in self.parameters()) / 1e6
    
    def dummy_input(self, batch_size = 1, seq_length = 6) -> dict:
        x = torch.randint(0, self.vocab_size, (batch_size, seq_length))
        x = x.to(self.device)

        logits, loss = self(x)

        return {"logits": logits, "loss": loss}
    
    def print_model_structure(self, model: nn.Module, indent: str = '') -> None:
        """
        Custom function to print model structure in a hierarchical format
        """
        for name, child in model.named_children():
            params = sum(p.numel() for p in child.parameters())
            print(f"{indent}├─ {name}: {child.__class__.__name__} ({params:,} parameters)")
            self.print_model_structure(child, indent + '│  ')


class BaseModelTrainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            file_dir: str,
            eval_iters: int = 200,
            max_iters: int = 1,
            eval_interval: int = 100,
            learning_rate: float = 3e-4,
            show_graph: bool = True,
            gradient_accumulation_steps: int = 1,
            use_scheduler: bool = False
            ) -> Dict[str, float]:
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eval_iters = eval_iters
        self.file_dir = file_dir
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.show_graph = show_graph
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_scheduler = use_scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scaler = torch.GradScaler("cuda")
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_iters) if self.use_scheduler else None

    @torch.no_grad()
    def eval_loss(self):
        output = {}
        self.model.eval()

        for split, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            losses = torch.zeros(self.eval_iters)
            for i, (x, y) in enumerate(loader):
                x = x.to(self.model.device)
                y = y.to(self.model.device)
                if i >= self.eval_iters:
                    break
                with torch.no_grad():
                    _, loss = self.model(x, y)
                losses[i] = loss.item()
            output[split] = losses.mean().item()
        
        self.model.train()

        return output
    
    def save_checkpoint(self, epoch: int, loss: float, file_path: str):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss
        }

        torch.save(checkpoint, file_path)

    def train(self):
        train_losses = []
        val_losses = []
        start_time = time.time()
        print(f"Started training on {len(self.train_loader)} data size over {self.max_iters} iterations")
        for iteration in range(self.max_iters):
            for batch_idx, (x_batch, y_batch) in tqdm(
                iterable=enumerate(self.train_loader),
                desc="Training on batches",
                total=len(self.train_loader)
                ):

                x_batch = x_batch.to(self.model.device)
                y_batch = y_batch.to(self.model.device)

                # calculate progress
                total_steps = self.max_iters * len(self.train_loader)
                current_step = iteration * len(self.train_loader) + batch_idx
                progress_percent = (current_step / total_steps) * 100
                # evaluation
                if batch_idx % self.eval_interval == 0 or batch_idx == len(self.train_loader) - 1:
                    losses = self.eval_loss()
                    train_losses.append(losses["train"])
                    val_losses.append(losses["val"])

                    elapsed_time = time.time() - start_time
                    estimated_total_time = elapsed_time / (current_step + 1) * total_steps
                    eta_seconds = estimated_total_time - elapsed_time
                    eta_formatted = str(timedelta(seconds=int(eta_seconds)))
                    elapsed_formatted = str(timedelta(seconds=int(elapsed_time)))
                    
                    print(
                        f"\niteration {iteration} / step {batch_idx}: "
                        f"({progress_percent:.2f}% done, Elapsed: {elapsed_formatted}, ETA: {eta_formatted}): "
                        f"train loss {losses['train']:.4f}, "
                        f"val loss {losses['val']:.4f}"
                    )
                
                # training step
                with torch.autocast("cuda"):
                    logits, loss = self.model(x_batch, y_batch)

                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    

            if self.scheduler:
                self.scheduler.step()
            
            # save checkpoint
            self.save_checkpoint(epoch = iteration,
                                 loss = loss.item(),
                                 file_path = f"{self.file_dir}/checkpoint_{iteration}.pth"
                                 )
            
        if self.show_graph:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label="Train Loss", marker='o')
            plt.plot(val_losses, label="Validation Loss", marker='o')
            plt.xlabel("Evaluation Step")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss Over Time")
            plt.legend()
            plt.grid()
            plt.show()

class FineTuning:
    def __init__(self, data_file: str, tokenizer: RegexTokenizer, model: BaseTransformerModel, model_path: str,
                 file_dir: str, split: float = 0.95, block_size: int = 256,
                 batch_size: int = 64, device: torch.device = None, n_embd: int = 512,
                 n_head: int = 8, n_layer: int = 4, dropout: float = 0.2, learning_rate: float = 1e-4, max_iters: int = 1,
                 eval_interval: int = 5, use_lora: bool = False, rank: int = 4, alpha: int = 8,
                 gradient_accumulation_steps: int = 1, use_scheduler: bool = False
                 ):
        
        self.model = model
        self.model_path = model_path
        self.file_dir = file_dir
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.eval_interval = eval_interval
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.padding_token = self.tokenizer.special_tokens["<|padding|>"]
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_tokens = 0
        self.split = split
        self.tokens = {
            "start": "<|start_turn|>",
            "end": "<|end_turn|>",
            "separator": "<|separator|>",
            "eos": "<|endoftext|>"
        }
        self.eos_id = self.tokenizer.encode(self.tokens["eos"], allowed_special="all")[0]

        #lora specific
        self.rank = rank
        self.alpha = alpha
        self.use_lora = use_lora

        self.data = self.load_json(data_file)
        self.vocab_size = self.get_vocab_size()

        #self.check_block_size()
        self.format_dataset()
        self.train_loader, self.val_loader = self.get_dataloaders()

        self.load_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scaler = torch.GradScaler("cuda")
        self.use_scheduler = use_scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_iters) if self.use_scheduler else None
        self.gradient_accumulation_steps = gradient_accumulation_steps
        

        
    def load_json(self, file) -> list:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data
    
    def load_model(self):
        if self.use_lora:
            self.add_lora_layers()
        checkpoint = torch.load(self.model_path, weights_only=True, map_location=self.device)
        model_state_dict = checkpoint["model_state_dict"]
        self.model.load_state_dict(model_state_dict)

    def check_block_size(self):
        for conversation in self.data:
            concatenated_messages = ""
            for message in conversation["turns"]:
                user_content = message["user"]
                response_content = message["response"]
                concatenated_messages += user_content + "\n" + response_content + "\n"

            tokens = self.tokenizer.encode(concatenated_messages)
            self.max_tokens = max(self.max_tokens, len(tokens))
            if len(tokens) > self.block_size:
                raise Exception(
                    f"Error: Token length exceeds block size. Length: {len(tokens)}, Block size: {self.block_size}")

    def format_message(self, message: dict) -> str:
        return f"{self.tokens['start']}{message['role']}{self.tokens['separator']}{message['content']}{self.tokens['end']}"
    
    def format_dataset(self):
        fine_tuning_data = []
        for conversation in self.data:
            concatenated_messages = ""
            for message in conversation["turns"]:
                user_message = self.format_message({
                    "role": "user",
                    "content": message["user"]
                })

                response_message = self.format_message({
                    "role": "response",
                    "content": message["response"]
                })

                current_tokens = self.tokenizer.encode(
                    text=concatenated_messages,
                    allowed_special="all"
                )

                user_response_tokens = self.tokenizer.encode(
                    text=user_message + response_message,
                    allowed_special="all"
                )

                if len(current_tokens) + len(user_response_tokens) >= self.block_size:
                    if current_tokens:
                        if len(current_tokens) < self.block_size:
                            current_tokens.append(self.eos_id)
                            fine_tuning_data.append(current_tokens)
                    
                    concatenated_messages = ""

                    if user_response_tokens:
                        if len(user_response_tokens) < self.block_size:
                            concatenated_messages += user_message + response_message
                else:
                    concatenated_messages += user_message + response_message
                
                
            if concatenated_messages:
                final_tokens = self.tokenizer.encode(
                    text=concatenated_messages + self.tokens["eos"],
                    allowed_special="all"
                )
                if final_tokens:
                    if len(final_tokens) < self.block_size:
                        fine_tuning_data.append(final_tokens)
                        

        
        self.train_data_tensor = self.apply_padding_to_data(
            data=fine_tuning_data,
            max_seq_length=self.block_size
        )

        self.split_index = int(self.split * len(self.train_data_tensor))

        self.train_data_split = self.train_data_tensor[:self.split_index]
        self.val_data_split = self.train_data_tensor[self.split_index:]

    def get_vocab_size(self) -> int:
        vocab = self.tokenizer.vocab
        special_tokens = self.tokenizer.special_tokens

        return len(vocab) + len(special_tokens)
    
    def apply_padding_to_data(self, data: list[list[int]], max_seq_length: int) -> torch.Tensor:
        tensors = []
        for i in range(len(data)):
            tensor = torch.tensor(data[i])
            padded_tensor = torch.nn.functional.pad(
                input=tensor,
                # for right padding:
                pad=(0, max_seq_length - len(tensor)),
                # pad=(block_size - len(tensor), 0),
                value=self.padding_token
            )
            tensors.append(padded_tensor)

        return torch.stack(tensors)
    
    def get_dataloaders(self):
        train = FineTuningDataset(
            data=self.train_data_split,
            padding_token=self.padding_token,
            device=self.device,
            tokenizer=self.tokenizer
        )

        val = FineTuningDataset(
            data=self.val_data_split,
            padding_token=self.padding_token,
            device=self.device,
            tokenizer=self.tokenizer
        )

        train_loader = DataLoader(
            dataset=train,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            dataset=val,
            batch_size=self.batch_size,
            shuffle=True
        )

        sample_batch = next(iter(train_loader))
        print(f"Sample batch is on: {sample_batch[0].device}, {sample_batch[1].device}")
        return train_loader, val_loader
    
    def add_lora_layers(self):
        self.model = get_lora_model(
            model=self.model,
            lora_config={
                "rank": self.rank,
                "alpha": self.alpha
            },
            device=self.device
        )
    
    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        output = {}
        self.model.eval()

        for split, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            losses = torch.zeros(self.eval_interval)
            for i, (x, y) in enumerate(loader):
                x = x.to(self.model.device)
                y = y.to(self.model.device)
                if i >= self.eval_interval:
                    break
                with torch.no_grad():
                    _, loss = self.model(x, y)
                losses[i] = loss.item()
            output[split] = losses.mean().item()
        
        self.model.train()

        return output
    
    def save_checkpoint(self, epoch: int, loss: float, file_path: str):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, file_path)

    def train(self, show_graph: bool = True):
        train_losses = []
        val_losses = []
        start_time = time.time()

        print(f"Started training on {len(self.train_loader)} data size over {self.max_iters} iterations")

        for iteration in range(self.max_iters):
            for batch_idx, (x_batch, y_batch) in tqdm(
                iterable=enumerate(self.train_loader),
                desc="Training on batches",
                total=len(self.train_loader)
            ):
                
                x_batch = x_batch.to(self.model.device)
                y_batch = y_batch.to(self.model.device)
                # calculate progress
                total_steps = self.max_iters * len(self.train_loader)
                current_step = iteration * len(self.train_loader) + batch_idx
                progress_percent = (current_step / total_steps) * 100
                # Evaluation
                if batch_idx % self.eval_interval == 0 or batch_idx == len(self.train_loader) - 1:
                    losses = self.estimate_loss()
                    train_losses.append(losses['train'])
                    val_losses.append(losses['val'])

                    elapsed_time = time.time() - start_time
                    estimated_total_time = elapsed_time / (current_step + 1) * total_steps
                    eta_seconds = estimated_total_time - elapsed_time
                    eta_formatted = str(timedelta(seconds=int(eta_seconds)))
                    elapsed_formatted = str(timedelta(seconds=int(elapsed_time)))

                    print(
                        f"iteration {iteration} / step {batch_idx}: "
                        f"({progress_percent:.2f}% done, Elapsed: {elapsed_formatted}, ETA: {eta_formatted}): "
                        f"train loss {losses['train']:.4f}, "
                        f"val loss {losses['val']:.4f}"
                    )

                # Training step
                with torch.autocast("cuda"):
                    logits, loss = self.model(x_batch, y_batch)

                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
            if self.scheduler:
                self.scheduler.step()

            # Save checkpoint
            self.save_checkpoint(
                epoch=iteration,
                loss=loss.item(),
                file_path=f"{self.file_dir}/checkpoint_{iteration}.pth"
            )

        if show_graph:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label="Train Loss", marker='o')
            plt.plot(val_losses, label="Validation Loss", marker='o')
            plt.xlabel("Evaluation Step")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss Over Time")
            plt.legend()
            plt.grid()
            plt.show()

    
class FineTuningDataset(Dataset):
    def __init__(self, data: torch.Tensor, device: torch.device, padding_token: int, tokenizer: RegexTokenizer):
        self.data = data  # shape: (num_samples, block_size)
        self.device = device
        self.padding_token = padding_token
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[index]
        x = sample.to(self.device)
        y = sample[1:].to(self.device)
        padding_tensor = torch.tensor([self.padding_token], device=self.device)
        y = torch.cat((y, padding_tensor))
        masked_y = self.apply_mask_to_target(y)
        return x, masked_y
    
    def apply_mask_to_target(self, y: torch.Tensor) -> torch.Tensor:
        response_turn_tokens = torch.tensor(
            self.tokenizer.encode(
                "<|start_turn|>response<|separator|>",
                allowed_special="all"
            ),
            device=self.device
        )
        sublist_length = len(response_turn_tokens)

        # Find the last occurrence of response_turn_tokens in y
        # This only works if you use right padding
        last_occurrence = -1
        for i in range(len(y) - sublist_length + 1):
            if torch.all(y[i:i+sublist_length] == response_turn_tokens):
                last_occurrence = i + sublist_length - 1

        if last_occurrence != -1:
            y[:last_occurrence + 1] = self.padding_token

        return y



class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        std_dev = 1/torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank)*std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.alpha*(x@self.A@self.B)
        return x


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def print_trainable_parameters(model: BaseTransformerModel) -> None:
    trainable_parameters = 0
    all_parameters = 0
    for _, param in model.named_parameters():
        all_parameters += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()

    print(
        f"All parameters: {all_parameters/1e6:.2f}M | "
        f"Trainable parameters: {trainable_parameters/1e6:.2f}M | "
        f"Trainable %: {100 * trainable_parameters / all_parameters:.2f}%"
    )


def get_lora_model(model: BaseTransformerModel, lora_config: dict, device: str) -> BaseTransformerModel:
    lora_model = copy.deepcopy(model)
    _replace_linear_layers_with_lora_layers(lora_model, lora_config)
    _freeze_non_lora_layers(lora_model)
    lora_model = lora_model.to(device)
    return lora_model


def _replace_linear_layers_with_lora_layers(module: nn.Module, lora_config: dict) -> None:
    rank = lora_config.get('rank', 4)
    alpha = lora_config.get('alpha', 8)

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LinearWithLoRA(
                child, rank=rank, alpha=alpha))
        else:
            _replace_linear_layers_with_lora_layers(
                child, lora_config)


def _freeze_non_lora_layers(model: BaseTransformerModel) -> None:
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
