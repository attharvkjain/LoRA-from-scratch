import os
import math
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import torch.optim as optim
from tqdm.auto import tqdm

from eval_utils import evaluate_metrics

# Importing LoRA utilities
from lora import replace_linear_with_lora, lora_parameters, save_lora_state_dict

# -------------------------
# Config
# -------------------------
MODEL_NAME = "gpt2"
RANK = 8                       # LoRA rank r
ALPHA = 32                     # LoRA alpha (scaling)
DROPOUT = 0.05                 # LoRA dropout
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 3e-4
MAX_LENGTH = 256               # max tokens for prompt+answer
SUBSET_SIZE = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "medquad.csv")
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

class QADataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.rows = df.to_dict(orient="records")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        # Build prompt: question <|sep|> answer (we'll use eos token)
        prompt = question.strip() + self.tokenizer.eos_token + answer.strip()
        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        # labels = input_ids (causal LM)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }

def collate_fn(batch, pad_token_id):
    # Pad inputs to longest
    input_ids = [item["input_ids"] for item in batch]
    attn = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    attn = torch.nn.utils.rnn.pad_sequence(
        attn, batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100  # ignore index
    )
    
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels
    }


def main():
    print(f"Device: {DEVICE}")
    
    # Load dataset
    df = pd.read_csv(CSV_PATH)
    print("Dataset rows:", len(df))
    
    if SUBSET_SIZE:
        df = df.head(min(SUBSET_SIZE, len(df)))
        print("Using subset rows:", len(df))

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(DEVICE)

    # Apply LoRA
    print("Replacing Linear layers with LoRA...")
    num_replaced = replace_linear_with_lora(
        model, r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT
    )
    print(f"Replaced {num_replaced} Linear modules with LoRA layers.")

    # Prepare dataloader
    dataset = QADataset(df, tokenizer, max_length=MAX_LENGTH)
    pad_token_id = tokenizer.pad_token_id
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id)
    )

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(list(lora_parameters(model)), lr=LEARNING_RATE)
    total_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps
    )

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        pbar = tqdm(dataloader)
        running_loss = 0.0
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_description(f"loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Save LoRA params after each epoch
        save_path = os.path.join(SAVE_DIR, f"gpt2_lora_epoch{epoch+1}.pt")
        save_lora_state_dict(model, save_path)
        print(f"Saved LoRA state to {save_path}")

    print("Training complete.")

    # Evaluation
    subset_df = df.head(10)
    references = subset_df['answer'].tolist()

    metrics, predictions = evaluate_metrics(
        model, tokenizer, references, device=DEVICE
    )

    print("Evaluation on LoRA fine-tuned GPT-2:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
