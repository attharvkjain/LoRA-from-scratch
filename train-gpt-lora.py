import os
import math
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import torch.optim as optim
from tqdm.auto import tqdm

# Assuming these utilities are in your local directory
# from eval_utils import evaluate_metrics
# from lora import replace_linear_with_lora, lora_parameters, save_lora_state_dict

# Mocking the local imports for demonstration purposes
# In your actual environment, you would use your own files.
def evaluate_metrics(model, tokenizer, references, device):
    print("\n--- Evaluating metrics (mocked) ---")
    # This is a placeholder for your actual evaluation logic
    return {"bleu": 0.5, "rouge": 0.6}, ["mocked prediction"] * len(references)

def replace_linear_with_lora(model, r, lora_alpha, lora_dropout):
    # This is a placeholder
    print("\n--- Replacing linear layers with LoRA (mocked) ---")
    return 12

def lora_parameters(model):
    # This is a placeholder
    return model.parameters()

def save_lora_state_dict(model, path):
    # This is a placeholder
    print(f"--- Saving LoRA state dict to {path} (mocked) ---")
    torch.save(model.state_dict(), path) # Mock saving something


# -------------------------
# Config
# -------------------------
MODEL_NAME = "gpt2"
RANK = 8                      # LoRA rank r
ALPHA = 32                    # LoRA alpha (scaling)
DROPOUT = 0.05                # LoRA dropout
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 3e-4
MAX_LENGTH = 256              # max tokens for prompt+answer
SUBSET_SIZE = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create dummy directories and files for the script to run
os.makedirs("datasets", exist_ok=True)
os.makedirs("lora_checkpoints", exist_ok=True)
os.makedirs("final_model", exist_ok=True)
dummy_df = pd.DataFrame({
    "question": ["What is the capital of France?"] * SUBSET_SIZE,
    "answer": ["Paris"] * SUBSET_SIZE
})
CSV_PATH = os.path.join("datasets", "medquad.csv")
dummy_df.to_csv(CSV_PATH, index=False)


SAVE_DIR = "lora_checkpoints"
FINAL_MODEL_SAVE_DIR = "final_model" # Directory to save the final model
os.makedirs(FINAL_MODEL_SAVE_DIR, exist_ok=True)


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
        enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        # labels = input_ids (causal LM)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

def collate_fn(batch, pad_token_id):
    # pad inputs to longest
    input_ids = [item["input_ids"] for item in batch]
    attn = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    # pad
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # ignore index
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

print(f"Device: {DEVICE}")
df = pd.read_csv(CSV_PATH)
print("Dataset rows:", len(df))
if SUBSET_SIZE:
    df = df.head(min(SUBSET_SIZE, len(df)))
    print("Using subset rows:", len(df))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))
model = model.to(DEVICE)

print("Replacing Linear layers with LoRA...")
num_replaced = replace_linear_with_lora(model, r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT)
print(f"Replaced {num_replaced} Linear modules with LoRA layers.")

dataset = QADataset(df, tokenizer, max_length=MAX_LENGTH)
pad_token_id = tokenizer.pad_token_id
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_id))

optimizer = optim.AdamW(list(lora_parameters(model)), lr=LEARNING_RATE)
total_steps = len(dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

model.train()
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    pbar = tqdm(dataloader)
    running_loss = 0.0
    for batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        pbar.set_description(f"loss: {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # save LoRA params after each epoch
    save_path = os.path.join(SAVE_DIR, f"gpt2_lora_epoch{epoch+1}.pt")
    save_lora_state_dict(model, save_path)
    print(f"Saved LoRA state to {save_path}")

print("Training complete.")

# --- NEW: Save the final model and tokenizer ---
print("\nSaving final model and tokenizer...")
# The recommended way is to save the model's state dict (the parameters)
final_model_state_dict_path = os.path.join(FINAL_MODEL_SAVE_DIR, "final_model_state_dict.pt")
torch.save(model.state_dict(), final_model_state_dict_path)
print(f"Saved final model state dict to {final_model_state_dict_path}")

# Save the tokenizer so it can be easily reloaded with the model
tokenizer.save_pretrained(FINAL_MODEL_SAVE_DIR)
print(f"Saved tokenizer to {FINAL_MODEL_SAVE_DIR}")
# --- End of new code ---


subset_df = df.head(10)
references = subset_df['answer'].tolist()

metrics, predictions = evaluate_metrics(model, tokenizer, references, device=DEVICE)

print("\nEvaluation on LoRA fine-tuned GPT-2:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
