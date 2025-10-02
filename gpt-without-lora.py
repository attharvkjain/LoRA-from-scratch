import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_utils import evaluate_metrics

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "datasets", "medquad.csv")
df = pd.read_csv(data_path)

subset_size = 10
subset_df = df.head(subset_size)

# Loading GPT-2
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

references = subset_df['answer'].tolist()

metrics, predictions = evaluate_metrics(model, tokenizer, references, device=device)

print("Evaluation on GPT-2 baseline (no finetuning):")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
