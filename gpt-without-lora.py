import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "datasets", "medquad.csv")
df = pd.read_csv(data_path)


# Loading GPT-2
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


# Initializing ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
smooth_fn = SmoothingFunction().method1


# Computing perplexity on a subset of rows
subset_size = min(50, len(df))  # evaluate on first 50 rows or fewer
subset_df = df.head(subset_size)

perplexities = []
for i, row in subset_df.iterrows():
    reference = row['answer']
    enc = tokenizer(reference, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**enc, labels=enc['input_ids'])
        loss = outputs.loss
        ppl = torch.exp(loss).item()
        perplexities.append(ppl)

print(f"Average Perplexity (subset of {subset_size} rows): {sum(perplexities)/len(perplexities):.2f}")


# BLEU, ROUGE, F1 on the same subset
bleu_scores = []
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

for i, row in subset_df.iterrows():
    question = row['question']
    reference = row['answer']
    
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            do_sample=False  # deterministic
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # BLEU
    bleu = sentence_bleu([reference.split()], generated_text.split(), smoothing_function=smooth_fn)
    bleu_scores.append(bleu)
    
    # ROUGE
    rouge = scorer.score(reference, generated_text)
    rouge1_scores.append(rouge['rouge1'].fmeasure)
    rouge2_scores.append(rouge['rouge2'].fmeasure)
    rougeL_scores.append(rouge['rougeL'].fmeasure)
    
    print(f"Q: {question}")
    print(f"GPT-2: {generated_text[:200]}...")
    print(f"Ref: {reference[:200]}...")
    print(f"BLEU: {bleu:.4f}, ROUGE-1: {rouge['rouge1'].fmeasure:.4f}")
    print("-"*80)

# Print average metrics on subset
print(f"Average BLEU (subset): {sum(bleu_scores)/len(bleu_scores):.4f}")
print(f"Average ROUGE-1 (subset): {sum(rouge1_scores)/len(rouge1_scores):.4f}")
print(f"Average ROUGE-2 (subset): {sum(rouge2_scores)/len(rouge2_scores):.4f}")
print(f"Average ROUGE-L (subset): {sum(rougeL_scores)/len(rougeL_scores):.4f}")
