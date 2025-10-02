import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
import math

smooth_fn = SmoothingFunction().method1  # For BLEU smoothing

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return enc.input_ids.squeeze(), enc.attention_mask.squeeze()

def evaluate_metrics(model, tokenizer, references, predictions=None, device="cuda"):
    """
    Evaluate metrics for GPT-2 predictions.
    If predictions is None, model generates them from references.
    """
    model.eval()
    model.to(device)

    if predictions is None:
        predictions = []
        for ref in references:
            inputs = tokenizer(ref, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1]+50,
                    pad_token_id=tokenizer.eos_token_id
                )
            pred_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            predictions.append(pred_text)

    bleu_scores, rouge1_scores, rouge2_scores, rougel_scores, f1_scores = [], [], [], [], []
    ppl_losses = []

    for reference, prediction in zip(references, predictions):
        # BLEU
        ref_tokens = reference.split()
        pred_tokens = prediction.split()
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_fn)
        bleu_scores.append(bleu)

        # ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougel_scores.append(scores['rougeL'].fmeasure)

        # F1 (token overlap)
        common = set(ref_tokens) & set(pred_tokens)
        y_true = [1 if t in common else 0 for t in ref_tokens]
        # pad/truncate pred_tokens
        if len(pred_tokens) < len(ref_tokens):
            pred_tokens += ["<pad>"] * (len(ref_tokens) - len(pred_tokens))
        y_pred = [1 if t in common else 0 for t in pred_tokens[:len(ref_tokens)]]
        if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
            f1_scores.append(f1_score(y_true, y_pred))

        # Perplexity
        enc = tokenizer(reference, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**enc, labels=enc["input_ids"])
            loss = outputs.loss
            ppl_losses.append(torch.exp(loss).item())

    metrics = {
        "BLEU": sum(bleu_scores)/len(bleu_scores),
        "ROUGE1": sum(rouge1_scores)/len(rouge1_scores),
        "ROUGE2": sum(rouge2_scores)/len(rouge2_scores),
        "ROUGEL": sum(rougel_scores)/len(rougel_scores),
        "F1": sum(f1_scores)/len(f1_scores) if f1_scores else 0.0,
        "Perplexity": sum(ppl_losses)/len(ppl_losses)
    }

    return metrics, predictions
