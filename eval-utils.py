import math
import torch
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def calculate_perplexity(model, text):
    encodings = tokenizer(text, return_tensors="pt")
    max_length = model.config.n_positions
    stride = 512
    lls = []

    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len
        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl.item()


def evaluate_model(model, df, subset_size=100):
    """Evaluate model on subset of dataset."""
    bleu_scores, rouge1_scores, rougeL_scores, f1_scores, perplexities = [], [], [], [], []

    for i, row in df.head(subset_size).iterrows():
        question, reference = row["question"], row["answer"]

        # Generate answer
        inputs = tokenizer.encode(question, return_tensors="pt")
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # BLEU
        bleu = sentence_bleu([reference.split()], prediction.split())
        bleu_scores.append(bleu)

        # ROUGE
        rouge = scorer.score(reference, prediction)
        rouge1_scores.append(rouge["rouge1"].fmeasure)
        rougeL_scores.append(rouge["rougeL"].fmeasure)

        # F1 (binary overlap of tokens)
        ref_tokens = reference.split()
        pred_tokens = prediction.split()
        common = set(ref_tokens) & set(pred_tokens)
        y_true = [1 if t in common else 0 for t in ref_tokens]
        y_pred = [1 if t in common else 0 for t in pred_tokens[:len(ref_tokens)]]
        if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
            f1_scores.append(f1_score(y_true, y_pred))

        # Perplexity
        ppl = calculate_perplexity(model, reference)
        perplexities.append(ppl)

    return {
        "BLEU": sum(bleu_scores) / len(bleu_scores),
        "ROUGE-1": sum(rouge1_scores) / len(rouge1_scores),
        "ROUGE-L": sum(rougeL_scores) / len(rougeL_scores),
        "F1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "Perplexity": sum(perplexities) / len(perplexities),
    }
