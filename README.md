# Biomedical QA with LoRA Fine-tuning on MedQuAD Dataset

## Abstract

This project implements and extends the Low-Rank Adaptation (LoRA) method for efficient fine-tuning of large language models, specifically applied to biomedical question answering using the MedQuAD dataset. The MedQuAD dataset contains 47,457 medical question-answer pairs curated from authoritative NIH sources, covering specialized domains including diseases, drugs, medical procedures, and healthcare topics. We implement LoRA from scratch and demonstrate its effectiveness in adapting pre-trained language models to the complex biomedical domain while maintaining parameter efficiency.

## Project Structure

```
├── README.md
├── requirements.txt
├── gpt-without-LoRA.py
├── LoRA.py
├── implementation.py
├── methodology.md
├── results.md
├── references.md
└── writeup.pdf
├── data/
    └── medquad.csv
```

## Tools & Technologies

### Core Libraries

- **PyTorch**: Deep learning framework for model implementation and training
- **Transformers (Hugging Face)**: Pre-trained models (GPT-2) and tokenization
- **Scikit-learn**: Evaluation metrics, data splitting, and performance analysis
- **NumPy/Pandas**: Data manipulation and numerical computations for medical data processing
- **Datasets**: Hugging Face datasets library for efficient data handling

### Model Components

- **Base Model**: GPT-2 (Hugging Face implementation)
- **LoRA Implementation**: Custom from-scratch implementation with configurable rank
- **Tokenizer**: GPT-2 Tokenizer adapted for medical terminology
- **Optimizer**: AdamW with learning rate scheduling

### Data Handling

- **Dataset**: MedQuAD - Medical Question Answering Dataset (47,457 QA pairs)
- **Source**: National Institutes of Health (NIH) resources
- **Domains**: Diseases, Drugs, Medical Procedures, Healthcare Information
- **Preprocessing**: Custom pipeline for medical QA formatting and text normalization
- **Train-Test Split**: 80-20 splitting maintaining domain distribution

## Evaluation Metrics

### Primary Metrics

1. **Perplexity**: Measures model's confidence in generating medically accurate answers
2. **BLEU Score**: Evaluates n-gram overlap between generated and reference medical answers
3. **ROUGE Score**:
    - ROUGE-1 (unigram overlap for medical terms)
    - ROUGE-2 (bigram overlap for medical phrases)
    - ROUGE-L (longest common subsequence for answer structure)

### Medical Domain Specific Metrics

1. **Exact Match (EM)**: Strict matching of generated vs reference medical answers
2. **F1 Score**: Harmonic mean of precision and recall for medical answer matching
3. **Medical Terminology Accuracy**: Custom metric for domain-specific term correctness
4. **Clinical Relevance Score**: Semantic similarity for medical content preservation

### Implementation Details

- All standard metrics implemented using Hugging Face Evaluate library
- Custom medical terminology validation using biomedical dictionaries and ontologies
- Statistical significance testing for performance comparisons across medical domains
- Domain-wise evaluation (diseases vs drugs vs procedures)

## Dataset Information

### MedQuAD Dataset Specifications

- **Total QA Pairs**: 14,984
- **Sources**: NIH resources including MedlinePlus, Genetics Home Reference, etc.
- **Domain Coverage**:
    - Diseases and Conditions
    - Drugs and Supplements
    - Medical Procedures and Tests
    - Healthcare Topics
    - Genetic Conditions
- **Format**: Question-Answer pairs with structured medical information

### Preprocessing Pipeline

1. Medical text normalization and cleaning
2. Question-Answer pair validation
3. Token length optimization for medical context
4. Domain-based stratification for training split
5. Special token handling for medical entities

## Configuration Options

- `-rank`: LoRA rank parameter (default: 4, 8, 16)
- `-medical_domain`: Specific medical domain focus (diseases, drugs, procedures)
- `-max_length`: Token length for medical context (512-1024)
- `-batch_size`: Adjusted for medical text complexity

## Expected Results

Comprehensive evaluation results comparing base GPT-2 vs LoRA-fine-tuned models on medical QA tasks are documented in `results.md`. The implementation demonstrates significant improvements in medical question answering performance, particularly for domain-specific terminology and clinical accuracy, while maintaining computational efficiency.

For detailed methodology, medical domain adaptation specifics, and extended results analysis, refer to the respective documentation files.
