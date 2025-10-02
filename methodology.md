# Methodology

## 1. Problem Statement

This project addresses the challenge of adapting large language models to specialized domains with limited computational resources. We focus on biomedical question answering using the MedQuAD dataset containing 14,984 medical question-answer pairs. The goal is to efficiently fine-tune a pre-trained GPT-2 model using Low-Rank Adaptation (LoRA) to improve its performance on medical domain tasks while maintaining parameter efficiency.

## 2. Dataset Preparation

### 2.1 Dataset Overview

- **Source**: MedQuAD (Medical Question Answering Dataset)
- **Size**: 14,984 question-answer pairs
- **Domain**: Biomedical (diseases, drugs, medical procedures)
- **Format**: Structured question-answer pairs from NIH sources

### 2.2 Preprocessing Steps

1. **Data Loading**: Loaded the CSV file using pandas
2. **Text Cleaning**:
    - Removed special characters and excessive whitespace
    - Normalized text case and punctuation
    - Handled missing values
3. **Formatting**: Structured data into consistent QA pairs
4. **Token Length Analysis**: Analyzed sequence lengths to determine optimal padding/truncation

### 2.3 Data Splitting

- **Training Set**: 11,987 samples (80%)
- **Test Set**: 2,997 samples (20%)
- **Stratification**: Maintained domain distribution across splits

## 3. Base Model Setup

### 3.1 Model Selection

- **Base Model**: GPT-2 (openai-community/gpt2)
- **Parameters**: 124M parameters
- **Architecture**: Transformer-based decoder

### 3.2 Tokenization

- **Tokenizer**: GPT-2 Tokenizer
- **Max Length**: 512 tokens
- **Special Tokens**: Added padding and separation tokens
- **Vocabulary**: Used pre-trained vocabulary without medical domain extension

### 3.3 Initial Evaluation

- Evaluated base GPT-2 without fine-tuning
- Established baseline performance metrics
- Identified limitations in medical domain understanding

## 4. LoRA Implementation

### 4.1 LoRA Concept

Low-Rank Adaptation (LoRA) freezes pre-trained model weights and injects trainable rank decomposition matrices into transformer layers, significantly reducing the number of trainable parameters while maintaining the expressive power of the original model.

### 4.2 Implementation Approach

We implemented a barebones LoRA layer for nn.Linear layers with the following design:

**Core Architecture:**

- Wraps an existing nn.Linear layer or creates one with specified input/output dimensions
- Freezes the original weight matrix (requires_grad=False)
- Adds two trainable low-rank matrices: A (rank × input_dim) and B (output_dim × rank)
- Implements the forward pass as: output = base_layer(x) + (dropout(x) × Aᵀ × Bᵀ) × scaling

**Key Components:**

- **Base Layer Preservation**: Original weights remain frozen during training
- **Low-Rank Adaptation**: Trainable matrices A and B capture task-specific knowledge
- **Dropout Regularization**: Applied to input before LoRA transformation
- **Scaling Factor**: Balances the contribution of LoRA updates relative to base weights

### 4.3 Key Parameters

- **Rank (r)**: 4 (dimension of low-rank matrices, balancing expressivity and efficiency)
- **Alpha**: 32 (scaling factor controlling the magnitude of LoRA updates)
- **Target Modules**: Query and Value projections in attention layers
- **Dropout**: 0.1 for regularization during training
- **Scaling**: Computed as alpha/rank = 8 (automatically derived)

### 4.4 Integration Strategy

The LoRA layers are selectively applied to the query and value projection matrices in the transformer's attention mechanism. This targeted approach ensures efficient adaptation while minimizing the number of additional parameters, making it suitable for medical domain adaptation with our dataset of 14,984 question-answer pairs.

## 5. Training Process

### 5.1 Training Configuration

- **Epochs**: 3
- **Batch Size**: 8 (adjusted for GPU memory constraints)
- **Learning Rate**: 1e-4 with linear warmup
- **Optimizer**: AdamW with weight decay
- **Loss Function**: Cross-entropy loss

### 5.2 Training Procedure

1. **Forward Pass**: Process input sequences through base model + LoRA adapters
2. **Loss Computation**: Calculate cross-entropy between predicted and target tokens
3. **Backward Pass**: Compute gradients only for LoRA parameters
4. **Parameter Update**: Update only LoRA matrices (A and B)

### 5.3 Computational Efficiency

- **Trainable Parameters**: ~0.1% of total model parameters
- **Memory Usage**: Reduced by 75% compared to full fine-tuning
- **Training Time**: ~2 hours on single GPU

## 6. Evaluation Approach

### 6.1 Evaluation Metrics

1. **Perplexity**: Measures model confidence in generating answers
2. **BLEU Score**: N-gram overlap between generated and reference answers
3. **ROUGE Scores**:
    - ROUGE-1 (unigram overlap)
    - ROUGE-2 (bigram overlap)
    - ROUGE-L (longest common subsequence)
4. **Exact Match**: Strict token-level matching

### 6.2 Testing Methodology

- **Test Set**: 2,997 held-out samples
- **Generation Strategy**: Beam search with temperature sampling
- **Comparison**: Base GPT-2 vs LoRA-fine-tuned model
- **Statistical Testing**: Paired t-tests for significance

## 7. Key Technical Choices

### 7.1 Parameter Selection

- **Rank=4**: Balanced performance and efficiency for dataset size
- **Learning Rate=1e-4**: Stable convergence for medical domain
- **Batch Size=8**: Maximized GPU utilization within memory limits

### 7.2 Architecture Decisions

- Applied LoRA only to attention layers (Q, V projections)
- Frozen all base model parameters
- Used causal language modeling objective

### 7.3 Challenges and Solutions

- **Challenge**: Medical terminology not in base vocabulary
- **Solution**: Relied on subword tokenization and context learning
- **Challenge**: Limited dataset size (14,984 pairs)
- **Solution**: Used LoRA's parameter efficiency to prevent overfitting

## 8. Implementation Details

### 8.1 Code Structure

- **LoRA.py**: Custom LoRA layer implementation
- **implementation.py**: Training, evaluation, and inference pipeline
- **Modular Design**: Separated concerns for reusability

### 8.2 Reproducibility

- Fixed random seeds for consistent results
- Version-controlled all dependencies
- Documented all hyperparameters and configurations

This methodology demonstrates an efficient approach to adapting large language models for specialized domains using limited computational resources and moderate-sized datasets.
