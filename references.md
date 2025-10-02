# References

## Primary Resources

### Code Repository

- **GitHub Repository**: [LoRA-from-scratch](https://github.com/attharvkjain/LoRA-from-scratch)
> The main codebase containing our implementation of LoRA from scratch, training scripts, and evaluation code for biomedical question answering.

### Base Model

- **GPT-2 Model**: [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
> Pre-trained GPT-2 model from Hugging Face used as our base language model for LoRA fine-tuning on medical question answering tasks.

### Dataset

- **MedQuAD Dataset**: [Medical Question Answer for AI Research](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research)
> The primary dataset containing 14,984 medical question-answer pairs from authoritative NIH sources, covering diseases, drugs, medical procedures, and healthcare topics.

### Research Paper

- **LoRA Original Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
> The foundational research paper by Hu et al. (2021) that introduced the Low-Rank Adaptation method for efficient fine-tuning of large language models.

## Supporting Libraries and Frameworks

### Machine Learning Frameworks

- **PyTorch**: https://pytorch.org/
    
    Deep learning framework used for model implementation, training, and inference.
    
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers
    
    Library providing pre-trained models, tokenizers, and training utilities for natural language processing.
    

### Evaluation Metrics

- **Hugging Face Evaluate**: https://huggingface.co/docs/evaluate
    
    Library used for implementing and computing evaluation metrics including BLEU, ROUGE, and perplexity.
    
- **Scikit-learn**: https://scikit-learn.org/
    
    Machine learning library used for data splitting, statistical analysis, and additional evaluation metrics.
    

### Data Processing

- **Pandas**: https://pandas.pydata.org/
    
    Data manipulation library used for preprocessing and analyzing the MedQuAD dataset.
    
- **NumPy**: https://numpy.org/
    
    Numerical computing library used for mathematical operations and array manipulations.
    

## Dataset Sources

### Original NIH Sources

The MedQuAD dataset is compiled from multiple authoritative NIH resources including:

- **MedlinePlus**: Comprehensive medical information resource
- **Genetics Home Reference**: Consumer-friendly information about genetic conditions
- **NIH Health Topics**: Authoritative health information
- **Other NIH databases**: Various specialized medical databases

## Related Research

### Large Language Model Fine-tuning

- **Parameter-Efficient Fine-tuning Methods**: Various approaches for adapting large language models with minimal parameter updates
- **Medical Domain Adaptation**: Research on adapting language models for healthcare and biomedical applications

### Biomedical NLP

- **Clinical Language Models**: Research on language models specialized for medical and clinical text
- **Medical Question Answering**: Approaches and benchmarks for healthcare information retrieval and question answering

## Implementation References

### LoRA Implementation

- Original LoRA implementation insights and mathematical foundations from the research paper
- Adaptation techniques for medical domain specialization
- Parameter efficiency considerations for resource-constrained environments

### Model Evaluation

- Standard NLP evaluation metrics adapted for medical domain specificity
- Clinical relevance assessment methodologies
- Medical terminology validation approaches
