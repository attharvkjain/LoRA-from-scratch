# Results

## Base GPT-2 Model Evaluation on MedQuAD Dataset

### Overview
This document presents the initial evaluation results of the base GPT-2 model (without any fine-tuning) on the MedQuAD biomedical question-answering dataset. The evaluation was conducted on a subset of 50 samples to establish baseline performance metrics before implementing LoRA fine-tuning.

### Evaluation Metrics Summary

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Average Perplexity** | 14.81 | Moderate confidence in predictions |
| **Average BLEU Score** | 0.0069 | Very low n-gram overlap with reference answers |
| **Average ROUGE-1** | 0.1892 | Low unigram similarity |
| **Average ROUGE-2** | 0.0475 | Very low bigram similarity |
| **Average ROUGE-L** | 0.1414 | Low longest common subsequence similarity |

### Detailed Analysis

#### 1. Perplexity
- **Score**: 14.81
- **Interpretation**: The model shows moderate confidence in its generated responses, but this doesn't necessarily correlate with medical accuracy.

#### 2. BLEU Score
- **Score**: 0.0069 (very low)
- **Interpretation**: Extremely poor n-gram overlap between generated and reference answers, indicating the base model struggles to produce medically accurate content.

#### 3. ROUGE Scores
- **ROUGE-1**: 0.1892 - Low unigram recall, showing poor content overlap
- **ROUGE-2**: 0.0475 - Very poor bigram overlap, indicating weak phrase-level similarity
- **ROUGE-L**: 0.1414 - Poor structural similarity in answers

### Qualitative Observations from Sample Outputs

#### Medical Inaccuracies Identified:
1. **Glaucoma Misclassification**: Model incorrectly describes glaucoma as a skin condition
2. **Repetitive Patterns**: Frequent repetition of phrases and questions
3. **Factual Errors**: Incorrect medical information and associations
4. **Lack of Specificity**: Vague and generic responses instead of precise medical information

#### Example Issues:
- Glaucoma described as affecting "skin" instead of eyes
- High blood pressure explanations containing circular reasoning
- Paget's disease explanations with incorrect genetic causes
- Urinary tract infection responses showing repetitive patterns

### Key Findings

1. **Domain Knowledge Gap**: Base GPT-2 lacks specialized medical knowledge required for accurate biomedical QA
2. **Structural Issues**: Model tends to repeat questions and provides circular explanations
3. **Content Quality**: Generated answers often contain medically inaccurate information
4. **Metric Consistency**: All automatic metrics (BLEU, ROUGE) indicate poor performance

### Conclusion

The base GPT-2 model performs poorly on the MedQuAD biomedical question-answering task, as evidenced by both quantitative metrics and qualitative analysis. The low BLEU and ROUGE scores, combined with medical inaccuracies in generated content, demonstrate the need for domain-specific fine-tuning.

**Next Steps**: Implement LoRA fine-tuning to adapt the model to the biomedical domain while maintaining parameter efficiency. Expected improvements include better medical accuracy, reduced hallucinations, and improved metric scores.

---
*Note: These results serve as the baseline for comparing the performance of our LoRA-fine-tuned model. The evaluation was conducted on October 2024 using GPT-2 base model from Hugging Face Transformers.*
