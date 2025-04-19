# FastText Model for Tax Relevance Classification

This directory contains the FastText model used by the Fiscozen tax chatbot to determine if user queries are related to tax topics.

## Overview

The FastText model is a crucial component of the chatbot that helps filter out non-tax related queries, ensuring the chatbot stays focused on its domain of expertise. The model classifies text into two categories:

- `Tax`: Text related to Italian tax topics, IVA, fiscal matters, etc.
- `Other`: Text unrelated to taxes

## Model Files

- `models/tax_classifier.bin`: The trained FastText model
- `relevance.py`: The Python implementation of the FastText relevance checker
- `trainer.py`: Script for managing model training
- `train_improved_model.py`: Advanced training script with hyperparameter tuning

## Training Data

The model is trained using multiple data sources:

1. **Labeled Conversations**: Structured data from `data_documents/worker_conversations_labeled_translated_file.xlsx`
2. **Argilla PDF Documents**: Tax-related documents from the `argilla_data_49` directory
3. **Other PDF Documents**: Additional documents from the `data_documents` directory
4. **Synthetic Examples**: Generated negative examples to balance the dataset

## Training Process

The improved training process includes:

1. **Data Collection & Preprocessing**:
   - Text extraction from PDFs
   - Cleaning and normalization
   - Label assignment
   - Dataset balancing

2. **Hyperparameter Tuning**:
   - Cross-validation to find optimal parameters
   - Testing various epoch counts, learning rates, n-gram sizes
   - Optimizing for accuracy

3. **Model Evaluation**:
   - Split testing to measure performance
   - Per-category accuracy metrics

## Performance

The current model achieves:
- ~99% accuracy on the test set
- 100% accuracy on tax-related queries
- ~98% accuracy on non-tax queries

## Usage

To use the model for classification:

```python
from fast_text.relevance import FastTextRelevanceChecker

# Initialize the checker (loads model automatically)
checker = FastTextRelevanceChecker()

# Check if text is tax-related
is_relevant, details = checker.is_relevant("Come funziona l'IVA?")
```

## Training/Retraining

To retrain the model:

```bash
# Use the improved training method (recommended)
python run_fasttext_training.py --force

# Test the trained model
python test_fasttext_model.py
```

## Customization

The model can be customized by:

1. Adding more labeled examples to `worker_conversations_labeled_translated_file.xlsx`
2. Adding more tax-related documents to the data directories
3. Adjusting hyperparameters in `train_improved_model.py`
4. Modifying the tax keywords and phrases in `relevance.py` 