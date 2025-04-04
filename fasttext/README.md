# FastText Implementation for Fiscozen Chatbot

This directory contains the FastText-based relevance checker implementation for the Fiscozen chatbot. FastText is used to classify user queries as either related to tax matters (IVA or Fiscozen) or off-topic.

## Benefits of FastText over BERT

- **Faster Training**: FastText trains much more quickly than BERT
- **Lower Resource Usage**: Requires less memory and CPU/GPU
- **Simpler Architecture**: More straightforward model with fewer parameters
- **Efficient Inference**: Classification is very fast at runtime

## Directory Structure

- `relevance.py` - Contains the FastTextRelevanceChecker class
- `train_classifier.py` - Script for training the FastText model with document data
- `initialize_model.py` - Script for initializing a basic FastText model
- `models/` - Directory containing the trained FastText model

## Usage

### Checking Query Relevance

```python
from fasttext.relevance import FastTextRelevanceChecker

# Initialize the checker
checker = FastTextRelevanceChecker(model_path="fasttext/models/tax_classifier.bin")

# Check if a query is tax-related
result = checker.check_relevance("How does IVA work in Italy?")
print(f"Is relevant: {result['is_relevant']}")
print(f"Topic: {result['topic']}")
print(f"Confidence: {result['confidence']}")
```

### Training with Custom Data

```python
from fasttext.relevance import FastTextRelevanceChecker

# Initialize the checker
checker = FastTextRelevanceChecker()

# Prepare training data
training_data = [
    ("How does IVA work in Italy?", "IVA"),
    ("What services does Fiscozen offer?", "Fiscozen"),
    ("What is the weather today?", "Other")
]

# Train the model
checker.train_with_data(training_data, epochs=20)

# Save the model
checker.save_model("fasttext/models/custom_model.bin")
```

## Model Format

The FastText model is trained to classify text into three classes:
- `IVA` - Queries related to Italian VAT
- `Fiscozen` - Queries related to Fiscozen services
- `Other` - Off-topic queries

The model returns probabilities for each class, and the relevance checker considers a query relevant if the combined probability of IVA and Fiscozen exceeds a threshold (default: 0.5).

## Performance Comparison vs BERT

| Metric | FastText | BERT |
|--------|----------|------|
| Training time | Minutes | Hours |
| Memory usage | ~100MB | ~500MB+ |
| Inference speed | ~1ms | ~50ms |
| Accuracy | Good | Excellent |

FastText provides a good balance between performance and resource usage, making it ideal for deployment in environments where resources are constrained.

## Credits

This implementation uses the FastText library by Facebook Research:
https://github.com/facebookresearch/fastText 