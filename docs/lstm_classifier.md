# LSTM Classifier

## Overview
The LSTM (Long Short-Term Memory) classifier is a recurrent neural network model designed for transaction classification. It's particularly effective at capturing sequential patterns in transaction descriptions and amounts.

## Architecture
- LSTM layers for sequence processing
- Embedding layer for text input
- Dense layers for final classification
- Bidirectional LSTM for better context understanding

## Features
- Sequence-based processing of transaction data
- Handles variable-length input sequences
- Captures temporal dependencies
- Can process both text and numerical features

## Usage
```python
from src.lstm_classifier import LSTMBasedClassifier

# Initialize classifier
classifier = LSTMBasedClassifier()

# Train the model
classifier.train('path/to/training/data.csv')

# Make predictions
category, probabilities = classifier.predict(
    description="GROCERY STORE PURCHASE",
    amount=50.00,
    date="2024-03-15"
)
```

## Training Data Format
The training data should be a CSV file with the following columns:
- `description`: Text description of the transaction
- `amount`: Transaction amount
- `date`: Transaction date
- `category`: Target category for classification

## Model Components
- Text tokenizer
- Embedding layer
- LSTM layers
- Dense classification layers
- Feature scalers for numerical data

## Saved Components
- Model weights
- Tokenizer
- Label encoder
- Feature scalers

## Performance Considerations
- Faster training than BERT models
- Lower memory requirements
- Good for sequential patterns
- Can be trained on CPU
- May not capture complex language patterns as well as BERT 