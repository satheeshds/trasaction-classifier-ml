# Hybrid Classifier

## Overview
The Hybrid Classifier combines BERT-based text processing with traditional machine learning techniques to classify transactions. It processes both textual descriptions and numerical features (amount, date) to make predictions.

## Architecture
- BERT component for text processing
- Neural network layers for numerical features
- Combined architecture for final classification
- Custom PyTorch model implementation

## Features
- Multi-modal input processing:
  - Text descriptions (BERT)
  - Transaction amounts
  - Date-based features
- Feature engineering for dates:
  - Year, month, day
  - Day of week
  - Weekend indicators
  - Month/quarter start/end indicators
- Standardized numerical features

## Usage
```python
from src.hybrid_classifier import HybridClassifier

# Initialize classifier
classifier = HybridClassifier()

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
- `date`: Transaction date (YYYY-MM-DD format)
- `category`: Target category for classification

## Model Components
- BERT model for text processing
- StandardScaler for numerical features
- Label Encoder for categories
- Custom PyTorch model with:
  - Text processing layers
  - Numerical feature processing
  - Date feature processing
  - Combined classification head

## Saved Components
- BERT model weights
- Full model state
- Tokenizer
- Label encoder
- Numerical feature scaler
- Date feature scaler

## Performance Considerations
- More complex than pure BERT model
- Requires more memory due to multiple feature types
- Can capture patterns across different data types
- Benefits from GPU acceleration 