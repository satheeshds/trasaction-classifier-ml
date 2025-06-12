# BERT Classifier

## Overview
The BERT (Bidirectional Encoder Representations from Transformers) classifier is a deep learning model that uses the DistilBERT architecture to classify transaction descriptions. It's particularly effective at understanding the semantic meaning of transaction descriptions.

## Architecture
- Uses DistilBERT base uncased model
- Fine-tuned for transaction classification
- Includes a classification head for the specific number of categories

## Features
- Text-based classification using transaction descriptions
- Handles variable-length input text
- Uses attention mechanisms to understand context
- Supports transfer learning from pre-trained BERT model

## Usage
```python
from src.bert_classifier import BertClassifier

# Initialize classifier
classifier = BertClassifier()

# Train the model
classifier.train('path/to/training/data.csv')

# Make predictions
category, probabilities = classifier.predict("GROCERY STORE PURCHASE")
```

## Training Data Format
The training data should be a CSV file with the following columns:
- `description`: Text description of the transaction
- `amount`: Transaction amount
- `category`: Target category for classification

## Model Components
- Tokenizer: DistilBERT tokenizer
- Model: DistilBERT with classification head
- Label Encoder: For converting categories to numeric labels

## Saved Components
- BERT model weights
- Tokenizer
- Label encoder

## Performance Considerations
- Requires significant computational resources
- Benefits from GPU acceleration
- Can handle large vocabularies and complex language patterns 