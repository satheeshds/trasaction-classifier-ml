# Feed-Forward Classifier

## Overview
The Feed-Forward Neural Network classifier is a simple yet effective model for transaction classification. It processes numerical features and encoded text features through multiple dense layers to make predictions.

## Architecture
- Multiple dense layers
- ReLU activation functions
- Dropout layers for regularization
- Final softmax layer for classification

## Features
- Simple and fast architecture
- Processes both text and numerical features
- Easy to train and deploy
- Low computational requirements

## Usage
```python
from src.ff_classifier import FeedForwardClassifier

# Initialize classifier
classifier = FeedForwardClassifier()

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
- Text feature encoder
- Numerical feature scaler
- Dense neural network layers
- Dropout layers
- Classification head

## Saved Components
- Model weights
- Feature encoders
- Label encoder
- Feature scalers

## Performance Considerations
- Fastest training and inference
- Lowest memory requirements
- Can run efficiently on CPU
- Good baseline model
- May not capture complex patterns as well as LSTM or BERT 