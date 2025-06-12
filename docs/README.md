# Transaction Classifier Documentation

This documentation provides detailed information about the different classifier models available in the transaction classification system.

## Available Classifiers

1. [BERT Classifier](bert_classifier.md)
   - Uses DistilBERT for text-based classification
   - Best for complex language understanding
   - Highest computational requirements

2. [Hybrid Classifier](hybrid_classifier.md)
   - Combines BERT with traditional ML
   - Processes text, amount, and date features
   - Balanced performance and complexity

3. [LSTM Classifier](lstm_classifier.md)
   - Uses LSTM for sequence processing
   - Good for capturing temporal patterns
   - Moderate computational requirements

4. [Feed-Forward Classifier](ff_classifier.md)
   - Simple neural network architecture
   - Fastest training and inference
   - Lowest computational requirements

## Choosing a Classifier

Consider the following factors when choosing a classifier:

- **Data Size**: For small datasets, consider Feed-Forward or LSTM
- **Computational Resources**: BERT and Hybrid require more resources
- **Accuracy Requirements**: BERT and Hybrid typically provide better accuracy
- **Speed Requirements**: Feed-Forward is fastest for inference
- **Feature Importance**: 
  - Text-heavy: BERT or Hybrid
  - Numerical-heavy: Feed-Forward
  - Sequential patterns: LSTM

## Common Features

All classifiers support:
- Training from CSV data
- Single transaction prediction
- Batch prediction
- Model saving and loading
- Probability outputs

## Data Format

All classifiers expect training data in CSV format with these columns:
- `description`: Transaction description
- `amount`: Transaction amount
- `date`: Transaction date
- `category`: Target category 