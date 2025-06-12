#!/usr/bin/env python
"""
Unified entry point for transaction classification.
Supports BERT, Hybrid, LSTM, and Feed-Forward NN classifiers based on configuration.
"""

import argparse
import pandas as pd
from src.bert_classifier import BertClassifier
from src.hybrid_classifier import HybridClassifier
from src.lstm_classifier import LSTMBasedClassifier
from src.ff_classifier import FeedForwardClassifier
from config.config_manager import ConfigManager

class TransactionClassifier:
    def __init__(self, classifier_type=None):
        self.config = ConfigManager()
        self.classifier_type = classifier_type or self.config.get_classifier_config()['type']
        self.classifier = self._initialize_classifier()

    def _initialize_classifier(self):
        """Initialize the appropriate classifier based on configuration."""
        if self.classifier_type == 'hybrid':
            return HybridClassifier()
        elif self.classifier_type == 'bert':
            return BertClassifier()
        elif self.classifier_type == 'lstm':
            return LSTMBasedClassifier()
        elif self.classifier_type == 'ff':
            return FeedForwardClassifier()
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def train(self, data_path):
        """Train the selected classifier."""
        print(f"Training {self.classifier_type} classifier...")
        self.classifier.train(data_path)
        print("Training completed successfully!")

    def predict_single(self, description, amount, date):
        """Predict category for a single transaction."""
        print(f"Using {self.classifier_type} classifier for prediction...")
        return self.classifier.predict(description, amount, date)

    def predict_file(self, file_path):
        """Predict categories for transactions in a file."""
        print(f"Using {self.classifier_type} classifier for batch prediction...")
        df = pd.read_csv(file_path)
        results = []
        probabilities = []

        for _, row in df.iterrows():
            category, probs = self.classifier.predict(
                    row['description'],
                    row['amount'],
                    row['date']
                )
            
            results.append(category)
            probabilities.append(probs)

        df['predicted_category'] = results
        return df, probabilities

def main():
    parser = argparse.ArgumentParser(description='Transaction Classifier')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the classifier')
    train_parser.add_argument('data_path', help='Path to the training data CSV file')
    train_parser.add_argument('--type', choices=['hybrid', 'bert', 'lstm', 'ff'], 
                            help='Type of classifier to train (default: from config)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict transaction categories')
    predict_parser.add_argument('--type', choices=['hybrid', 'bert', 'lstm', 'ff'],
                              help='Type of classifier to use (default: from config)')
    
    # Create two groups for predict command
    file_group = predict_parser.add_argument_group('File Input')
    transaction_group = predict_parser.add_argument_group('Single Transaction Input')
    
    file_group.add_argument('--file', type=str, help='Path to CSV file containing transactions')
    transaction_group.add_argument('--description', type=str, help='Transaction description')
    transaction_group.add_argument('--amount', type=float, help='Transaction amount')
    transaction_group.add_argument('--date', type=str, help='Transaction date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        # Create classifier with specified type if provided
        classifier = TransactionClassifier(classifier_type=args.type if hasattr(args, 'type') and args.type else None)
        
        if args.command == 'train':
            classifier.train(args.data_path)
        elif args.command == 'predict':
            if args.file:
                df, probabilities = classifier.predict_file(args.file)
                print("\nPrediction Results:")
                print("==================")
                print(df[['description', 'amount', 'predicted_category']].to_string(index=False))
                
                print("\nConfidence Scores (First Transaction):")
                print("====================================")
                for category, prob in probabilities[0].items():
                    print(f"{category}: {prob:.2%}")
            else:
                if not all([args.description, args.amount, args.date]):
                    predict_parser.error("When not using --file, all of --description, --amount, and --date are required")
                
                category, probabilities = classifier.predict_single(
                    args.description,
                    args.amount,
                    args.date
                )
                
                print("\nPrediction Results:")
                print("==================")
                print(f"Description: {args.description}")
                print(f"Amount: ${args.amount:.2f}")
                print(f"Date: {args.date}")
                print(f"Predicted Category: {category}")
                
                print("\nCategory Probabilities:")
                print("=====================")
                for cat, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                    print(f"{cat}: {prob:.2%}")
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 