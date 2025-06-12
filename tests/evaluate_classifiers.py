#!/usr/bin/env python
"""
Script to evaluate and compare the performance of different transaction classifiers.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import logging
from tqdm import tqdm

from src.bert_classifier import BertClassifier
from src.hybrid_classifier import HybridClassifier
from src.lstm_classifier import LSTMBasedClassifier
from src.ff_classifier import FeedForwardClassifier
from src.classifier_utils import extract_date_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClassifierEvaluator:
    """Evaluates and compares the performance of different transaction classifiers."""
    
    def __init__(self, test_data_path: str, results_dir: str = 'results/evaluation'):
        """
        Initialize the evaluator.
        
        Args:
            test_data_path: Path to the test data CSV file
            results_dir: Directory to save evaluation results
        """
        self.test_data_path = test_data_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize classifiers
        self.classifiers = {
            'bert': BertClassifier(),
            'hybrid': HybridClassifier(),
            'lstm': LSTMBasedClassifier(),
            'ff': FeedForwardClassifier()
        }
        
        logger.info(f"Initialized evaluator with test data: {test_data_path}")
        logger.info(f"Results will be saved to: {results_dir}")
    
    def evaluate_classifier(self, classifier_name: str, classifier: Any) -> Dict[str, Any]:
        """
        Evaluate a single classifier and return metrics.
        
        Args:
            classifier_name: Name of the classifier
            classifier: Classifier instance
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating {classifier_name} classifier...")
        
        try:
            # Load test data
            test_df = pd.read_csv(self.test_data_path)
            logger.info(f"Loaded test data with {len(test_df)} samples")
            
            # Time prediction
            start_time = time.time()
            predictions = []
            probabilities = []
            
            # Make predictions with progress bar
            for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Predicting with {classifier_name}"):
                try:
                    pred, probs = classifier.predict(
                        row['description'],
                        row['amount'],
                        row['date']
                    )
                    predictions.append(pred)
                    probabilities.append(probs)
                except Exception as e:
                    logger.error(f"Error predicting sample: {str(e)}")
                    predictions.append(None)
                    probabilities.append(None)
            
            inference_time = time.time() - start_time
            
            # Filter out failed predictions
            valid_indices = [i for i, p in enumerate(predictions) if p is not None]
            if not valid_indices:
                raise ValueError("No valid predictions were made")
                
            predictions = [predictions[i] for i in valid_indices]
            true_labels = [test_df['category'].iloc[i] for i in valid_indices]
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, 
                predictions, 
                average='weighted'
            )
            
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(true_labels, predictions)
            
            # Calculate average prediction time
            avg_prediction_time = inference_time / len(valid_indices)
            
            # Generate classification report
            class_report = classification_report(
                true_labels, 
                predictions, 
                output_dict=True
            )
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_prediction_time': avg_prediction_time,
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': class_report,
                'categories': test_df['category'].unique().tolist(),
                'total_samples': len(test_df),
                'valid_predictions': len(valid_indices)
            }
            
            logger.info(f"Completed evaluation of {classifier_name} classifier")
            logger.info(f"Accuracy: {accuracy:.3f}, F1 Score: {f1:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {classifier_name} classifier: {str(e)}")
            return {'error': str(e)}
    
    def evaluate_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all classifiers and save results.
        
        Returns:
            Dictionary containing evaluation results for all classifiers
        """
        results = {}
        
        for name, classifier in self.classifiers.items():
            try:
                # Load the trained model
                logger.info(f"Loading {name} classifier...")
                classifier.load()
                
                # Evaluate
                metrics = self.evaluate_classifier(name, classifier)
                results[name] = metrics
                
            except Exception as e:
                logger.error(f"Error evaluating {name} classifier: {str(e)}")
                results[name] = {'error': str(e)}
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Save evaluation results and generate visualizations.
        
        Args:
            results: Dictionary containing evaluation results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw results
        results_file = self.results_dir / f'evaluation_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved raw results to {results_file}")
        
        # Generate comparison table
        self._generate_comparison_table(results, timestamp)
        
        # Generate confusion matrices
        self._generate_confusion_matrices(results, timestamp)
        
        # Generate performance plots
        self._generate_performance_plots(results, timestamp)
    
    def _generate_comparison_table(self, results: Dict[str, Dict[str, Any]], timestamp: str) -> None:
        """Generate a markdown comparison table."""
        table = "# Classifier Performance Comparison\n\n"
        table += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary metrics
        table += "## Summary Metrics\n\n"
        table += "| Classifier | Accuracy | Precision | Recall | F1 Score | Avg Prediction Time (s) | Valid Predictions |\n"
        table += "|------------|----------|-----------|---------|-----------|------------------------|------------------|\n"
        
        for name, metrics in results.items():
            if 'error' not in metrics:
                table += f"| {name} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | "
                table += f"{metrics['recall']:.3f} | {metrics['f1_score']:.3f} | "
                table += f"{metrics['avg_prediction_time']:.3f} | {metrics['valid_predictions']}/{metrics['total_samples']} |\n"
        
        # Add classification reports
        table += "\n## Detailed Classification Reports\n\n"
        for name, metrics in results.items():
            if 'error' not in metrics:
                table += f"### {name.title()} Classifier\n\n"
                table += "```\n"
                table += pd.DataFrame(metrics['classification_report']).to_string()
                table += "\n```\n\n"
        
        # Save table
        table_file = self.results_dir / f'comparison_table_{timestamp}.md'
        with open(table_file, 'w') as f:
            f.write(table)
        logger.info(f"Generated comparison table: {table_file}")
    
    def _generate_confusion_matrices(self, results: Dict[str, Dict[str, Any]], timestamp: str) -> None:
        """Generate confusion matrix plots."""
        for name, metrics in results.items():
            if 'error' not in metrics:
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    metrics['confusion_matrix'],
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=metrics['categories'],
                    yticklabels=metrics['categories']
                )
                plt.title(f'Confusion Matrix - {name.title()} Classifier')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.tight_layout()
                
                # Save plot
                plot_file = self.results_dir / f'confusion_matrix_{name}_{timestamp}.png'
                plt.savefig(plot_file)
                plt.close()
                logger.info(f"Generated confusion matrix for {name}: {plot_file}")
    
    def _generate_performance_plots(self, results: Dict[str, Dict[str, Any]], timestamp: str) -> None:
        """Generate performance comparison plots."""
        # Prepare data for plotting
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        classifiers = [name for name in results.keys() if 'error' not in results[name]]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [results[name][metric] for name in classifiers]
            sns.barplot(x=classifiers, y=values, ax=axes[i])
            axes[i].set_title(f'{metric.title()} Comparison')
            axes[i].set_ylim(0, 1)
            axes[i].set_ylabel(metric.title())
            axes[i].set_xlabel('Classifier')
            
            # Add value labels
            for j, v in enumerate(values):
                axes[i].text(j, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f'performance_comparison_{timestamp}.png'
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Generated performance comparison plot: {plot_file}")

def main():
    """Main function to run the evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate transaction classifiers')
    parser.add_argument('test_data', help='Path to test data CSV file')
    parser.add_argument('--results-dir', default='results/evaluation',
                      help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create evaluator and run evaluation
    evaluator = ClassifierEvaluator(args.test_data, args.results_dir)
    results = evaluator.evaluate_all()
    
    # Print summary
    print("\nEvaluation Summary:")
    print("=" * 50)
    for name, metrics in results.items():
        if 'error' not in metrics:
            print(f"\n{name.title()} Classifier:")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"F1 Score: {metrics['f1_score']:.3f}")
            print(f"Avg Prediction Time: {metrics['avg_prediction_time']:.3f}s")
        else:
            print(f"\n{name.title()} Classifier: Error - {metrics['error']}")

if __name__ == '__main__':
    main() 