#!/usr/bin/env python
"""
Script to train all transaction classifier models together.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
import pandas as pd
from typing import Dict, Any
import time
from datetime import datetime

from src.bert_classifier import BertClassifier
from src.hybrid_classifier import HybridClassifier
from src.lstm_classifier import LSTMBasedClassifier
from src.ff_classifier import FeedForwardClassifier
from config.config_manager import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles training of all transaction classifier models."""
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize the trainer.
        
        Args:
            models_dir: Directory where models will be saved
        """
        self.config = ConfigManager()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize classifiers
        self.classifiers = {
            'bert': BertClassifier(),
            'hybrid': HybridClassifier(),
            'lstm': LSTMBasedClassifier(),
            'ff': FeedForwardClassifier()
        }
        
        logger.info(f"Initialized trainer with models directory: {models_dir}")
    
    def train_all(self, data_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Train all classifier models.
        
        Args:
            data_path: Path to training data CSV file
            
        Returns:
            Dictionary containing training results for all classifiers
        """
        results = {}
        start_time = time.time()
        
        # Load and validate data
        logger.info(f"Loading training data from {data_path}")
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} training samples")
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            raise
        
        # Train each classifier
        for name, classifier in self.classifiers.items():
            try:
                logger.info(f"\nTraining {name} classifier...")
                classifier_start_time = time.time()
                
                # Train the model
                history = classifier.train(data_path)
                
                # Calculate training time
                training_time = time.time() - classifier_start_time
                
                results[name] = {
                    'status': 'success',
                    'training_time': training_time,
                    'history': history
                }
                
                logger.info(f"Successfully trained {name} classifier in {training_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error training {name} classifier: {str(e)}")
                results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Calculate total training time
        total_time = time.time() - start_time
        
        # Log summary
        logger.info("\nTraining Summary:")
        logger.info("=" * 50)
        logger.info(f"Total training time: {total_time:.2f} seconds")
        for name, result in results.items():
            if result['status'] == 'success':
                logger.info(f"{name}: Success ({result['training_time']:.2f} seconds)")
            else:
                logger.info(f"{name}: Failed - {result['error']}")
        
        return results

def main():
    """Main function to run the training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all transaction classifier models')
    parser.add_argument('data_path', help='Path to training data CSV file')
    parser.add_argument('--models-dir', default='models',
                      help='Directory to save trained models')
    
    args = parser.parse_args()
    
    try:
        # Create trainer and run training
        trainer = ModelTrainer(args.models_dir)
        results = trainer.train_all(args.data_path)
        
        # Print final summary
        print("\nTraining Results:")
        print("=" * 50)
        for name, result in results.items():
            if result['status'] == 'success':
                print(f"\n{name.title()} Classifier:")
                print(f"Status: Success")
                print(f"Training Time: {result['training_time']:.2f} seconds")
            else:
                print(f"\n{name.title()} Classifier:")
                print(f"Status: Failed")
                print(f"Error: {result['error']}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 