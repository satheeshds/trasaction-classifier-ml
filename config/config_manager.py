"""
Configuration manager for the transaction classifier.
Handles loading and overriding of configuration settings.
"""

import os
from typing import Any, Dict
from .default_config import (
    MODEL_CONFIG, 
    DATA_CONFIG, 
    TRAINING_CONFIG, 
    PATH_CONFIG,
    CLASSIFIER_CONFIG,
    BERT_CONFIG
)

class ConfigManager:
    def __init__(self):
        self.model_config = MODEL_CONFIG.copy()
        self.data_config = DATA_CONFIG.copy()
        self.training_config = TRAINING_CONFIG.copy()
        self.path_config = PATH_CONFIG.copy()
        self.classifier_config = CLASSIFIER_CONFIG.copy()
        self.bert_config = BERT_CONFIG.copy()
        self._load_environment_overrides()

    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables."""
        # Model config overrides
        if 'TRANSACTION_SEQ_LENGTH' in os.environ:
            self.model_config['sequence_length'] = int(os.environ['TRANSACTION_SEQ_LENGTH'])
        if 'TRANSACTION_BATCH_SIZE' in os.environ:
            self.model_config['batch_size'] = int(os.environ['TRANSACTION_BATCH_SIZE'])
        if 'TRANSACTION_EPOCHS' in os.environ:
            self.model_config['epochs'] = int(os.environ['TRANSACTION_EPOCHS'])
        
        # Classifier config overrides
        if 'TRANSACTION_CLASSIFIER_TYPE' in os.environ:
            self.classifier_config['type'] = os.environ['TRANSACTION_CLASSIFIER_TYPE']

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.model_config

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.data_config

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.training_config

    def get_path_config(self) -> Dict[str, Any]:
        """Get path configuration."""
        return self.path_config

    def get_classifier_config(self) -> Dict[str, Any]:
        """Get classifier configuration."""
        return self.classifier_config

    def get_bert_config(self) -> Dict[str, Any]:
        """Get BERT configuration."""
        return self.bert_config

    def print_config(self):
        """Print current configuration."""
        print("\nCurrent Configuration:")
        print("\nModel Configuration:")
        for key, value in self.model_config.items():
            print(f"  {key}: {value}")
        
        print("\nData Configuration:")
        for key, value in self.data_config.items():
            print(f"  {key}: {value}")
        
        print("\nTraining Configuration:")
        for key, value in self.training_config.items():
            print(f"  {key}: {value}")
        
        print("\nPath Configuration:")
        for key, value in self.path_config.items():
            print(f"  {key}: {value}")
            
        print("\nClassifier Configuration:")
        for key, value in self.classifier_config.items():
            print(f"  {key}: {value}")
            
        print("\nBERT Configuration:")
        for key, value in self.bert_config.items():
            print(f"  {key}: {value}")

    def set_classifier_type(self, classifier_type: str):
        """Override the classifier type."""
        if classifier_type not in ['hybrid', 'bert', 'lstm', 'ff']:
            raise ValueError(f"Invalid classifier type: {classifier_type}")
        self.classifier_config['type'] = classifier_type 