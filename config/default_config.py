"""
Default configuration settings for the transaction classifier.
These can be overridden by environment variables or a local config file.
"""

# Model Configuration
MODEL_CONFIG = {
    'sequence_length': 10,  # Maximum word length of transaction descriptions
    'embedding_dim': 16,    # Dimension of word embeddings
    'lstm_units': 16,       # Number of LSTM units
    'dense_units': 32,      # Number of units in dense layer
    'vocab_size': 5000,    # Maximum vocabulary size
    'batch_size': 16,       # Training batch size
    'epochs': 20,           # Number of training epochs
}

# Data Configuration
DATA_CONFIG = {
    'date_features': [
        'year', 'month', 'day', 'day_of_week', 'hour', 'quarter',
        'is_weekend', 'is_month_end', 'is_month_start',
        'is_quarter_end', 'is_quarter_start', 'time_of_day'
    ],
    'numeric_features': ['amount'],
    'text_feature': 'description',
    'target_feature': 'category'
}

# Training Configuration
TRAINING_CONFIG = {
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy'],
    'validation_split': 0.2,
    'early_stopping_patience': 3
}

# File Paths
PATH_CONFIG = {
    'model_save_path': 'models/{model_type}_classifier.keras',
    'tokenizer_save_path': 'models/{model_type}_tokenizer.pkl',
    'label_encoder_save_path': 'models/{model_type}_label_encoder.pkl',
    'numeric_scaler_path': 'models/{model_type}_numeric_scaler.pkl',
    'date_scaler_path': 'models/{model_type}_date_scaler.pkl',
    'bert_model_path': 'models/{model_type}_model'  # Special case for BERT models
}

# Add new BERT configuration
BERT_CONFIG = {
    'model_type': 'distilbert-base-uncased',
    'max_length': 128,
    'batch_size': 8,
    'epochs': 3,
    'learning_rate': 2e-5,
    'warmup_steps': 500,
    'weight_decay': 0.01
}

# Classifier Configuration
CLASSIFIER_CONFIG = {
    'type': 'hybrid',  # Options: 'hybrid', 'bert', 'lstm', 'ff'
    'hybrid': {
        'enabled': True,
        'description': 'Hybrid classifier combining BERT, numeric, and date features'
    },
    'bert': {
        'enabled': True,
        'description': 'BERT-only classifier for text-based classification'
    },
    'lstm': {
        'enabled': True,
        'description': 'LSTM-based classifier for sequential text processing'
    },
    'ff': {
        'enabled': True,
        'description': 'Feed-forward neural network classifier'
    }
} 