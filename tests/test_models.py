import pytest
import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.bert_classifier import BertClassifier
from src.ff_classifier import FeedForwardClassifier
from src.lstm_classifier import LSTMBasedClassifier
from src.hybrid_classifier import HybridClassifier
from config.config_manager import ConfigManager

@pytest.fixture
def config():
    """Create a ConfigManager instance for testing."""
    return ConfigManager()

@pytest.fixture
def sample_data_path():
    """Return path to sample data file."""
    return "data/sample_transactions.csv"

@pytest.fixture
def test_transaction():
    """Return a sample transaction for testing."""
    return {
        'description': 'Uber ride to airport',
        'amount': 45.50,
        'date': '2024-03-15'
    }

@pytest.fixture(params=[
    (BertClassifier, 'bert'),
    (FeedForwardClassifier, 'ff'),
    (LSTMBasedClassifier, 'lstm'),
    (HybridClassifier, 'hybrid')
])
def model_class(request):
    """Fixture that provides model class and name pairs."""
    return request.param

def test_model(model_class, config, sample_data_path, test_transaction):
    """Test a single model's training and prediction."""
    model_class, model_name = model_class
    print(f"\nTesting {model_name} model...")
    
    # Initialize model
    model = model_class()
    
    # Train model
    print(f"Training {model_name} model...")
    model.train(sample_data_path)
    
    # Make prediction
    print(f"Making prediction with {model_name} model...")
    category, probabilities = model.predict(
        test_transaction['description'],
        test_transaction['amount'],
        test_transaction['date']
    )
    
    # Print results
    print(f"\nPredicted category: {category}")
    print("Probabilities:")
    for cat, prob in probabilities.items():
        print(f"  {cat}: {prob:.4f}")
    
    # Basic assertions
    assert isinstance(category, str)
    assert isinstance(probabilities, dict)
    assert all(0 <= prob <= 1 for prob in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6

def test_all_models(config, sample_data_path, test_transaction):
    """Test all models in sequence."""
    models = [
        (BertClassifier, 'bert'),
        (FeedForwardClassifier, 'ff'),
        (LSTMBasedClassifier, 'lstm'),
        (HybridClassifier, 'hybrid')
    ]
    
    results = {}
    
    for model_class, model_name in models:
        print(f"\nTesting {model_name} model...")
        
        # Initialize model
        model = model_class()
        
        # Train model
        print(f"Training {model_name} model...")
        model.train(sample_data_path)
        
        # Make prediction
        print(f"Making prediction with {model_name} model...")
        category, probabilities = model.predict(
            test_transaction['description'],
            test_transaction['amount'],
            test_transaction['date']
        )
        
        # Store results
        results[model_name] = {
            'category': category,
            'probabilities': probabilities
        }
        
        # Print results
        print(f"\nPredicted category: {category}")
        print("Probabilities:")
        for cat, prob in probabilities.items():
            print(f"  {cat}: {prob:.4f}")
    
    # Print summary
    print("\nSummary of all models:")
    for model_name, result in results.items():
        print(f"\n{model_name.upper()} Model:")
        print(f"Predicted category: {result['category']}")
        print("Top 3 probabilities:")
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        for cat, prob in sorted_probs:
            print(f"  {cat}: {prob:.4f}")
    
    # Basic assertions for all models
    for model_name, result in results.items():
        assert isinstance(result['category'], str)
        assert isinstance(result['probabilities'], dict)
        assert all(0 <= prob <= 1 for prob in result['probabilities'].values())
        assert abs(sum(result['probabilities'].values()) - 1.0) < 1e-6

def main():
    """Run tests for all models."""
    # Sample data path
    sample_data_path = os.path.join(project_root, 'data', 'sample_transactions.csv')
    
    # Test all models
    config = ConfigManager()
    test_transaction = {
        'description': 'Uber ride to airport',
        'amount': 45.50,
        'date': '2024-03-15'
    }
    
    test_all_models(config, sample_data_path, test_transaction)

if __name__ == "__main__":
    main() 