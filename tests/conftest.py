import os
import sys
import pytest
from config.config_manager import ConfigManager

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

@pytest.fixture
def config():
    """Provide a ConfigManager instance for tests."""
    return ConfigManager()

@pytest.fixture
def sample_data_path():
    """Provide the path to sample data."""
    return os.path.join(project_root, 'data', 'sample_transactions.csv')

@pytest.fixture
def test_transaction():
    """Provide a sample transaction for testing."""
    return {
        'description': "Uber ride to airport",
        'amount': 45.50,
        'date': "2024-03-15"
    } 