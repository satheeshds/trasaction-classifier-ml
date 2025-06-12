"""
Configuration package for the transaction classifier.
"""

from .config_manager import ConfigManager
from .default_config import MODEL_CONFIG, DATA_CONFIG, TRAINING_CONFIG, PATH_CONFIG

__all__ = ['ConfigManager', 'MODEL_CONFIG', 'DATA_CONFIG', 'TRAINING_CONFIG', 'PATH_CONFIG'] 