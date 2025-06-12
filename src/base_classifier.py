from config.config_manager import ConfigManager

class BaseClassifier:
    """Base class for all classifiers with common functionality."""
    
    def __init__(self, model_type):
        self.config = ConfigManager()
        self.model_type = model_type
        self.model = None
        
    def _get_path(self, path_key):
        """Get the formatted path for a given key."""
        path_template = self.config.get_path_config()[path_key]
        return path_template.format(model_type=self.model_type)
    
    def _save_components(self):
        """Base method for saving components. Override in subclasses."""
        raise NotImplementedError
    
    def load(self):
        """Base method for loading components. Override in subclasses."""
        raise NotImplementedError
    
    def train(self, data_path):
        """Base method for training. Override in subclasses."""
        raise NotImplementedError
    
    def predict(self, *args, **kwargs):
        """Base method for prediction. Override in subclasses."""
        raise NotImplementedError