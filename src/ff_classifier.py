import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.utils import to_categorical
import joblib
from config.config_manager import ConfigManager
from src.classifier_utils import extract_date_features
from src.base_classifier import BaseClassifier

class FeedForwardClassifier(BaseClassifier):
    """Feed-forward neural network classifier."""
    
    def __init__(self):
        super().__init__('ff')
        self.model = None
        self.text_vectorizer = None
        self.numeric_scaler = None
        self.date_scaler = None
        self.label_encoder = None

    def create_model(self, input_dim, num_classes):
        """Create the feed-forward model architecture."""
        # Create input layer
        inputs = Input(shape=(input_dim,))
        
        # Build model using functional API
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(num_classes, activation='softmax')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=self.config.get_training_config()['optimizer'],
            loss=self.config.get_training_config()['loss'],
            metrics=self.config.get_training_config()['metrics']
        )
        return model

    def train(self, data_path):
        """Train the feed-forward model."""
        # Load and preprocess data
        df = pd.read_csv(data_path)
        X, y = self._prepare_features(df)
        
        # Create and train model
        self.model = self.create_model(X.shape[1], len(self.label_encoder.classes_))
        
        # Training callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.get_training_config()['early_stopping_patience'],
                restore_best_weights=True
            ),
            ModelCheckpoint(
                self._get_path('model_save_path'),
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=self.config.get_model_config()['epochs'],
            batch_size=self.config.get_model_config()['batch_size'],
            validation_split=self.config.get_training_config()['validation_split'],
            callbacks=callbacks
        )
        
        # Save components
        self._save_components()
        return history

    def _prepare_features(self, df):
        """Prepare features for the feed-forward model."""
        # Text features
        self.text_vectorizer = CountVectorizer(
            max_features=self.config.get_model_config()['vocab_size'],
            token_pattern=r'(?u)\b\w+\b'
        )
        text_features = self.text_vectorizer.fit_transform(df['description']).toarray()
        
        # Numeric features
        self.numeric_scaler = StandardScaler()
        numeric_features = self.numeric_scaler.fit_transform(df[['amount']])
        
        # Date features
        date_features = extract_date_features(df['date'])
        self.date_scaler = StandardScaler()
        date_features = self.date_scaler.fit_transform(date_features)
        
        # Combine features
        X = np.hstack([text_features, numeric_features, date_features])
        
        # Prepare labels
        self.label_encoder = LabelEncoder()
        y = to_categorical(self.label_encoder.fit_transform(df['category']))
        
        return X, y

    def _save_components(self):
        """Save model components."""
        self.model.save(self._get_path('model_save_path'))
        joblib.dump(self.text_vectorizer, self._get_path('tokenizer_save_path'))
        joblib.dump(self.numeric_scaler, self._get_path('numeric_scaler_path'))
        joblib.dump(self.date_scaler, self._get_path('date_scaler_path'))
        joblib.dump(self.label_encoder, self._get_path('label_encoder_save_path'))

    def load(self):
        """Load saved model and components."""
        self.model = load_model(self._get_path('model_save_path'))
        self.text_vectorizer = joblib.load(self._get_path('tokenizer_save_path'))
        self.numeric_scaler = joblib.load(self._get_path('numeric_scaler_path'))
        self.date_scaler = joblib.load(self._get_path('date_scaler_path'))
        self.label_encoder = joblib.load(self._get_path('label_encoder_save_path'))

    def predict(self, description, amount, date):
        """Make prediction for a single transaction."""
        if self.model is None:
            self.load()
            
        # Prepare features
        text_features = self.text_vectorizer.transform([description]).toarray()
        
        # Ensure amount is in the same format as training data
        amount_df = pd.DataFrame([[amount]], columns=['amount'])
        numeric_features = self.numeric_scaler.transform(amount_df)
        
        # Ensure date features are in the same format as training data
        date_df = pd.DataFrame(extract_date_features([date]))
        date_features = self.date_scaler.transform(date_df)
        
        # Combine features
        X = np.hstack([text_features, numeric_features, date_features])
        
        # Make prediction
        pred = self.model.predict(X, verbose=0)[0]  # Get first prediction
        pred = pred.numpy() if hasattr(pred, 'numpy') else pred  # Convert tensor to numpy if needed
        category = self.label_encoder.inverse_transform([np.argmax(pred)])[0]
        
        # Create probability dictionary for all categories
        probabilities = {
            cat: float(prob)
            for cat, prob in zip(self.label_encoder.classes_, pred)
        }
        
        return category, probabilities