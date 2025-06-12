import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
from config.config_manager import ConfigManager
from src.classifier_utils import extract_date_features, create_tokenizer, text_to_sequences
from src.base_classifier import BaseClassifier

class LSTMBasedClassifier(BaseClassifier):
    """LSTM-based transaction classifier."""
    
    def __init__(self):
        super().__init__('lstm')
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        
    def create_model(self, num_classes):
        """Create the LSTM-based model architecture."""
        # Text Branch
        text_input = Input(shape=(self.config.get_model_config()['sequence_length'],))
        embed = Embedding(
            input_dim=self.config.get_model_config()['vocab_size'],
            output_dim=self.config.get_model_config()['embedding_dim']
        )(text_input)
        lstm = LSTM(self.config.get_model_config()['lstm_units'])(embed)
        lstm = Dropout(0.3)(lstm)

        # Amount Branch
        amount_input = Input(shape=(1,))

        # Date Branch
        date_input = Input(shape=(12,))

        # Combine features
        concat = Concatenate()([lstm, amount_input, date_input])
        dense = Dense(
            self.config.get_model_config()['dense_units'],
            activation='relu',
            kernel_regularizer=l2(0.01)
        )(concat)
        dense = Dropout(0.3)(dense)
        output = Dense(num_classes, activation='softmax')(dense)

        model = Model(inputs=[text_input, amount_input, date_input], outputs=output)
        model.compile(
            optimizer=self.config.get_training_config()['optimizer'],
            loss=self.config.get_training_config()['loss'],
            metrics=self.config.get_training_config()['metrics']
        )
        return model

    def train(self, data_path):
        """Train the LSTM-based model."""
        # Load and preprocess data
        df = pd.read_csv(data_path)
        text_data, amount_data, date_data, tokenizer = self._preprocess_data(df)
        
        # Encode labels
        le = LabelEncoder()
        y = to_categorical(le.fit_transform(df['category']))
        
        # Create and train model
        self.model = self.create_model(y.shape[1])
        self.tokenizer = tokenizer
        self.label_encoder = le
        
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
            [text_data, amount_data, date_data],
            y,
            epochs=self.config.get_model_config()['epochs'],
            batch_size=self.config.get_model_config()['batch_size'],
            validation_split=self.config.get_training_config()['validation_split'],
            callbacks=callbacks
        )
        
        # Save components
        self._save_components()
        return history

    def _preprocess_data(self, df):
        """Preprocess the data for LSTM model."""
        # Text preprocessing
        tokenizer = create_tokenizer(
            df['description'],
            self.config.get_model_config()['vocab_size']
        )
        text_data = text_to_sequences(
            df['description'],
            tokenizer,
            self.config.get_model_config()['sequence_length']
        )
        
        # Amount preprocessing
        amount_scaler = StandardScaler()
        amount_data = amount_scaler.fit_transform(df[['amount']])
        
        # Date preprocessing
        date_data = extract_date_features(df['date'])
        
        return text_data, amount_data, date_data, tokenizer

    def _save_components(self):
        """Save model components."""
        self.model.save(self._get_path('model_save_path'))
        joblib.dump(self.tokenizer, self._get_path('tokenizer_save_path'))
        joblib.dump(self.label_encoder, self._get_path('label_encoder_save_path'))

    def load(self):
        """Load saved model and components."""
        self.model = load_model(self._get_path('model_save_path'))
        self.tokenizer = joblib.load(self._get_path('tokenizer_save_path'))
        self.label_encoder = joblib.load(self._get_path('label_encoder_save_path'))

    def predict(self, description, amount, date):
        """Make prediction for a single transaction."""
        if self.model is None:
            self.load()
            
        # Preprocess inputs
        text_seq = text_to_sequences(
            [description],
            self.tokenizer,
            self.config.get_model_config()['sequence_length']
        )
        
        # Ensure amount is in the same format as training data
        amount_df = pd.DataFrame([[amount]], columns=['amount'])
        amount_scaled = StandardScaler().fit_transform(amount_df)
        
        # Ensure date features are in the same format as training data
        date_features = extract_date_features([date])
        date_features = np.array(date_features, dtype=np.float32)
        
        # Make prediction
        pred = self.model.predict([text_seq, amount_scaled, date_features], verbose=0)[0]  # Get first prediction
        pred = pred.numpy() if hasattr(pred, 'numpy') else pred  # Convert tensor to numpy if needed
        category = self.label_encoder.inverse_transform([np.argmax(pred)])[0]
        
        # Create probability dictionary for all categories
        probabilities = {
            cat: float(prob)
            for cat, prob in zip(self.label_encoder.classes_, pred)
        }
        
        return category, probabilities