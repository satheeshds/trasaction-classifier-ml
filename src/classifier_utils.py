import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.utils import to_categorical
import joblib
from config.config_manager import ConfigManager

def extract_date_features(dates):
    """Extract date features from a list of date strings."""
    date_features = []
    for date_str in dates:
        date = pd.to_datetime(date_str)
        features = [
            date.year, date.month, date.day, date.dayofweek,
            date.hour, date.quarter,
            int(date.dayofweek >= 5),  # is_weekend
            int(date.is_month_end), int(date.is_month_start),
            int(date.is_quarter_end), int(date.is_quarter_start),
            date.hour  # time_of_day
        ]
        date_features.append(features)
    return np.array(date_features)

def create_tokenizer(texts, vocab_size):
    """Create and fit tokenizer on texts."""
    tokenizer = CountVectorizer(
        max_features=vocab_size,
        token_pattern=r'(?u)\b\w+\b'
    )
    tokenizer.fit(texts)
    return tokenizer

def text_to_sequences(texts, tokenizer, max_len):
    """Convert texts to sequences."""
    sequences = tokenizer.transform(texts).toarray()
    # Pad sequences
    padded_sequences = np.zeros((len(sequences), max_len))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :min(len(seq), max_len)] = seq[:max_len]
    return padded_sequences