import pandas as pd
import numpy as np
from src.classifier_utils import extract_date_features
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset
import joblib
from config.config_manager import ConfigManager
from src.base_classifier import BaseClassifier
from transformers import AutoModelForSequenceClassification

class HybridTransactionDataset(Dataset):
    def __init__(self, texts, numeric_features, date_features, labels, tokenizer, max_length=128):
        self.texts = texts
        self.numeric_features = numeric_features
        self.date_features = date_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        numeric = torch.tensor(self.numeric_features[idx], dtype=torch.float)
        date = torch.tensor(self.date_features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numeric_features': numeric,
            'date_features': date,
            'labels': label
        }

class HybridTransactionModel(nn.Module):
    def __init__(self, num_labels, numeric_dim, date_dim):
        super().__init__()
        # BERT for text
        self.bert = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_labels
        )
        
        # Numeric features processing
        self.numeric_processor = nn.Sequential(
            nn.Linear(numeric_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Date features processing
        self.date_processor = nn.Sequential(
            nn.Linear(date_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_labels + 32 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_labels)
        )

    def forward(self, input_ids, attention_mask, numeric_features, date_features, labels=None):
        # Process text with BERT
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Process numeric features
        numeric_output = self.numeric_processor(numeric_features)
        
        # Process date features
        date_output = self.date_processor(date_features)
        
        # Combine all features
        combined = torch.cat([
            bert_output.logits,
            numeric_output,
            date_output
        ], dim=1)
        
        # Final classification
        logits = self.classifier(combined)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        return logits

class HybridClassifier(BaseClassifier):
    """Hybrid classifier combining BERT and traditional ML approaches."""
    
    def __init__(self):
        """Initialize the hybrid classifier."""
        super().__init__('hybrid')
        self.classifier_type = 'hybrid'
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.numeric_scaler = StandardScaler()
        self.date_scaler = StandardScaler()
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def _save_components(self):
        """Save model components."""
        # Save the BERT part of the model
        self.model.bert.save_pretrained(self._get_path('bert_model_path'))
        # Save the full model state
        torch.save(self.model.state_dict(), self._get_path('model_save_path'))
        joblib.dump(self.tokenizer, self._get_path('tokenizer_save_path'))
        joblib.dump(self.label_encoder, self._get_path('label_encoder_save_path'))
        joblib.dump(self.numeric_scaler, self._get_path('numeric_scaler_path'))
        joblib.dump(self.date_scaler, self._get_path('date_scaler_path'))
    
    def load(self):
        """Load saved model and components."""
        try:
            # First load the label encoder to get number of classes
            self.label_encoder = joblib.load(self._get_path('label_encoder_save_path'))
            num_labels = len(self.label_encoder.classes_)
            
            # Load the BERT part
            bert_model = DistilBertForSequenceClassification.from_pretrained(
                self._get_path('bert_model_path')
            )
            
            # Create the full model
            self.model = HybridTransactionModel(
                num_labels=num_labels,
                numeric_dim=1,
                date_dim=11
            )
            
            # Load the full model state
            self.model.load_state_dict(torch.load(self._get_path('model_save_path')))
            # Set the BERT part
            self.model.bert = bert_model
            
            # Load remaining components
            self.tokenizer = joblib.load(self._get_path('tokenizer_save_path'))
            self.numeric_scaler = joblib.load(self._get_path('numeric_scaler_path'))
            self.date_scaler = joblib.load(self._get_path('date_scaler_path'))
            
            print(f"Successfully loaded {self.classifier_type} classifier")
            
        except Exception as e:
            print(f"Error loading model components: {str(e)}")
            raise

    def prepare_features(self, df):
        """Prepare all features for the model."""
        # Text features
        texts = df['description'].values
        
        # Numeric features
        numeric_features = self.numeric_scaler.fit_transform(df[['amount']])
        
        # Date features
        df['date'] = pd.to_datetime(df['date'])
        date_features = pd.DataFrame({
            'year': df['date'].dt.year,
            'month': df['date'].dt.month,
            'day': df['date'].dt.day,
            'day_of_week': df['date'].dt.dayofweek,
            'hour': df['date'].dt.hour,
            'quarter': df['date'].dt.quarter,
            'is_weekend': (df['date'].dt.dayofweek >= 5).astype(int),
            'is_month_end': df['date'].dt.is_month_end.astype(int),
            'is_month_start': df['date'].dt.is_month_start.astype(int),
            'is_quarter_end': df['date'].dt.is_quarter_end.astype(int),
            'is_quarter_start': df['date'].dt.is_quarter_start.astype(int)
        })
        date_features = self.date_scaler.fit_transform(date_features)
        
        # Create dataset
        dataset = HybridTransactionDataset(
            texts=texts,
            numeric_features=numeric_features,
            date_features=date_features,
            labels=None,  # Labels will be added in train method
            tokenizer=self.tokenizer,
            max_length=self.config.get_bert_config()['max_length']
        )
        
        return dataset

    def train(self, data_path):
        """Train the hybrid classifier."""
        try:
            print("Loading data...")
            df = pd.read_csv(data_path)
            print(f"Loaded {len(df)} transactions")
            
            # Encode labels
            print("Encoding labels...")
            self.label_encoder = LabelEncoder()
            labels = self.label_encoder.fit_transform(df['category'])
            print(f"Number of unique categories: {len(self.label_encoder.classes_)}")
            
            # Create model
            print("Creating model...")
            self.model = HybridTransactionModel(
                num_labels=len(self.label_encoder.classes_),
                numeric_dim=1,
                date_dim=11
            )
            
            # Prepare features
            print("Preparing features...")
            dataset = self.prepare_features(df)
            dataset.labels = labels  # Add labels to dataset
            
            # Split data
            print("Splitting data...")
            train_size = int(0.8 * len(df))
            train_dataset = torch.utils.data.Subset(dataset, range(train_size))
            val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(df)))
            
            # Training arguments
            print("Setting up training arguments...")
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                dataloader_pin_memory=False,
                use_cpu=True,
                report_to="none"
            )
            
            # Initialize trainer
            print("Initializing trainer...")
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset
            )
            
            # Train model
            print("Starting training...")
            trainer.train()
            
            # Save components
            print("Saving model components...")
            self._save_components()
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
    def predict(self, description, amount, date):
        """Make prediction for a single transaction."""
        if self.model is None:
            self.load()
            
        # Prepare features
        # Text features
        inputs = self.tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.get_bert_config()['max_length']
        )
        
        # Numeric features
        amount_df = pd.DataFrame([[amount]], columns=['amount'])
        numeric_features = self.numeric_scaler.transform(amount_df)
        
        # Date features
        date = pd.to_datetime(date)
        date_features = pd.DataFrame({
            'year': [date.year],
            'month': [date.month],
            'day': [date.day],
            'day_of_week': [date.dayofweek],
            'hour': [date.hour],
            'quarter': [date.quarter],
            'is_weekend': [(date.dayofweek >= 5)],
            'is_month_end': [date.is_month_end],
            'is_month_start': [date.is_month_start],
            'is_quarter_end': [date.is_quarter_end],
            'is_quarter_start': [date.is_quarter_start]
        })
        date_features = self.date_scaler.transform(date_features)
        
        # Make prediction
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                numeric_features=torch.tensor(numeric_features, dtype=torch.float32),
                date_features=torch.tensor(date_features, dtype=torch.float32)
            )
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=-1).numpy()[0]
            category = self.label_encoder.inverse_transform([np.argmax(probs)])[0]
            
            # Create probability dictionary
            probabilities = {
                cat: float(prob)
                for cat, prob in zip(self.label_encoder.classes_, probs)
            }
            
            return category, probabilities 