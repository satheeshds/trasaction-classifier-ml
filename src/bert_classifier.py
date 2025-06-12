import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from config.config_manager import ConfigManager
from src.base_classifier import BaseClassifier
from datasets import Dataset

class BertClassifier(BaseClassifier):
    """BERT-based transaction classifier."""
    
    def __init__(self):
        super().__init__('bert')
        self.tokenizer = None
        self.label_encoder = None
        
    def train(self, data_path):
        """Train the BERT classifier."""
        try:
            print("Loading data...")
            # Load data
            df = pd.read_csv(data_path)
            print(f"Loaded {len(df)} transactions")
            
            # Prepare text data
            texts = df['description'] + ' ' + df['amount'].astype(str) + ' ' + df['date'].astype(str)
            
            # Encode labels
            print("Encoding labels...")
            self.label_encoder = LabelEncoder()
            labels = self.label_encoder.fit_transform(df['category'])
            print(f"Number of unique categories: {len(self.label_encoder.classes_)}")
            
            # Initialize tokenizer
            print("Initializing tokenizer...")
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
            # Create model
            print("Creating model...")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=len(self.label_encoder.classes_)
            )
            
            # Split data
            print("Splitting data...")
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            
            # Create datasets
            print("Creating datasets...")
            train_dataset = self._create_dataset(train_texts, train_labels)
            val_dataset = self._create_dataset(val_texts, val_labels)
            
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
    
    def _create_dataset(self, texts, labels):
        """Create a dataset for BERT training."""
        try:
            print(f"Tokenizing {len(texts)} texts...")
            # Create a dictionary with the data
            data_dict = {
                'text': texts.tolist(),
                'label': labels.tolist()
            }
            
            # Create HuggingFace dataset
            dataset = Dataset.from_dict(data_dict)
            
            # Tokenize the dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    padding='max_length',
                    truncation=True,
                    max_length=128
                )
            
            # Apply tokenization
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=['text']
            )
            
            print("Dataset created successfully")
            return tokenized_dataset
            
        except Exception as e:
            print(f"Error creating dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_components(self):
        """Save model components."""
        self.model.save_pretrained(self._get_path('bert_model_path'))
        joblib.dump(self.tokenizer, self._get_path('tokenizer_save_path'))
        joblib.dump(self.label_encoder, self._get_path('label_encoder_save_path'))
    
    def load(self):
        """Load saved model and components."""
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self._get_path('bert_model_path')
        )
        self.tokenizer = joblib.load(self._get_path('tokenizer_save_path'))
        self.label_encoder = joblib.load(self._get_path('label_encoder_save_path'))
    
    def predict(self, description, amount, date):
        """Predict category for a single transaction."""
        if self.model is None:
            self.load()
        
        # Combine features into a single text
        text = f"{description} {amount} {date}"
        
        # Prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.get_bert_config()['max_length']
        )
        
        # Make prediction
        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1).detach().numpy()[0]
        
        # Get predicted category
        predicted_idx = np.argmax(probs)
        category = self.label_encoder.inverse_transform([predicted_idx])[0]
        
        # Create probability dictionary
        probabilities = {
            cat: float(prob)
            for cat, prob in zip(self.label_encoder.classes_, probs)
        }
        
        return category, probabilities