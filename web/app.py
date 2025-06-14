import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src import (
    BertClassifier,
    LSTMBasedClassifier,
    FeedForwardClassifier,
    HybridClassifier
)

# Set page config
st.set_page_config(
    page_title="Transaction Classifier",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("Transaction Classifier")
st.markdown("""
This application helps you classify transactions using various machine learning models.
Upload your transaction data and select a model to get started.
""")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox(
    "Choose a model",
    ["BERT", "LSTM", "Feed Forward", "Hybrid"]
)

# Model mapping
MODEL_MAPPING = {
    "BERT": BertClassifier,
    "LSTM": LSTMBasedClassifier,
    "Feed Forward": FeedForwardClassifier,
    "Hybrid": HybridClassifier
}

# File uploader
st.header("Upload Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the data
    try:
        data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        # Model training and prediction section
        st.header("Model Training and Prediction")
        
        if st.button("Train and Predict"):
            with st.spinner("Training model and making predictions..."):
                try:
                    # Initialize the selected model
                    model_class = MODEL_MAPPING[model_type]
                    model = model_class()
                    
                    # Train the model
                    model.train(data)
                    
                    # Make predictions
                    predictions = model.predict(data)
                    
                    # Display results
                    st.subheader("Predictions")
                    results_df = pd.DataFrame({
                        'Transaction': data['description'] if 'description' in data.columns else data.iloc[:, 0],
                        'Predicted Category': predictions
                    })
                    st.dataframe(results_df)
                    
                    # Add some basic statistics
                    st.subheader("Prediction Statistics")
                    category_counts = pd.Series(predictions).value_counts()
                    st.bar_chart(category_counts)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

# Add some helpful information in the sidebar
st.sidebar.markdown("""
### About
This application uses various machine learning models to classify transactions:
- **BERT**: Uses BERT embeddings for text classification
- **LSTM**: Long Short-Term Memory neural network
- **Feed Forward**: Simple neural network with dense layers
- **Hybrid**: Combines multiple approaches for better accuracy
""")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • Transaction Classifier ML") 