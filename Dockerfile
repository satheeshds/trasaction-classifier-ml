# Use Python 3.11 as base image (TensorFlow compatible)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and setup files first (for better caching)
COPY requirements.txt setup.py ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the package
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSACTION_SEQ_LENGTH=10

# Command to run the service
CMD ["python", "train_model.py", "--data", "data/transactions.csv"] 