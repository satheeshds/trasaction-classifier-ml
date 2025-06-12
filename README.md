# 🧐 Transaction Classifier (ML)

This is an open-source project to classify financial transactions into categories like "Food", "Bills", "Transport", etc., using multiple neural network architectures that process text (descriptions), numeric (amount), and temporal (date) features.

> 📦 Inspired by tools like Firefly III, Splitwise, and Actual Budget.

---

## 🚀 Features

- Multiple classifier architectures:
  - BERT-based text classification
  - Hybrid (BERT + Neural Network) for mixed features
  - LSTM for sequence-based processing
  - Feed-forward neural network for fast inference
- Comprehensive evaluation framework
- Preprocessing pipeline for mixed-type features
- Sample data and training scripts
- Notebooks for experimentation
- Easy-to-extend for real-world integrations

---

## 📁 Project Structure

```
transaction-classifier-ml/
├── config/                 # Configuration files
├── data/                   # Sample and user-provided transaction data
├── docs/                   # Documentation for each classifier
├── logs/                   # Training and evaluation logs
├── models/                 # Saved models and components
├── notebooks/              # Jupyter notebooks for EDA & prototyping
├── results/               # Evaluation results and visualizations
├── src/                   # Source code
│   ├── bert_classifier.py    # BERT-based classifier
│   ├── hybrid_classifier.py  # Hybrid BERT+NN classifier
│   ├── lstm_classifier.py    # LSTM-based classifier
│   ├── ff_classifier.py      # Feed-forward classifier
│   ├── base_classifier.py    # Base classifier class
│   ├── classifier_utils.py   # Shared utilities
│   └── evaluate_classifiers.py # Evaluation framework
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── setup.py
└── README.md
```

---

## 📊 Sample Features

| Feature        | Type     | Description                        |
|----------------|----------|------------------------------------|
| `description`  | Text     | Transaction details (e.g. Uber)    |
| `amount`       | Numeric  | Amount of transaction              |
| `date`         | Date     | Date and time of transaction       |
| `category`     | Label    | Target label (food, rent, etc.)    |

---

## 🛠️ Getting Started

```bash
# 1. Clone
git clone https://github.com/yourusername/transaction-classifier-ml.git
cd transaction-classifier-ml

# 2. Install
pip install -r requirements.txt

# 3. Train a specific classifier
python transaction_classifier.py train data/sample_transactions.csv --type bert
python transaction_classifier.py train data/sample_transactions.csv --type hybrid
python transaction_classifier.py train data/sample_transactions.csv --type lstm
python transaction_classifier.py train data/sample_transactions.csv --type ff

# 4. Make predictions
# Single transaction prediction
python transaction_classifier.py predict --type bert --description "Uber ride to airport" --amount 45.50 --date "2024-03-15"
python transaction_classifier.py predict --type hybrid --description "Uber ride to airport" --amount 45.50 --date "2024-03-15"
python transaction_classifier.py predict --type lstm --description "Uber ride to airport" --amount 45.50 --date "2024-03-15"
python transaction_classifier.py predict --type ff --description "Uber ride to airport" --amount 45.50 --date "2024-03-15"

# Batch predictions from CSV
python transaction_classifier.py predict --type hybrid --file data/predictions.csv

# 5. Evaluate all classifiers
python src/evaluate_classifiers.py data/sample_transactions.csv --results-dir results/evaluation
```

---

## 📚 Classifier Documentation

Each classifier has its own documentation in the `docs/` directory:
- [BERT Classifier](docs/bert_classifier.md)
- [Hybrid Classifier](docs/hybrid_classifier.md)
- [LSTM Classifier](docs/lstm_classifier.md)
- [Feed-Forward Classifier](docs/ff_classifier.md)
- [Performance Comparison](docs/performance_comparison.md)

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 🤝 Contributing

Contributions welcome! Check out [CONTRIBUTING.md](CONTRIBUTING.md) soon for guidelines.

---

## 📄 License

MIT License.

