"""
config.py
=========
Centralized Configuration Management for the NLP Sentiment Analysis System.
Eliminates magic constants and hardcoded parameters across data, models, logging,
hyperparameters, and dashboard settings.
"""

import os
from pathlib import Path

# Base Directory of Project
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TESTS_DIR = os.path.join(BASE_DIR, "tests")

# Ensure required directories exist
for directory in [DATA_DIR, MODEL_DIR, IMAGE_DIR, LOG_DIR, RESULTS_DIR, TESTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset Paths
DATASET_PATH = os.path.join(DATA_DIR, "DataSet (W5).csv")

# Saved Model & Vectorizer Paths
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

# Log File Paths
TRAINING_LOG_PATH = os.path.join(LOG_DIR, "training.log")
PREDICTION_LOG_PATH = os.path.join(LOG_DIR, "prediction.log")
APPLICATION_LOG_PATH = os.path.join(LOG_DIR, "application.log")

# Results Export Paths
METRICS_CSV_PATH = os.path.join(RESULTS_DIR, "metrics.csv")
CLASSIFICATION_REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.txt")
BENCHMARK_TABLE_PATH = os.path.join(RESULTS_DIR, "benchmark_table.csv")
PREDICTION_EXAMPLES_PATH = os.path.join(RESULTS_DIR, "prediction_examples.csv")
EVALUATION_REPORT_PATH = os.path.join(RESULTS_DIR, "evaluation_report.md")

# Machine Learning & Pipeline Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# TF-IDF Vectorizer Parameters
TFIDF_PARAMETERS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True
}

# Grid Search Hyperparameter Tuning Grids
GRID_SEARCH_PARAMS = {
    "Logistic Regression": {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["liblinear"]
    },
    "Multinomial Naive Bayes": {
        "alpha": [0.01, 0.1, 0.5, 1.0]
    },
    "Linear SVM": {
        "C": [0.1, 1.0, 5.0],
        "loss": ["hinge", "squared_hinge"]
    }
}

# Streamlit Application Configuration
APP_CONFIGURATION = {
    "page_title": "Sentify | Enterprise NLP Sentiment Analysis",
    "page_icon": "⚡",
    "layout": "wide",
    "batch_max_rows": 10000
}
