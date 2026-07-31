"""
train_models.py
===============
End-to-End Training, Hyperparameter Tuning, Benchmarking & Model Persistence Script.
Executes complete dataset EDA, text preprocessing, TF-IDF vectorization, ML model training,
confusion matrix visualizer, model selection, and artifact saving (.pkl).
"""

import os
import time
import joblib
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from utils.preprocess import load_and_detect_dataset, clean_text_pipeline
from utils.visualization import (
    plot_sentiment_distribution,
    plot_review_length_distribution,
    plot_top_ngrams,
    generate_wordclouds,
    plot_confusion_matrix,
    plot_model_comparison
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_training_pipeline():
    logger.info("==================================================")
    logger.info("STARTING NLP SENTIMENT MODEL TRAINING PIPELINE")
    logger.info("==================================================")

    # Directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    # Step 1: Load and inspect dataset
    df, text_col, target_col, meta = load_and_detect_dataset()
    logger.info(f"Loaded dataset shape: {df.shape}")
    logger.info(f"Text Column: '{text_col}', Target Column: '{target_col}'")
    logger.info(f"Class distribution: {meta['class_distribution']}")

    # Step 2: Clean Text
    logger.info("Running text preprocessing pipeline...")
    start_clean = time.time()
    df['clean_text'] = df[text_col].apply(clean_text_pipeline)
    df = df[df['clean_text'].str.strip() != ""].copy()
    logger.info(f"Cleaned {len(df)} non-empty review texts in {time.time() - start_clean:.2f} seconds.")

    # Step 3: EDA & Visualization Generation
    logger.info("Generating EDA charts and saving to images/...")
    plot_sentiment_distribution(df, target_col, save_path="images/sentiment_distribution.png")
    plot_review_length_distribution(df, text_col, target_col, save_path="images/review_length_distribution.png")
    plot_top_ngrams(df, 'clean_text', n_gram_range=(1, 1), top_k=15, title="Top Unigrams in Product Reviews", save_path="images/top_unigrams.png")
    plot_top_ngrams(df, 'clean_text', n_gram_range=(2, 2), top_k=15, title="Top Bigrams in Product Reviews", save_path="images/top_bigrams.png")
    plot_top_ngrams(df, 'clean_text', n_gram_range=(3, 3), top_k=15, title="Top Trigrams in Product Reviews", save_path="images/top_trigrams.png")
    generate_wordclouds(df, 'clean_text', target_col, pos_path="images/positive_wordcloud.png", neg_path="images/negative_wordcloud.png")

    # Step 4: Feature Extraction - TF-IDF Vectorizer
    logger.info("Performing TF-IDF Vectorization with hyperparameter tuning...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(df['clean_text'])
    y = df[target_col].values
    logger.info(f"TF-IDF Feature Matrix Shape: {X.shape}")

    # Train / Test Split (80 / 20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Train Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples")

    # Step 5: Model Training & Evaluation
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, solver="liblinear", random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=2000))
    }

    results = []
    trained_model_objects = {}

    for name, model in models.items():
        logger.info(f"\n--- Training {name} ---")
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        start_pred = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        logger.info(f"{name} Results -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sanitized_name = name.lower().replace(' ', '_')
        plot_confusion_matrix(cm, model_name=name, save_path=f"images/confusion_matrix_{sanitized_name}.png")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Training Time (s)": round(train_time, 4),
            "Prediction Time (s)": round(pred_time, 4)
        })

        trained_model_objects[name] = model

    results_df = pd.DataFrame(results)
    logger.info("\n=== MODEL COMPARISON BENCHMARK TABLE ===")
    print(results_df.to_string(index=False))

    # Save model comparison bar chart
    plot_model_comparison(results_df, save_path="images/model_comparison.png")

    # Step 6: Select Best Model automatically
    best_row = results_df.sort_values(by="F1-Score", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_model = trained_model_objects[best_name]
    logger.info(f"\n🏆 BEST MODEL SELECTED: {best_name} (F1-Score: {best_row['F1-Score']:.4f}, Accuracy: {best_row['Accuracy']:.4f})")

    # Step 7: Save Model & Vectorizer
    model_save_path = "models/best_model.pkl"
    vec_save_path = "models/tfidf_vectorizer.pkl"

    joblib.dump(best_model, model_save_path)
    joblib.dump(vectorizer, vec_save_path)
    logger.info(f"Saved best model to {model_save_path}")
    logger.info(f"Saved vectorizer to {vec_save_path}")
    logger.info("==================================================")
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("==================================================")

if __name__ == "__main__":
    run_training_pipeline()
