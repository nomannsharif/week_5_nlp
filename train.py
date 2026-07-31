"""
train.py
========
End-to-End Training, Hyperparameter Grid Search, Stratified Cross-Validation & Model Versioning.
Complies with Step 17 Production Engineering Standards.
Outputs best model artifacts and models/model_info.json metadata.
"""

import os
import time
import json
import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from config import (
    BEST_MODEL_PATH, TFIDF_VECTORIZER_PATH, MODEL_INFO_PATH,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, TFIDF_PARAMETERS, GRID_SEARCH_PARAMS
)
from utils.logger import training_logger
from utils.preprocess import load_and_detect_dataset, clean_text_pipeline
from utils.visualization import (
    plot_sentiment_distribution, plot_review_length_distribution,
    plot_top_ngrams, generate_wordclouds, plot_confusion_matrix, plot_model_comparison
)

def run_pipeline():
    training_logger.info("==================================================")
    training_logger.info("STARTING E2E TRAINING & GRID SEARCH PIPELINE")
    training_logger.info("==================================================")

    # 1. Load Dataset & Audit
    df, text_col, target_col, meta = load_and_detect_dataset()
    training_logger.info(f"Dataset Shape: {df.shape} | Text Col: {text_col} | Target Col: {target_col}")

    # 2. Text Preprocessing
    t0 = time.time()
    df['clean_text'] = df[text_col].apply(clean_text_pipeline)
    df = df[df['clean_text'].str.strip() != ""].copy()
    clean_duration = time.time() - t0
    training_logger.info(f"Cleaned {len(df)} non-empty reviews in {clean_duration:.2f} seconds.")

    # 3. Generate & Save EDA Visualizations
    training_logger.info("Generating EDA figures...")
    plot_sentiment_distribution(df, target_col, save_path="images/sentiment_distribution.png")
    plot_review_length_distribution(df, text_col, target_col, save_path="images/review_length_distribution.png")
    plot_top_ngrams(df, 'clean_text', n_gram_range=(1, 1), top_k=15, title="Top Unigrams", save_path="images/top_unigrams.png")
    plot_top_ngrams(df, 'clean_text', n_gram_range=(2, 2), top_k=15, title="Top Bigrams", save_path="images/top_bigrams.png")
    plot_top_ngrams(df, 'clean_text', n_gram_range=(3, 3), top_k=15, title="Top Trigrams", save_path="images/top_trigrams.png")
    generate_wordclouds(df, 'clean_text', target_col, pos_path="images/positive_wordcloud.png", neg_path="images/negative_wordcloud.png")

    # 4. Feature Extraction - TF-IDF
    training_logger.info(f"Fitting TF-IDF Vectorizer with params: {TFIDF_PARAMETERS}")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMETERS)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df[target_col].values

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    training_logger.info(f"Train Set: {X_train.shape[0]} | Test Set: {X_test.shape[0]}")

    # 5. Stratified K-Fold Cross Validation & GridSearch
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    candidate_models = {
        "Logistic Regression": (LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), GRID_SEARCH_PARAMS["Logistic Regression"]),
        "Multinomial Naive Bayes": (MultinomialNB(), GRID_SEARCH_PARAMS["Multinomial Naive Bayes"]),
        "Linear SVM": (LinearSVC(random_state=RANDOM_STATE, max_iter=2000), GRID_SEARCH_PARAMS["Linear SVM"])
    }

    benchmark_results = []
    trained_estimators = {}

    for name, (base_model, param_grid) in candidate_models.items():
        training_logger.info(f"\n--- Running Stratified CV & GridSearch for {name} ---")
        grid = GridSearchCV(base_model, param_grid, cv=skf, scoring='f1', n_jobs=-1)
        
        t_start = time.time()
        grid.fit(X_train, y_train)
        train_duration = time.time() - t_start

        best_estimator = grid.best_estimator_
        best_params = grid.best_params_
        training_logger.info(f"{name} Best Params: {best_params} (CV Best F1: {grid.best_score_:.4f})")

        # Evaluate Stratified CV Scores
        cv_scores = cross_val_score(best_estimator, X_train, y_train, cv=skf, scoring='accuracy')
        training_logger.info(f"{name} 5-Fold Stratified CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        training_logger.info(f"{name} Fold Results: {[round(s, 4) for s in cv_scores]}")

        # Wrap Linear SVM in CalibratedClassifierCV if probability is needed
        if name == "Linear SVM":
            calibrated = CalibratedClassifierCV(best_estimator)
            calibrated.fit(X_train, y_train)
            eval_model = calibrated
        else:
            eval_model = best_estimator

        # Test set evaluation
        t_pred = time.time()
        y_pred = eval_model.predict(X_test)
        pred_duration = time.time() - t_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, model_name=name, save_path=f"images/confusion_matrix_{name.lower().replace(' ', '_')}.png")

        benchmark_results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "CV_Mean_Acc": cv_scores.mean(),
            "CV_Std": cv_scores.std(),
            "Best_Params": str(best_params),
            "Training Time (s)": round(train_duration, 4),
            "Prediction Time (s)": round(pred_duration, 4)
        })

        trained_estimators[name] = eval_model

    results_df = pd.DataFrame(benchmark_results)
    training_logger.info("\n=== MODEL COMPARISON BENCHMARK TABLE ===")
    print(results_df.to_string(index=False))

    plot_model_comparison(results_df, save_path="images/model_comparison.png")

    # 6. Select Best Estimator
    best_row = results_df.sort_values(by="F1-Score", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_model = trained_estimators[best_name]

    training_logger.info(f"\n🏆 BEST MODEL SELECTED: {best_name} (F1-Score: {best_row['F1-Score']:.4f}, Accuracy: {best_row['Accuracy']:.4f})")

    # 7. Model Persistence & Versioning Metadata (model_info.json)
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)

    model_info = {
        "model_name": best_name,
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": meta["dataset_path"],
        "dataset_size": df.shape[0],
        "training_samples": X_train.shape[0],
        "testing_samples": X_test.shape[0],
        "training_accuracy": round(best_model.score(X_train, y_train), 4),
        "testing_accuracy": round(best_row["Accuracy"], 4),
        "precision": round(best_row["Precision"], 4),
        "recall": round(best_row["Recall"], 4),
        "f1_score": round(best_row["F1-Score"], 4),
        "cv_mean_accuracy": round(best_row["CV_Mean_Acc"], 4),
        "cv_std_dev": round(best_row["CV_Std"], 4),
        "vectorizer_parameters": TFIDF_PARAMETERS,
        "best_hyperparameters": best_row["Best_Params"],
        "training_time_seconds": best_row["Training Time (s)"],
        "prediction_time_seconds": best_row["Prediction Time (s)"]
    }

    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4)

    training_logger.info(f"Saved best model to {BEST_MODEL_PATH}")
    training_logger.info(f"Saved vectorizer to {TFIDF_VECTORIZER_PATH}")
    training_logger.info(f"Saved model versioning metadata to {MODEL_INFO_PATH}")
    training_logger.info("==================================================")
    training_logger.info("TRAINING & VERSIONING PIPELINE COMPLETED SUCCESSFULLY!")
    training_logger.info("==================================================")

if __name__ == "__main__":
    run_pipeline()
