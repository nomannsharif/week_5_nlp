"""
evaluate.py
===========
Independent Model Evaluation & Automated Results Exporter Script.
Generates structured metric artifacts in results/ directory:
- results/metrics.csv
- results/classification_report.txt
- results/benchmark_table.csv
- results/prediction_examples.csv
- results/evaluation_report.md
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from config import (
    BEST_MODEL_PATH, TFIDF_VECTORIZER_PATH, RESULTS_DIR,
    METRICS_CSV_PATH, CLASSIFICATION_REPORT_PATH, BENCHMARK_TABLE_PATH,
    PREDICTION_EXAMPLES_PATH, EVALUATION_REPORT_PATH, RANDOM_STATE, TEST_SIZE
)
from utils.logger import training_logger
from utils.preprocess import load_and_detect_dataset, clean_text_pipeline

def run_evaluation():
    training_logger.info("==================================================")
    training_logger.info("STARTING MODEL EVALUATION & RESULTS EXPORT")
    training_logger.info("==================================================")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load Dataset & Clean Text
    df, text_col, target_col, meta = load_and_detect_dataset()
    df['clean_text'] = df[text_col].apply(clean_text_pipeline)
    df = df[df['clean_text'].str.strip() != ""].copy()

    # 2. Load Model & Vectorizer Artifacts
    if not os.path.exists(BEST_MODEL_PATH) or not os.path.exists(TFIDF_VECTORIZER_PATH):
        raise FileNotFoundError("Model or Vectorizer artifact not found. Please run 'python train.py' first.")

    model = joblib.load(BEST_MODEL_PATH)
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

    # 3. Vectorize Text & Split
    X = vectorizer.transform(df['clean_text'])
    y = df[target_col].values

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 4. Predict
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Compute Core Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    clf_report_text = classification_report(y_test, y_pred, target_names=["Negative (0)", "Positive (1)"])

    # --- Export 1: metrics.csv ---
    metrics_df = pd.DataFrame([{
        "Metric": m,
        "Score": round(s, 4)
    } for m, s in [("Accuracy", acc), ("Precision", prec), ("Recall", rec), ("F1-Score", f1)]])
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    training_logger.info(f"Saved {METRICS_CSV_PATH}")

    # --- Export 2: classification_report.txt ---
    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=== CLASSIFICATION REPORT ===\n\n")
        f.write(clf_report_text)
    training_logger.info(f"Saved {CLASSIFICATION_REPORT_PATH}")

    # --- Export 3: benchmark_table.csv ---
    benchmark_df = pd.DataFrame([{
        "Model": type(model).__name__,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4)
    }])
    benchmark_df.to_csv(BENCHMARK_TABLE_PATH, index=False)
    training_logger.info(f"Saved {BENCHMARK_TABLE_PATH}")

    # --- Export 4: prediction_examples.csv ---
    test_df = df.loc[test_idx].copy()
    test_df["actual_label"] = y_test
    test_df["predicted_label"] = y_pred
    test_df["predicted_sentiment"] = np.where(y_pred == 1, "Positive", "Negative")
    if y_probs is not None:
        test_df["confidence_pct"] = np.round(np.where(y_pred == 1, y_probs * 100, (1 - y_probs) * 100), 2)
    
    sample_examples = test_df[[text_col, "clean_text", "actual_label", "predicted_label", "predicted_sentiment"]].head(25)
    sample_examples.to_csv(PREDICTION_EXAMPLES_PATH, index=False)
    training_logger.info(f"Saved {PREDICTION_EXAMPLES_PATH}")

    # --- Export 5: evaluation_report.md ---
    md_content = f"""# 📊 Automated Model Evaluation Report

**Dataset:** `{meta['dataset_path']}`  
**Model Architecture:** `{type(model).__name__}`  
**Total Test Samples:** `{len(y_test)}`  

## 🎯 Overall Metric Scores

| Metric | Score |
| :--- | :---: |
| **Accuracy** | `{acc * 100:.2f}%` |
| **Precision** | `{prec * 100:.2f}%` |
| **Recall** | `{rec * 100:.2f}%` |
| **F1-Score** | `{f1 * 100:.2f}%` |

## 📄 Classification Report

```text
{clf_report_text}
```

## 🔍 Sample Prediction Analysis

Saved 25 detailed prediction examples to [`results/prediction_examples.csv`](file:///{PREDICTION_EXAMPLES_PATH}).
"""
    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    training_logger.info(f"Saved {EVALUATION_REPORT_PATH}")

    training_logger.info("==================================================")
    training_logger.info("ALL EVALUATION ARTIFACTS EXPORTED SUCCESSFULLY!")
    training_logger.info("==================================================")

if __name__ == "__main__":
    run_evaluation()
