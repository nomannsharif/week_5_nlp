"""
create_notebook.py
==================
Generates a complete, fully executable Jupyter Notebook (.ipynb) with markdown explanations
and code cells for Step 12 requirement.
"""

import os
import json

def build_notebook():
    notebook_path = "notebooks/week5_nlp.ipynb"
    os.makedirs("notebooks", exist_ok=True)

    cells = []

    # Title & Markdown intro
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ⚡ Week 5: Production NLP Sentiment Analysis Project\n",
            "**Author:** Senior AI / ML Engineering Specialist  \n",
            "**Dataset:** Amazon Product Reviews Dataset (`DataSet (W5).csv`)  \n",
            "**Objective:** Build an end-to-end sentiment classification pipeline using TF-IDF feature extraction and benchmarked machine learning models (Logistic Regression, Naive Bayes, Linear SVM).\n"
        ]
    })

    # Imports Cell
    cells.append({
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import re\n",
            "import time\n",
            "import joblib\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from wordcloud import WordCloud\n",
            "\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.feature_extraction.text import TfidfVectorizer\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.naive_bayes import MultinomialNB\n",
            "from sklearn.svm import LinearSVC\n",
            "from sklearn.calibration import CalibratedClassifierCV\n",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix\n",
            "\n",
            "import sys\n",
            "sys.path.append('..')\n",
            "from utils.preprocess import load_and_detect_dataset, clean_text_pipeline\n",
            "from utils.visualization import plot_sentiment_distribution, plot_review_length_distribution, generate_wordclouds, plot_confusion_matrix, plot_model_comparison\n",
            "\n",
            "print('All libraries imported successfully!')"
        ]
    })

    # Step 1: Data Inspection
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Data Loading & Dynamic Inspection\n",
            "Using our dataset auto-loader to automatically locate the CSV file and infer text and target columns."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "df, text_col, target_col, meta = load_and_detect_dataset('../data/DataSet (W5).csv')\n",
            "print(f'Dataset Shape: {df.shape}')\n",
            "print(f'Detected Review Text Column: {text_col}')\n",
            "print(f'Detected Target Sentiment Column: {target_col}')\n",
            "print('\\nData Head:')\n",
            "display(df.head(5))\n",
            "print('\\nMissing Values Count:\\n', df.isnull().sum())\n",
            "print('\\nClass Distribution:\\n', df[target_col].value_counts())"
        ]
    })

    # Step 2: Text Preprocessing
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Text Preprocessing & Cleaning Pipeline\n",
            "Applying lowercasing, HTML/URL removal, punctuation stripping, stopword filtering with negation preservation, and NLTK lemmatization."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "start_time = time.time()\n",
            "df['clean_text'] = df[text_col].apply(clean_text_pipeline)\n",
            "df = df[df['clean_text'].str.strip() != ''].copy()\n",
            "print(f'Cleaned {len(df)} non-empty reviews in {time.time() - start_time:.2f} seconds.')\n",
            "print('\\nSample Cleaned Text Output:')\n",
            "for i in range(3):\n",
            "    print(f'RAW: {df[text_col].iloc[i]}')\n",
            "    print(f'CLEANED: {df[\"clean_text\"].iloc[i]}\\n')"
        ]
    })

    # Step 3: EDA & Visualizations
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Exploratory Data Analysis (EDA)\n",
            "Visualizing class distribution, review length metrics, and Positive/Negative Word Clouds."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig_dist = plot_sentiment_distribution(df, target_col)\n",
            "plt.show()\n",
            "\n",
            "fig_len = plot_review_length_distribution(df, text_col, target_col)\n",
            "plt.show()\n",
            "\n",
            "fig_pos, fig_neg = generate_wordclouds(df, 'clean_text', target_col)\n",
            "plt.show()"
        ]
    })

    # Step 4: TF-IDF Feature Engineering
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Feature Extraction - TF-IDF Vectorization\n",
            "Transforming text into numeric term frequency-inverse document frequency vectors with sublinear scaling."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [],
        "source": [
            "vectorizer = TfidfVectorizer(\n",
            "    max_features=5000,\n",
            "    ngram_range=(1, 2),\n",
            "    min_df=2,\n",
            "    max_df=0.95,\n",
            "    sublinear_tf=True\n",
            ")\n",
            "X = vectorizer.fit_transform(df['clean_text'])\n",
            "y = df[target_col].values\n",
            "\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)\n",
            "print(f'TF-IDF Feature Matrix Shape: {X.shape}')\n",
            "print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')"
        ]
    })

    # Step 5 & 6: Model Training & Evaluation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5 & 6. Model Training, Benchmarking & Evaluation\n",
            "Training Logistic Regression, Multinomial Naive Bayes, and Calibrated Linear SVM models."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [],
        "source": [
            "models = {\n",
            "    'Logistic Regression': LogisticRegression(C=1.0, solver='liblinear', random_state=42),\n",
            "    'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),\n",
            "    'Linear SVM': CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=2000))\n",
            "}\n",
            "\n",
            "results = []\n",
            "for name, model in models.items():\n",
            "    t0 = time.time()\n",
            "    model.fit(X_train, y_train)\n",
            "    train_t = time.time() - t0\n",
            "    \n",
            "    p0 = time.time()\n",
            "    y_pred = model.predict(X_test)\n",
            "    pred_t = time.time() - p0\n",
            "    \n",
            "    acc = accuracy_score(y_test, y_pred)\n",
            "    prec = precision_score(y_test, y_pred, zero_division=0)\n",
            "    rec = recall_score(y_test, y_pred, zero_division=0)\n",
            "    f1 = f1_score(y_test, y_pred, zero_division=0)\n",
            "    \n",
            "    results.append({\n",
            "        'Model': name,\n",
            "        'Accuracy': acc,\n",
            "        'Precision': prec,\n",
            "        'Recall': rec,\n",
            "        'F1-Score': f1,\n",
            "        'Train Time (s)': round(train_t, 4),\n",
            "        'Pred Time (s)': round(pred_t, 4)\n",
            "    })\n",
            "    \n",
            "    print(f'=== {name} ===')\n",
            "    print(classification_report(y_test, y_pred))\n",
            "    cm = confusion_matrix(y_test, y_pred)\n",
            "    plot_confusion_matrix(cm, model_name=name)\n",
            "    plt.show()\n",
            "\n",
            "res_df = pd.DataFrame(results)\n",
            "display(res_df)\n",
            "plot_model_comparison(res_df)\n",
            "plt.show()"
        ]
    })

    # Step 7: Model Persistence
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Model Selection & Persistence\n",
            "Selecting the top F1-Score model and persisting artifacts using `joblib`."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": 7,
        "metadata": {},
        "outputs": [],
        "source": [
            "best_row = res_df.sort_values(by='F1-Score', ascending=False).iloc[0]\n",
            "best_name = best_row['Model']\n",
            "print(f'🏆 Best Model Selected: {best_name} (F1: {best_row[\"F1-Score\"]:.4f})')\n",
            "\n",
            "best_model = models[best_name]\n",
            "os.makedirs('../models', exist_ok=True)\n",
            "joblib.dump(best_model, '../models/best_model.pkl')\n",
            "joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')\n",
            "print('Successfully saved model and vectorizer to ../models/ directory!')"
        ]
    })

    nb_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb_content, f, indent=2)

    print(f"Created notebook at {notebook_path}")

if __name__ == "__main__":
    build_notebook()
