"""
utils/preprocess.py
===================
Robust NLP Text Preprocessing Pipeline, Quality Auditor & Dataset Auto-Detector.
Complies with Step 17 Production Engineering Standards.
"""

import os
import re
import glob
from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from config import DATASET_PATH, DATA_DIR
from utils.logger import training_logger

# Ensure required NLTK resources are available
def _download_nltk_deps():
    resources = ['stopwords', 'wordnet', 'punkt', 'omw-1.4']
    for res in resources:
        try:
            nltk.data.find(f'corpora/{res}' if res in ['stopwords', 'wordnet', 'omw-1.4'] else f'tokenizers/{res}')
        except LookupError:
            try:
                nltk.download(res, quiet=True)
            except Exception as e:
                training_logger.warning(f"Could not download NLTK resource {res}: {e}")

_download_nltk_deps()


class TextPreprocessor:
    """
    Production-grade Text Preprocessor for NLP Sentiment Analysis.
    Handles lowercasing, HTML/URL stripping, regex punctuation removal,
    tokenization, stopword removal, and lemmatization with negation preservation.
    """
    def __init__(self, preserve_negations: bool = True):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = {'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}

        # Preserve negation tokens for accurate sentiment signaling
        self.negation_words = {'no', 'not', 'nor', 'neither', 'never', 'none', 'cannot', 'cant', 'couldnt', 'didnt', 'doesnt', 'dont', 'isnt', 'wasnt', 'wouldnt'}
        if preserve_negations:
            self.stop_words = self.stop_words - self.negation_words

    def clean_text(self, text: Any) -> str:
        """Executes complete multi-stage cleaning pipeline on raw text input."""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # 1. Lowercasing
        text = text.lower()

        # 2. HTML tag removal
        text = re.sub(r'<[^>]+>', ' ', text)

        # 3. URL removal
        text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)

        # 4. Email address removal
        text = re.sub(r'\S+@\S+', ' ', text)

        # 5. Emoji & Special Symbol/Digits removal
        text = re.sub(r'[^a-z\s]', ' ', text)

        # 6. Extra whitespace removal
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return ""

        # 7. Tokenization
        tokens = text.split()

        # 8. Stopword removal & Lemmatization
        cleaned_tokens = []
        for token in tokens:
            if len(token) > 1 and token not in self.stop_words:
                lemmatized = self.lemmatizer.lemmatize(token, pos='v')
                lemmatized = self.lemmatizer.lemmatize(lemmatized, pos='n')
                cleaned_tokens.append(lemmatized)

        return " ".join(cleaned_tokens)


_global_preprocessor = TextPreprocessor()

def clean_text_pipeline(text: Any) -> str:
    """Convenience functional wrapper for single text string cleaning."""
    return _global_preprocessor.clean_text(text)


def load_and_detect_dataset(dataset_path: Optional[str] = None) -> Tuple[pd.DataFrame, str, str, Dict[str, Any]]:
    """
    Automatically detects CSV files, loads dataset, and infers review text & sentiment columns.
    """
    candidate_paths = []
    if dataset_path and os.path.exists(dataset_path):
        if os.path.isdir(dataset_path):
            csvs = glob.glob(os.path.join(dataset_path, "*.csv"))
            candidate_paths.extend(csvs)
        else:
            candidate_paths.append(dataset_path)

    # Add default config location and fallback locations
    default_locations = [
        DATASET_PATH,
        os.path.join(DATA_DIR, "*.csv"),
        "data/DataSet (W5).csv",
        "C:\\Users\\HP\\Downloads\\IT SIMPLERA INTERNSHIP\\week f]8iv\\DataSet (W5).csv",
    ]

    for loc in default_locations:
        if "*" in loc:
            candidate_paths.extend(glob.glob(loc))
        elif os.path.exists(loc):
            candidate_paths.append(loc)

    if not candidate_paths:
        raise FileNotFoundError("Could not locate any CSV dataset file in candidate paths.")

    target_csv = candidate_paths[0]
    training_logger.info(f"Loading dataset from: {target_csv}")
    df = pd.read_csv(target_csv)

    # Dynamic Column Name Inference
    text_col_candidates = ['verified_reviews', 'review', 'text', 'reviews', 'review_text', 'comment', 'content']
    target_col_candidates = ['feedback', 'sentiment', 'rating', 'label', 'target', 'score']

    text_col = None
    for col in df.columns:
        if col.lower() in text_col_candidates:
            text_col = col
            break

    if not text_col:
        str_cols = df.select_dtypes(include=['object', 'string']).columns
        text_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean()) if len(str_cols) > 0 else df.columns[0]

    target_col = None
    for col in df.columns:
        if col.lower() in target_col_candidates:
            target_col = col
            break

    if not target_col:
        num_cols = df.select_dtypes(include=[np.number]).columns
        target_col = num_cols[-1] if len(num_cols) > 0 else df.columns[-1]

    # Convert rating (1-5) to binary target if required
    if df[target_col].nunique() > 2 and df[target_col].max() <= 5:
        df['binary_sentiment'] = (df[target_col] >= 4).astype(int)
        target_col = 'binary_sentiment'

    # Dataset Quality Audit
    qa_audit = audit_dataset_quality(df, text_col, target_col)

    metadata = {
        "dataset_path": target_csv,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "text_column": text_col,
        "target_column": target_col,
        "qa_audit": qa_audit,
        "class_distribution": df[target_col].value_counts().to_dict()
    }

    return df, text_col, target_col, metadata


def audit_dataset_quality(df: pd.DataFrame, text_col: str, target_col: str) -> Dict[str, Any]:
    """Runs Quality Assurance Audit checking nulls, duplicates, empty reviews, and class balance."""
    null_counts = df[[text_col, target_col]].isnull().sum().to_dict()
    dup_count = int(df.duplicated(subset=[text_col]).sum())
    empty_review_count = int((df[text_col].astype(str).str.strip() == "").sum())
    class_counts = df[target_col].value_counts().to_dict()
    
    total = len(df)
    imbalance_ratio = round(max(class_counts.values()) / min(class_counts.values()), 2) if min(class_counts.values()) > 0 else 0.0

    audit_result = {
        "null_counts": null_counts,
        "duplicate_count": dup_count,
        "empty_review_count": empty_review_count,
        "class_distribution": class_counts,
        "imbalance_ratio": imbalance_ratio,
        "is_imbalanced": imbalance_ratio > 2.0
    }

    training_logger.info(f"QA Audit Complete -> Nulls: {null_counts}, Duplicates: {dup_count}, Imbalance Ratio: {imbalance_ratio}:1")
    return audit_result
