"""
tests/test_training.py
=======================
Unit tests for data auto-loader and vectorizer feature extraction.
"""

import unittest
import os
import pandas as pd
from utils.preprocess import load_and_detect_dataset, clean_text_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

class TestTraining(unittest.TestCase):
    def test_dataset_loading(self):
        df, text_col, target_col, meta = load_and_detect_dataset()
        self.assertGreater(len(df), 0)
        self.assertIn(text_col, df.columns)
        self.assertIn(target_col, df.columns)

    def test_tfidf_matrix_shape(self):
        df, text_col, target_col, _ = load_and_detect_dataset()
        cleaned = df[text_col].head(100).apply(clean_text_pipeline)
        vec = TfidfVectorizer(max_features=500)
        X = vec.fit_transform(cleaned)
        self.assertEqual(X.shape[0], 100)
        self.assertLessEqual(X.shape[1], 500)

if __name__ == "__main__":
    unittest.main()
