"""
tests/test_preprocess.py
=========================
Unit tests for text preprocessing pipeline and dataset auto-detector.
"""

import unittest
import pandas as pd
from utils.preprocess import clean_text_pipeline, TextPreprocessor, load_and_detect_dataset

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor(preserve_negations=True)

    def test_lowercasing_and_punctuation(self):
        raw = "GREAT Product!!! Loved it."
        cleaned = self.preprocessor.clean_text(raw)
        self.assertNotIn("!", cleaned)
        self.assertTrue(cleaned.islower() or cleaned == "")

    def test_html_and_url_removal(self):
        raw = "<p>Check out http://amazon.com for info</p>"
        cleaned = self.preprocessor.clean_text(raw)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("<p>", cleaned)

    def test_negation_preservation(self):
        raw = "Not good at all, would never recommend!"
        cleaned = self.preprocessor.clean_text(raw)
        self.assertIn("not", cleaned)
        self.assertIn("never", cleaned)

    def test_empty_input(self):
        self.assertEqual(clean_text_pipeline(""), "")
        self.assertEqual(clean_text_pipeline(None), "")

if __name__ == "__main__":
    unittest.main()
