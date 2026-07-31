"""
tests/test_prediction.py
=========================
Unit tests for SentimentPredictor inference engine.
"""

import unittest
import pandas as pd
from utils.prediction import get_predictor, SentimentPredictor

class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.predictor = get_predictor()

    def test_predictor_loaded(self):
        self.assertTrue(self.predictor.is_loaded)
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.vectorizer)

    def test_positive_prediction(self):
        text = "This Echo Dot is fantastic! Sound quality is crystal clear and setup took 2 minutes."
        res = self.predictor.predict(text)
        self.assertEqual(res["sentiment"], "Positive")
        self.assertGreaterEqual(res["confidence"], 50.0)
        self.assertIn("Positive", res["probabilities"])

    def test_negative_prediction(self):
        text = "Terrible experience, product broke on day 1 and customer service was awful."
        res = self.predictor.predict(text)
        self.assertEqual(res["sentiment"], "Negative")
        self.assertGreaterEqual(res["confidence"], 50.0)

    def test_batch_prediction(self):
        sample_list = [
            "Loved it!",
            "Worst purchase ever."
        ]
        df_res = self.predictor.predict_batch(sample_list)
        self.assertIsInstance(df_res, pd.DataFrame)
        self.assertEqual(len(df_res), 2)
        self.assertIn("predicted_sentiment", df_res.columns)

if __name__ == "__main__":
    unittest.main()
