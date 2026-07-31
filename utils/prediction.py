"""
utils/prediction.py
====================
Inference Engine & Real-time Batch/Single Sentiment Predictor.
Loads trained ML models and TF-IDF vectorizers, logs predictions to logs/prediction.log,
computes confidence scores, and extracts key influential words.
"""

import os
import joblib
from typing import Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd

from config import BEST_MODEL_PATH, TFIDF_VECTORIZER_PATH, MODEL_DIR
from utils.logger import prediction_logger
from utils.preprocess import clean_text_pipeline, load_and_detect_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class SentimentPredictor:
    """
    Production Inference Engine for NLP Sentiment Classification.
    """
    def __init__(self, model_path: str = BEST_MODEL_PATH, vectorizer_path: str = TFIDF_VECTORIZER_PATH):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads fitted model and vectorizer from disk or initializes fallback model."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.is_loaded = True
                prediction_logger.info("Successfully loaded model and vectorizer artifacts.")
            else:
                prediction_logger.warning("Artifacts missing. Initializing fallback inline model...")
                self._train_fallback_model()
        except Exception as e:
            prediction_logger.error(f"Error loading artifacts: {e}. Falling back to inline training.")
            self._train_fallback_model()

    def _train_fallback_model(self):
        """Trains a lightweight fallback model if saved artifacts are missing."""
        try:
            df, text_col, target_col, _ = load_and_detect_dataset()
            df['clean_text'] = df[text_col].apply(clean_text_pipeline)
            df = df[df['clean_text'].str.strip() != ""]

            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X = self.vectorizer.fit_transform(df['clean_text'])
            y = df[target_col].values

            self.model = LogisticRegression(C=1.0, max_iter=1000)
            self.model.fit(X, y)
            self.is_loaded = True

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vectorizer, self.vectorizer_path)
            prediction_logger.info("Fallback model successfully trained and saved.")
        except Exception as e:
            prediction_logger.error(f"Failed to train fallback model: {e}")
            self.is_loaded = False

    def predict(self, text: str) -> Dict[str, Any]:
        """Predicts sentiment for an arbitrary input text string."""
        if not text or not text.strip():
            return {
                "text": text,
                "clean_text": "",
                "sentiment": "Neutral",
                "confidence": 0.0,
                "probabilities": {"Positive": 0.5, "Negative": 0.5},
                "emoji": "😐",
                "influential_words": []
            }

        cleaned = clean_text_pipeline(text)
        if not cleaned.strip():
            return {
                "text": text,
                "clean_text": "",
                "sentiment": "Neutral",
                "confidence": 50.0,
                "probabilities": {"Positive": 0.5, "Negative": 0.5},
                "emoji": "😐",
                "influential_words": []
            }

        vec_input = self.vectorizer.transform([cleaned])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec_input)[0]
            prob_neg = float(probs[0])
            prob_pos = float(probs[1])
        else:
            decision = self.model.decision_function(vec_input)[0]
            prob_pos = float(1 / (1 + np.exp(-decision)))
            prob_neg = float(1 - prob_pos)

        pred_class = 1 if prob_pos >= 0.5 else 0
        sentiment_label = "Positive" if pred_class == 1 else "Negative"
        confidence = prob_pos * 100 if pred_class == 1 else prob_neg * 100
        emoji = "😊" if pred_class == 1 else "😞"

        influential_words = self._explain_prediction(vec_input, cleaned)

        prediction_logger.info(f"Prediction -> Input len: {len(text)} | Sentiment: {sentiment_label} | Conf: {confidence:.2f}%")

        return {
            "text": text,
            "clean_text": cleaned,
            "sentiment": sentiment_label,
            "confidence": round(confidence, 2),
            "probabilities": {
                "Positive": round(prob_pos, 4),
                "Negative": round(prob_neg, 4)
            },
            "emoji": emoji,
            "influential_words": influential_words
        }

    def predict_batch(self, texts: List[str]) -> pd.DataFrame:
        """Runs fast batch inference on a list of input review texts."""
        cleaned_texts = [clean_text_pipeline(t) for t in texts]
        vec_inputs = self.vectorizer.transform(cleaned_texts)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec_inputs)
            prob_negs = probs[:, 0]
            prob_poses = probs[:, 1]
        else:
            decisions = self.model.decision_function(vec_inputs)
            prob_poses = 1 / (1 + np.exp(-decisions))
            prob_negs = 1 - prob_poses

        preds = (prob_poses >= 0.5).astype(int)
        labels = ["Positive" if p == 1 else "Negative" for p in preds]
        confidences = np.where(preds == 1, prob_poses * 100, prob_negs * 100)

        results_df = pd.DataFrame({
            "original_text": texts,
            "cleaned_text": cleaned_texts,
            "predicted_sentiment": labels,
            "confidence_pct": np.round(confidences, 2),
            "prob_positive": np.round(prob_poses, 4),
            "prob_negative": np.round(prob_negs, 4)
        })

        prediction_logger.info(f"Batch Prediction Complete for {len(texts)} samples.")
        return results_df

    def _explain_prediction(self, vec_input, cleaned_text: str) -> List[Tuple[str, float]]:
        """Extracts key positive or negative word weights from TF-IDF feature space."""
        try:
            feature_names = np.array(self.vectorizer.get_feature_names_out())
            feature_indices = vec_input.nonzero()[1]

            if len(feature_indices) == 0:
                return []

            if hasattr(self.model, "coef_"):
                coefs = self.model.coef_[0]
            elif hasattr(self.model, "calibrated_classifiers_"):
                coefs = np.mean([clf.base_estimator.coef_[0] for clf in self.model.calibrated_classifiers_], axis=0)
            else:
                return []

            word_weights = []
            for idx in feature_indices:
                word = feature_names[idx]
                weight = coefs[idx]
                word_weights.append((word, round(float(weight), 4)))

            word_weights = sorted(word_weights, key=lambda x: abs(x[1]), reverse=True)
            return word_weights[:8]
        except Exception:
            return []


_global_predictor = None

def get_predictor() -> SentimentPredictor:
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = SentimentPredictor()
    return _global_predictor
