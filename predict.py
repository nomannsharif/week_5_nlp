"""
predict.py
==========
Standalone CLI Interface & Batch Prediction Script.
Supports single text classification and CSV batch processing.
Usage:
    python predict.py --text "This product is fantastic!"
    python predict.py --file data/sample_reviews.csv --output results/batch_predictions.csv
"""

import sys
import argparse
import pandas as pd

# Safe terminal output encoding for Windows CLI
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from utils.prediction import get_predictor
from utils.logger import prediction_logger

def main():
    parser = argparse.ArgumentParser(description="Real-Time Sentiment Predictor CLI & Batch Inference Engine")
    parser.add_argument("--text", type=str, help="Single review text string to classify")
    parser.add_argument("--file", type=str, help="Path to input CSV file for batch prediction")
    parser.add_argument("--output", type=str, default="results/batch_predictions.csv", help="Path to save batch CSV predictions")
    parser.add_argument("--col", type=str, default=None, help="Specific column name containing text in CSV")

    args = parser.parse_args()

    predictor = get_predictor()

    if args.text:
        res = predictor.predict(args.text)
        print("\n=== SENTIMENT PREDICTION RESULT ===")
        print(f"Original Text: {res['text']}")
        print(f"Cleaned Text:  {res['clean_text']}")
        print(f"Sentiment:     {res['sentiment']}")
        print(f"Confidence:    {res['confidence']}%")
        print(f"Probabilities: {res['probabilities']}")
        if res['influential_words']:
            print("Influential Words:", res['influential_words'])
        print("================广================\n")

    elif args.file:
        print(f"Loading CSV batch file from: {args.file}")
        df = pd.read_csv(args.file)
        
        if args.col and args.col in df.columns:
            text_col = args.col
        else:
            str_cols = df.select_dtypes(include=['object', 'string']).columns
            text_col = str_cols[0] if len(str_cols) > 0 else df.columns[0]

        print(f"Using text column: '{text_col}'")
        texts = df[text_col].astype(str).tolist()
        
        results_df = predictor.predict_batch(texts)
        results_df.to_csv(args.output, index=False)
        print(f"Successfully processed {len(results_df)} reviews. Saved results to: {args.output}")

    else:
        print("Please specify either --text \"your review\" or --file \"path/to/file.csv\". Use --help for details.")

if __name__ == "__main__":
    main()
