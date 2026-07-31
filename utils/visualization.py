"""
utils/visualization.py
======================
High-resolution Visualization Generator for EDA and Model Evaluation.
Generates publication-quality charts, Word Clouds, Confusion Matrices, 
ROC Curves, and Model Comparison benchmarks using centralized config.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

from config import IMAGE_DIR
from utils.logger import training_logger

# Set global matplotlib aesthetic parameters
plt.style.use('ggplot')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

COLOR_POS = "#2ecc71"
COLOR_NEG = "#e74c3c"
COLOR_ACCENT = "#3498db"

def plot_sentiment_distribution(df: pd.DataFrame, target_col: str, save_path: Optional[str] = None) -> plt.Figure:
    """Plots and saves binary sentiment class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[target_col].value_counts()
    labels = ["Positive (1)" if k == 1 else "Negative (0)" for k in counts.index]
    colors = [COLOR_POS if k == 1 else COLOR_NEG for k in counts.index]

    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(f'{height}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title("Dataset Sentiment Class Distribution", fontweight='bold', pad=15)
    ax.set_ylabel("Number of Reviews")
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        training_logger.info(f"Saved sentiment distribution chart to {save_path}")

    return fig


def plot_review_length_distribution(df: pd.DataFrame, text_col: str, target_col: Optional[str] = None, save_path: Optional[str] = None) -> plt.Figure:
    """Plots review word length distribution histogram & KDE."""
    df_copy = df.copy()
    df_copy['word_count'] = df_copy[text_col].astype(str).apply(lambda x: len(x.split()))

    fig, ax = plt.subplots(figsize=(10, 5))
    
    if target_col and target_col in df_copy.columns:
        sns.histplot(data=df_copy, x='word_count', hue=target_col, palette={1: COLOR_POS, 0: COLOR_NEG},
                     kde=True, element="step", common_norm=False, ax=ax, alpha=0.6)
        plt.legend(title="Sentiment", labels=["Positive", "Negative"])
    else:
        sns.histplot(df_copy['word_count'], kde=True, color=COLOR_ACCENT, ax=ax, alpha=0.6)

    ax.set_title("Review Length Distribution (Word Count)", fontweight='bold', pad=15)
    ax.set_xlabel("Word Count per Review")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, np.percentile(df_copy['word_count'], 99))
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        training_logger.info(f"Saved review length distribution chart to {save_path}")

    return fig


def plot_top_ngrams(df: pd.DataFrame, text_col: str, n_gram_range: tuple = (1, 1), top_k: int = 15, title: str = "Top Words", save_path: Optional[str] = None) -> plt.Figure:
    """Generates bar chart for top N-grams (unigrams, bigrams, trigrams)."""
    vec = CountVectorizer(ngram_range=n_gram_range, stop_words='english', max_features=1000)
    bag_of_words = vec.fit_transform(df[text_col].fillna(""))
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]

    ngram_df = pd.DataFrame(words_freq, columns=['ngram', 'count'])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=ngram_df, x='count', y='ngram', hue='ngram', palette="viridis", legend=False, ax=ax, edgecolor="black", alpha=0.85)

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f'{int(width)}',
                    xy=(width, p.get_y() + p.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel("Frequency Count")
    ax.set_ylabel("N-Gram Term")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        training_logger.info(f"Saved n-gram chart to {save_path}")

    return fig


def generate_wordclouds(df: pd.DataFrame, text_col: str, target_col: str, pos_path: Optional[str] = None, neg_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Figure]:
    """Generates and saves Positive and Negative Word Clouds."""
    pos_text = " ".join(df[df[target_col] == 1][text_col].dropna().astype(str))
    neg_text = " ".join(df[df[target_col] == 0][text_col].dropna().astype(str))

    wc_pos = WordCloud(width=800, height=400, background_color="white", colormap="Greens", max_words=150).generate(pos_text if pos_text.strip() else "good great excellent")
    fig_pos, ax_pos = plt.subplots(figsize=(10, 5))
    ax_pos.imshow(wc_pos, interpolation="bilinear")
    ax_pos.axis("off")
    ax_pos.set_title("Positive Sentiment Word Cloud", fontweight="bold", fontsize=16, pad=15)
    plt.tight_layout()

    if pos_path:
        os.makedirs(os.path.dirname(pos_path), exist_ok=True)
        fig_pos.savefig(pos_path, dpi=300, bbox_inches='tight')
        training_logger.info(f"Saved positive wordcloud to {pos_path}")

    wc_neg = WordCloud(width=800, height=400, background_color="white", colormap="Reds", max_words=150).generate(neg_text if neg_text.strip() else "bad terrible worst refund")
    fig_neg, ax_neg = plt.subplots(figsize=(10, 5))
    ax_neg.imshow(wc_neg, interpolation="bilinear")
    ax_neg.axis("off")
    ax_neg.set_title("Negative Sentiment Word Cloud", fontweight="bold", fontsize=16, pad=15)
    plt.tight_layout()

    if neg_path:
        os.makedirs(os.path.dirname(neg_path), exist_ok=True)
        fig_neg.savefig(neg_path, dpi=300, bbox_inches='tight')
        training_logger.info(f"Saved negative wordcloud to {neg_path}")

    return fig_pos, fig_neg


def plot_confusion_matrix(cm: np.ndarray, model_name: str = "Model", save_path: Optional[str] = None) -> plt.Figure:
    """Plots heatmap confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'], ax=ax, annot_kws={"size": 14, "weight": "bold"})
    
    ax.set_title(f"Confusion Matrix - {model_name}", fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Sentiment", fontweight='bold')
    ax.set_ylabel("Actual Sentiment", fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        training_logger.info(f"Saved confusion matrix for {model_name} to {save_path}")

    return fig


def plot_model_comparison(metrics_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plots grouped bar chart comparing performance metrics across models."""
    metrics_melted = metrics_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                     var_name='Metric', value_name='Score')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=metrics_melted, x='Metric', y='Score', hue='Model', palette='Set2', ax=ax, edgecolor='black')

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:.3f}',
                        xy=(p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title("Machine Learning Models Benchmarking Comparison", fontweight='bold', pad=15)
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_ylim(0, 1.15)
    plt.legend(title="Classifier", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        training_logger.info(f"Saved model comparison chart to {save_path}")

    return fig
