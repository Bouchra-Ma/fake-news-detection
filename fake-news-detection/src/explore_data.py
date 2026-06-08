"""
Fake News Detection - Data Exploration & Preprocessing
Exploratory Data Analysis (EDA) on the Kaggle Fake News dataset.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

OUTPUT_DIR = "outputs/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"<.*?>", "", text)                    # remove HTML tags
    text = re.sub(r"[^a-z\s']", " ", text)              # keep only letters
    text = re.sub(r"\s+", " ", text).strip()            # collapse whitespace
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("")
    else:
        df["content"] = df.iloc[:, 0].astype(str)

    df = df[["content", "label"]].dropna()
    df["content_clean"] = df["content"].apply(clean_text)
    df["word_count"] = df["content_clean"].apply(lambda x: len(x.split()))
    df["char_count"] = df["content_clean"].apply(len)
    return df


# ─────────────────────────────────────────────
# EDA helpers
# ─────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame):
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Fake (0)", "Real (1)"], [counts.get(0, 0), counts.get(1, 0)],
                  color=["#e74c3c", "#2ecc71"], edgecolor="black")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=11)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
    plt.close()
    print("Saved class_distribution.png")


def plot_word_count_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, color, name in [(0, "#e74c3c", "Fake"), (1, "#2ecc71", "Real")]:
        subset = df[df["label"] == label]["word_count"]
        for ax in axes:
            ax.hist(subset, bins=50, alpha=0.6, color=color, label=name, density=True)
    axes[0].set_title("Word Count Distribution")
    axes[0].set_xlabel("Word Count")
    axes[0].legend()
    axes[1].set_xlim(0, 1000)
    axes[1].set_title("Word Count Distribution (zoomed)")
    axes[1].set_xlabel("Word Count")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "word_count_distribution.png"), dpi=150)
    plt.close()
    print("Saved word_count_distribution.png")


def plot_wordclouds(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, label, title in [(axes[0], 0, "Fake News"), (axes[1], 1, "Real News")]:
        texts = " ".join(df[df["label"] == label]["content_clean"].tolist())
        wc = WordCloud(width=700, height=400, background_color="white",
                       colormap="Reds" if label == 0 else "Greens",
                       max_words=100).generate(texts)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "wordclouds.png"), dpi=150)
    plt.close()
    print("Saved wordclouds.png")


def print_sample_stats(df: pd.DataFrame):
    print("\n=== Dataset Statistics ===")
    print(f"Total samples  : {len(df):,}")
    print(f"Fake articles  : {(df['label']==0).sum():,}")
    print(f"Real articles  : {(df['label']==1).sum():,}")
    print(f"\nWord count stats:")
    print(df.groupby("label")["word_count"].describe().rename(index={0: "Fake", 1: "Real"}))


def load_data(data_path: str) -> pd.DataFrame:
    true_path = os.path.join(data_path, "True.csv")
    fake_path = os.path.join(data_path, "Fake.csv")

    if os.path.exists(true_path) and os.path.exists(fake_path):
        true_df = pd.read_csv(true_path)
        fake_df = pd.read_csv(fake_path)
        true_df["label"] = 1
        fake_df["label"] = 0
        return pd.concat([true_df, fake_df], ignore_index=True)
    else:
        print("Dataset not found. Using synthetic demo data for EDA.")
        from src.train import _generate_demo_data
        return _generate_demo_data(n=1000)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    df_raw = load_data("data/")
    df = preprocess(df_raw)
    print_sample_stats(df)

    plot_class_distribution(df)
    plot_word_count_distribution(df)
    try:
        plot_wordclouds(df)
    except ImportError:
        print("wordcloud not installed – skipping wordcloud plot.")

    df.to_csv(os.path.join(OUTPUT_DIR, "preprocessed_data.csv"), index=False)
    print("\nPreprocessed data saved to outputs/eda/preprocessed_data.csv")
    print("EDA complete!")


if __name__ == "__main__":
    main()
