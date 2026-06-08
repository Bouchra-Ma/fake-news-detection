"""
Fake News Detection - Baseline Models
TF-IDF + classical classifiers for comparison against DistilBERT.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import re, warnings
warnings.filterwarnings("ignore")

SEED = 42
OUTPUT_DIR = "outputs/baselines"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(data_path=r"C:\Users\Utilisateur\Downloads\fake-news-detection\data"):
    true_path = os.path.join(data_path, "True.csv")
    fake_path = os.path.join(data_path, "Fake.csv")
    if os.path.exists(true_path) and os.path.exists(fake_path):
        true_df = pd.read_csv(true_path)
        fake_df = pd.read_csv(fake_path)
        true_df["label"] = 1
        fake_df["label"] = 0
        df = pd.concat([true_df, fake_df], ignore_index=True)
        if "title" in df.columns and "text" in df.columns:
            df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
        elif "text" in df.columns:
            df["content"] = df["text"].fillna("")
    else:
        print("Using synthetic demo data for baselines.")
        import random, string
        random.seed(SEED)
        real = [random.choice(["Scientists confirm ", "Report shows "]) + "".join(random.choices(string.ascii_lowercase + " ", k=120)) for _ in range(500)]
        fake = [random.choice(["EXPOSED: ", "SHOCKING: "]) + "".join(random.choices(string.ascii_lowercase + " ", k=120)) for _ in range(500)]
        df = pd.DataFrame({"content": real + fake, "label": [1]*500 + [0]*500})

    df["content"] = df["content"].apply(clean_text)
    return df[["content", "label"]].dropna()


def run_baselines():
    df = load_data()
    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    models = {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50000)),
            ("clf", MultinomialNB()),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("clf", LinearSVC(max_iter=2000, random_state=SEED)),
        ]),
    }

    results = []
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(classification_report(y_test, preds, target_names=["Fake", "Real"]))
        results.append({"Model": name, "Accuracy": acc, "F1-score": f1})

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "baseline_results.csv"), index=False)
    print("\nBaseline Results:")
    print(results_df.to_string(index=False))

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results_df))
    w = 0.35
    ax.bar(x - w/2, results_df["Accuracy"], w, label="Accuracy", color="#3498db")
    ax.bar(x + w/2, results_df["F1-score"], w, label="F1-score", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Baseline Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "baseline_comparison.png"), dpi=150)
    plt.close()
    print("Saved baseline_comparison.png")


if __name__ == "__main__":
    run_baselines()
