"""
Fake News Detection - Training Script
Fine-tuning DistilBERT on the Kaggle Fake News Dataset
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from transformers import AdamW
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_len": 256,
    "batch_size": 16,
    "epochs": 3,
    "lr": 2e-5,
    "warmup_steps": 100,
    "seed": 42,
    "output_dir": "outputs",
    "data_path": "data/",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# Data Loading & Preprocessing
# ─────────────────────────────────────────────
def load_and_preprocess(data_path: str) -> pd.DataFrame:
    """Load Kaggle Fake News dataset (True.csv + Fake.csv)."""
    true_path = os.path.join(data_path, "True.csv")
    fake_path = os.path.join(data_path, "Fake.csv")

    if os.path.exists(true_path) and os.path.exists(fake_path):
        true_df = pd.read_csv(true_path)
        fake_df = pd.read_csv(fake_path)
        true_df["label"] = 1  # real
        fake_df["label"] = 0  # fake
        df = pd.concat([true_df, fake_df], ignore_index=True)
        print(f"Loaded {len(true_df)} real and {len(fake_df)} fake articles.")
    else:
        print("Dataset files not found. Generating synthetic demo data...")
        df = _generate_demo_data()

    # Build text feature: title + body
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("")
    else:
        df["content"] = df.iloc[:, 0].astype(str)

    df = df[["content", "label"]].dropna()
    df["content"] = df["content"].str.strip()
    df = df[df["content"].str.len() > 20]
    df = df.sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)
    return df


def _generate_demo_data(n=2000) -> pd.DataFrame:
    """Create a small synthetic dataset for demo / CI purposes."""
    import random, string
    random.seed(CONFIG["seed"])
    real_starters = [
        "Scientists confirm ", "Government announces ", "Report shows ",
        "New study finds ", "Officials say ", "According to sources, ",
    ]
    fake_starters = [
        "BREAKING: Secret ", "They don't want you to know ", "EXPOSED: ",
        "Shocking truth about ", "You won't believe ", "Hidden agenda behind ",
    ]
    records = []
    for i in range(n // 2):
        records.append({"text": random.choice(real_starters) + "".join(random.choices(string.ascii_lowercase + " ", k=120)), "label": 1})
        records.append({"text": random.choice(fake_starters) + "".join(random.choices(string.ascii_lowercase + " ", k=120)), "label": 0})
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds_all.extend(outputs.logits.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="weighted")
    return avg_loss, acc, f1


def eval_epoch(model, loader):
    model.eval()
    total_loss, preds_all, labels_all = 0, [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds_all.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="weighted")
    return avg_loss, acc, f1, preds_all, labels_all


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────
def plot_history(history: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["loss", "accuracy", "f1"]):
        ax.plot(history[f"train_{metric}"], label="Train", marker="o")
        ax.plot(history[f"val_{metric}"], label="Val", marker="o")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_history.png"), dpi=150)
    plt.close()
    print("Saved training_history.png")


def plot_confusion(labels, preds):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # 1. Data
    df = load_and_preprocess(CONFIG["data_path"])
    print(f"\nDataset shape: {df.shape}")
    print(df["label"].value_counts().rename({0: "Fake", 1: "Real"}))

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=CONFIG["seed"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=CONFIG["seed"])
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 2. Tokenizer & Datasets
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG["model_name"])

    train_ds = FakeNewsDataset(train_df["content"].tolist(), train_df["label"].tolist(), tokenizer, CONFIG["max_len"])
    val_ds   = FakeNewsDataset(val_df["content"].tolist(),   val_df["label"].tolist(),   tokenizer, CONFIG["max_len"])
    test_ds  = FakeNewsDataset(test_df["content"].tolist(),  test_df["label"].tolist(),  tokenizer, CONFIG["max_len"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"])
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"])

    # 3. Model
    model = DistilBertForSequenceClassification.from_pretrained(CONFIG["model_name"], num_labels=2)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"])
    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, CONFIG["warmup_steps"], total_steps)

    # 4. Training loop
    history = {k: [] for k in ["train_loss", "train_accuracy", "train_f1", "val_loss", "val_accuracy", "val_f1"]}
    best_val_f1 = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n{'='*50}\nEpoch {epoch}/{CONFIG['epochs']}\n{'='*50}")
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, scheduler)
        vl_loss, vl_acc, vl_f1, _, _ = eval_epoch(model, val_loader)

        history["train_loss"].append(tr_loss);   history["val_loss"].append(vl_loss)
        history["train_accuracy"].append(tr_acc); history["val_accuracy"].append(vl_acc)
        history["train_f1"].append(tr_f1);        history["val_f1"].append(vl_f1)

        print(f"Train — Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f} | F1: {tr_f1:.4f}")
        print(f"Val   — Loss: {vl_loss:.4f} | Acc: {vl_acc:.4f} | F1: {vl_f1:.4f}")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            model.save_pretrained(os.path.join(CONFIG["output_dir"], "best_model"))
            tokenizer.save_pretrained(os.path.join(CONFIG["output_dir"], "best_model"))
            print("✓ Best model saved.")

    # 5. Test evaluation
    print("\n" + "="*50 + "\nTest Set Evaluation\n" + "="*50)
    _, test_acc, test_f1, test_preds, test_labels = eval_epoch(model, test_loader)
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test F1-score : {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=["Fake", "Real"]))

    # 6. Plots
    plot_history(history)
    plot_confusion(test_labels, test_preds)

    # 7. Save metrics
    metrics = {"test_accuracy": test_acc, "test_f1": test_f1}
    pd.DataFrame([metrics]).to_csv(os.path.join(CONFIG["output_dir"], "metrics.csv"), index=False)
    print("\nAll done! Results saved to outputs/")


if __name__ == "__main__":
    main()
