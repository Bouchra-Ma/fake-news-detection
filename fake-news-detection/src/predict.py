"""
Fake News Detection - Inference Script
Run predictions on new articles using the fine-tuned model.
"""

import os
import torch
import argparse
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: str = "outputs/best_model"):
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model, max_len: int = 256) -> dict:
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    label = int(probs.argmax())
    return {
        "label": "REAL" if label == 1 else "FAKE",
        "confidence": float(probs[label]),
        "prob_fake": float(probs[0]),
        "prob_real": float(probs[1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict whether a news article is fake or real.")
    parser.add_argument("--text", type=str, default=None, help="Article text to classify.")
    parser.add_argument("--file", type=str, default=None, help="Path to a .txt file containing the article.")
    parser.add_argument("--model_dir", type=str, default="outputs/best_model", help="Path to saved model.")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Model not found at '{args.model_dir}'. Run src/train.py first.")
        return

    print(f"Loading model from {args.model_dir}...")
    tokenizer, model = load_model(args.model_dir)

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        # Interactive demo
        print("\n=== Fake News Detector (interactive) ===")
        print("Type or paste an article snippet. Enter an empty line to quit.\n")
        while True:
            text = input("Article: ").strip()
            if not text:
                break
            result = predict(text, tokenizer, model)
            print(f"  ➜  {result['label']}  (confidence: {result['confidence']:.2%})")
            print(f"     P(Fake)={result['prob_fake']:.3f}  P(Real)={result['prob_real']:.3f}\n")
        return

    result = predict(text, tokenizer, model)
    print(f"\nPrediction : {result['label']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print(f"P(Fake)    : {result['prob_fake']:.3f}")
    print(f"P(Real)    : {result['prob_real']:.3f}")


if __name__ == "__main__":
    main()
