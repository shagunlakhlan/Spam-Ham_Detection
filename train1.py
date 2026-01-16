"""
Train a spam detector from spamB.csv (or data.csv) and save spam_model.pkl
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import csv

# Increase CSV field size limit
csv.field_size_limit(10**7)


def infer_columns(df: pd.DataFrame, text_col: str = "auto", label_col: str = "auto"):
    cols = df.columns.tolist()

    # Detect label column
    if label_col == "auto":
        for c in ["label", "Label", "category", "Category", "class", "target", "is_spam", "spam", "v1"]:
            if c in cols:
                label_col = c
                break

    # Detect text column
    if text_col == "auto":
        for c in ["text", "Text", "message", "Message", "email", "Email", "body", "content", "sms", "v2"]:
            if c in cols:
                text_col = c
                break

        if text_col == "auto":
            object_cols = [c for c in cols if df[c].dtype == object]
            if object_cols:
                avg_len = {c: df[c].astype(str).str.len().mean() for c in object_cols}
                text_col = max(avg_len, key=avg_len.get)

    return text_col, label_col


def normalize_label(x: str) -> str:
    x = str(x).strip().lower()
    return "spam" if "spam" in x or x in ["1", "true", "yes"] else "ham"


def main():
    # Auto-detect dataset
    if os.path.exists("spamB.csv"):
        data_path = "spamB.csv"
    elif os.path.exists("data.csv"):
        data_path = "data.csv"
    else:
        print("❌ Error: Could not find spamB.csv or data.csv in this folder.")
        sys.exit(1)

    print(f"📂 Loading dataset: {data_path}")

    # Try multiple encodings
    encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(data_path, encoding=enc, engine="python")
            print(f"✅ Loaded data using encoding: {enc}")
            break
        except Exception:
            continue
    else:
        print("❌ Could not read file with common encodings.")
        sys.exit(1)

    # Infer column names
    tcol, lcol = infer_columns(df)
    if tcol is None or lcol is None:
        print("❌ Could not infer text/label columns.")
        sys.exit(1)

    print(f"✅ Detected text column: {tcol}, label column: {lcol}")

    # Clean dataframe
    df = df[[lcol, tcol]].rename(columns={lcol: "label", tcol: "text"}).dropna()

    # Lowercase text (VERY important)
    df["text"] = df["text"].astype(str).str.lower()

    df["label"] = df["label"].apply(normalize_label)
    df = df.drop_duplicates(subset=["text"])

    # Must have both classes
    if df["label"].nunique() < 2:
        print("❌ Dataset must contain both ham and spam labels.")
        sys.exit(1)

    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
        )
    except ValueError as e:
        print("❌ Stratify error:", e)
        print("Your dataset may not have enough spam or ham messages.")
        sys.exit(1)

    # Improved TF-IDF + LR pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True
        )),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ])

    print("🚀 Training model...")
    pipe.fit(X_train, y_train)

    # Evaluation
    y_pred = pipe.predict(X_test)
    print("\n🎯 Training Complete!")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model
    joblib.dump(pipe, "spam_model.pkl")
    print("\n💾 Saved trained model to spam_model.pkl")


if __name__ == "__main__":
    main()


