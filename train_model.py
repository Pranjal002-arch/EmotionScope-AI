from pathlib import Path
import os
import re
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------------------
# CONFIG
# ---------------------------
LABEL_ORDER = ["sadness", "joy", "love", "anger", "fear", "surprise"]

BASE_DIR = Path(__file__).resolve().parent
CANDIDATE_DIRS = [
    BASE_DIR,
    BASE_DIR.parent,
    BASE_DIR / "archive",
    BASE_DIR.parent / "archive",
]

OUTPUT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------
# FIND FILE
# ---------------------------
def find_file(filename):
    for folder in CANDIDATE_DIRS:
        path = folder / filename
        if path.exists():
            return path
    searched = "\n".join(str(folder / filename) for folder in CANDIDATE_DIRS)
    raise FileNotFoundError(f"{filename} not found. Searched:\n{searched}")


TRAIN_FILE = find_file("train.txt")
TEST_FILE = find_file("test.txt")


# ---------------------------
# CLEAN TEXT
# ---------------------------
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------
# LOAD DATA
# ---------------------------
def load_data(file_path):
    rows = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ";" not in line:
                continue

            text, label = line.rsplit(";", 1)
            text = clean_text(text)
            label = str(label).strip().lower()

            if not text:
                continue
            if label not in LABEL_ORDER:
                continue

            rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No valid rows found in {file_path}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"] != ""]
    df = df[df["label"].isin(LABEL_ORDER)]

    if df.empty:
        raise ValueError(f"All rows became invalid after cleaning in {file_path}")

    return df.reset_index(drop=True)


# ---------------------------
# VALIDATE DATA
# ---------------------------
def validate_dataframe(df, name):
    if df.empty:
        raise ValueError(f"{name} is empty")

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{name} must have text and label columns")

    if df["text"].isna().any():
        raise ValueError(f"{name} contains null text values")

    if df["label"].isna().any():
        raise ValueError(f"{name} contains null label values")

    if (df["text"].str.strip() == "").any():
        raise ValueError(f"{name} contains blank text")

    invalid = sorted(set(df["label"]) - set(LABEL_ORDER))
    if invalid:
        raise ValueError(f"{name} contains invalid labels: {invalid}")

    print(f"\n{name} label distribution:")
    print(df["label"].value_counts())


# ---------------------------
# MAIN
# ---------------------------
def main():
    print("Loading datasets...")
    print("TRAIN_FILE:", TRAIN_FILE)
    print("TEST_FILE:", TEST_FILE)

    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)

    validate_dataframe(train_df, "train_df")
    validate_dataframe(test_df, "test_df")

    X_train_text = train_df["text"]
    X_test_text = test_df["text"]
    y_train = train_df["label"]
    y_test = test_df["label"]

    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("Training model...")
    model = LinearSVC(C=1.0)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("\n📊 TRAIN ACCURACY:", round(train_acc * 100, 2), "%")
    print("📊 TEST ACCURACY:", round(test_acc * 100, 2), "%")

    print("\n📄 Classification Report:\n")
    print(classification_report(y_test, test_pred, labels=LABEL_ORDER, zero_division=0))

    cm = confusion_matrix(y_test, test_pred, labels=LABEL_ORDER)
    print("\n📊 Confusion Matrix:\n", cm)

    results_df = pd.DataFrame({
        "Text": X_test_text.values,
        "Actual": y_test.values,
        "Predicted": test_pred
    })

    print("\n🔍 Sample Predictions:\n")
    print(results_df.head(5))

    sample_index = 0
    sample_text = X_test_text.iloc[sample_index]
    actual_label = y_test.iloc[sample_index]
    sample_vec = vectorizer.transform([sample_text])
    pred_label = model.predict(sample_vec)[0]

    print("\n🎯 FINAL PREDICTION RESULT")
    print("--------------------------------------------------")
    print("📝 Text:", sample_text)
    print("✅ Actual Emotion:", actual_label)
    print("🤖 Predicted Emotion:", pred_label)

    if pred_label == actual_label:
        print("✔️ Prediction Status: CORRECT")
    else:
        print("❌ Prediction Status: WRONG")

    with open(OUTPUT_DIR / "emotion_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(OUTPUT_DIR / "emotion_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    metrics = {
        "train_accuracy": round(float(train_acc), 4),
        "validation_accuracy": round(float(test_acc), 4),
        "labels": LABEL_ORDER,
        "confusion_matrix": cm.tolist()
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    results_df.to_csv(OUTPUT_DIR / "prediction_samples.csv", index=False)

    print("\n🔥 Model trained and saved successfully!")
    print("Saved files in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()