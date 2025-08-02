# predict.py
import joblib
import pandas as pd
import re
import numpy as np
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Load the Vectorizer
try:
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    print("Error: vectorizer.pkl not found. Please run preprocess.py first.")
    exit(1)

# Load the Latest Saved Best Model
import glob
model_files = sorted(glob.glob("best_model_*.pkl"), reverse=True)

if not model_files:
    print("No trained models found. Please run train.py first.")
    exit(1)

model_path = model_files[0]
model = joblib.load(model_path)
print(f"Loaded model: {model_path}")

# Function to Clean and Preprocess Input Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# Main Prediction Loop
print("\nWelcome to The Office - Character Predictor!")
print("Enter a sentence and the model will predict who is most likely to have said it.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Your sentence: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    cleaned = clean_text(user_input)
    transformed = vectorizer.transform([cleaned])

    prediction = model.predict(transformed)[0]
    probabilities = model.predict_proba(transformed)[0]

    # Display Result
    print(f"\nPredicted Character: **{prediction}**")

    # Show Top 3 Probabilities
    top3_idx = np.argsort(probabilities)[::-1][:3]
    top3_labels = model.classes_[top3_idx]
    top3_probs = probabilities[top3_idx]

    print("\nTop 3 predictions:")
    for label, prob in zip(top3_labels, top3_probs):
        print(f"{label}: {prob:.2%}")

    print("\n" + "-"*40 + "\n")
