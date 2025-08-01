# train.py
import os
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import sklearn

# Timestamp for Output Files

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Output folders

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

report_file = f"reports/evaluation_report_{timestamp}.txt"

# Log Library Versions

with open(report_file, "w") as f:
    f.write(f"Library Versions:\n")
    f.write(f"Pandas: {pd.__version__}\n")
    f.write(f"NumPy: {np.__version__}\n")
    f.write(f"Scikit-learn: {sklearn.__version__}\n\n")

# Load Data

print("Loading data")
try:
    X = load_npz('preprocessed/X_preprocessed.npz')
    y = pd.read_csv('preprocessed/Y_preprocessed.csv.gz', compression='gzip')['Character']
    if X.shape[0] != y.shape[0]:
        raise ValueError("Mismatch between X and y sample sizes.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
print("Class distribution:\n", y.value_counts(normalize=True))

# Split Data

print("\nSplitting into train/test sets")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Split complete.")

# Compute Class Weights

print("\nCalculating class weights")
classes = np.unique(y)
weights = compute_class_weight("balanced", classes=classes, y=y)
class_weights = dict(zip(classes, weights))
print("Class weights calculated.")

# Define Models

print("\nDefining models and hyperparameter grids")

lr = LogisticRegression(solver="saga", max_iter=1000, class_weight=class_weights, n_jobs=-1, verbose=1)
lr_params = [
    {"C": np.logspace(-2, 2, 5), "penalty": ["l1", "l2"]},
    {"C": np.logspace(-2, 2, 5), "penalty": ["elasticnet"], "l1_ratio": [0.1, 0.5, 0.9]}
]

nb = MultinomialNB()
nb_params = {"alpha": [0.1, 0.5, 1.0]}

rf = RandomForestClassifier(class_weight=class_weights, n_jobs=-1, n_estimators=100)
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

models = {
    "LR": (lr, lr_params),
    "NB": (nb, nb_params),
    "RF": (rf, rf_params)
}

# Training and Evaluation

print("\nStarting training with hyperparameter tuning\n")

best_model = None
best_score = -1
best_name = ""

for name, (model, param_grid) in tqdm(models.items(), desc="Training Models"):
    print(f"\nTuning {name}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        model, param_grid, n_iter=5, scoring="balanced_accuracy",
        n_jobs=-1, cv=cv, verbose=1, random_state=42
    )

    search.fit(X_train, y_train)
    best_estimator = search.best_estimator_

    print(f"\nBest parameters for {name}: {search.best_params_}")
    print(f"Best CV Balanced Accuracy: {search.best_score_:.4f}")

    # Final Evaluation

    y_pred = best_estimator.predict(X_test)
    acc = balanced_accuracy_score(y_test, y_pred)

    print(f"Test Balanced Accuracy for {name}: {acc:.4f}")

    # Save Classification Report

    pd.DataFrame({"true": y_test.values, "predicted": y_pred}).to_csv(
        f"results/predictions_{name}_{timestamp}.csv", index=False
    )

    report = classification_report(y_test, y_pred)
    with open(report_file, "a") as f:
        f.write(f"\n{'='*30}\nModel: {name}\nBest Params: {search.best_params_}\n"
                f"CV Balanced Accuracy: {search.best_score_:.4f}\nTest Balanced Accuracy: {acc:.4f}\n{report}\n")

    # Save Confusion Matrix

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_estimator.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"results/confusion_matrix_{name}_{timestamp}.png")
    plt.close()

    # Feature Importance for Random Forest

    if name == "RF":
        importances = pd.DataFrame({
            "Feature": range(X.shape[1]),
            "Importance": best_estimator.feature_importances_
        }).sort_values("Importance", ascending=False)
        importances.to_csv(f"results/feature_importances_{name}_{timestamp}.csv", index=False)

    # Save Model

    joblib.dump(best_estimator, f"models/model_{name}_{timestamp}.pkl")

    if acc > best_score:
        best_score = acc
        best_model = best_estimator
        best_name = name

# Finalize

joblib.dump(best_model, f"models/best_model_{timestamp}.pkl")
print(f"\nBest model: {best_name} (Balanced Accuracy: {best_score:.4f})")
print(f"Saved as models/best_model_{timestamp}.pkl")
print(f"Evaluation report saved to {report_file}")
