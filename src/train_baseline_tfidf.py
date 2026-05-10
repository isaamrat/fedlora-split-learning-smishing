"""
train_baseline_tfidf.py
E0-A: TF-IDF + Logistic Regression
E0-B: TF-IDF + Linear SVM

Reads:  data/splits/train.csv, val.csv, test_clean.csv
Saves:  reports/results_tfidf_lr.csv
        reports/results_tfidf_svm.csv
        reports/figures/confusion_matrix_tfidf_*.png
        models/tfidf_lr.pkl
        models/tfidf_svm.pkl

Usage:
  python src/train_baseline_tfidf.py
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# local imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import (load_split, encode_labels, LABEL2ID, ID2LABEL,
                   MODELS_DIR, set_seed)
from evaluate import compute_metrics, report_metrics

MODELS_DIR.mkdir(parents=True, exist_ok=True)
set_seed(42)

TFIDF_PARAMS = dict(
    max_features   = 60_000,
    ngram_range    = (1, 2),
    sublinear_tf   = True,
    strip_accents  = "unicode",
    analyzer       = "word",
    token_pattern  = r"\w{1,}",
    min_df         = 2,
)


def load_data():
    train = load_split("train")
    val   = load_split("val")
    test  = load_split("test_clean")  

    X_train = train["cleaned_text"].fillna("").values
    y_train = encode_labels(train["label"])

    X_val  = val["cleaned_text"].fillna("").values
    y_val  = encode_labels(val["label"])

    X_test = test["cleaned_text"].fillna("").values
    y_test = encode_labels(test["label"])

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_experiment(name, tag, clf, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n[{name}] Fitting pipeline...")
    clf.fit(X_train, y_train)

    print(f"[{name}] Evaluating on validation set...")
    y_val_pred = clf.predict(X_val)
    val_metrics = compute_metrics(y_val, y_val_pred)
    report_metrics(val_metrics, f"{name} (val)", f"{tag}_val",
                   y_val, y_val_pred,
                   extra={"split": "val", "n_features": TFIDF_PARAMS["max_features"]})

    print(f"[{name}] Evaluating on test set...")
    y_test_pred = clf.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred)
    report_metrics(test_metrics, f"{name} (test)", f"{tag}_test",
                   y_test, y_test_pred,
                   extra={"split": "test", "n_features": TFIDF_PARAMS["max_features"]})

    # Save model
    model_path = MODELS_DIR / f"{tag}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"  Model saved -> {model_path.name}")

    return test_metrics


def main():
    print("=" * 55)
    print("  TF-IDF Baseline Training")
    print("=" * 55)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Class weights for sklearn
    classes = np.array(sorted(set(y_train)))
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    print(f"\nClass weights: {class_weight_dict}")

    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   LogisticRegression(
            C            = 1.0,
            max_iter     = 1000,
            class_weight = class_weight_dict,
            solver       = "lbfgs",
            random_state = 42,
        )),
    ])
    run_experiment("TF-IDF + Logistic Regression", "tfidf_lr",
                   lr_pipeline, X_train, y_train, X_val, y_val, X_test, y_test)

    svm_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   LinearSVC(
            C            = 0.5,
            max_iter     = 2000,
            class_weight = class_weight_dict,
            random_state = 42,
        )),
    ])
    run_experiment("TF-IDF + Linear SVM", "tfidf_svm",
                   svm_pipeline, X_train, y_train, X_val, y_val, X_test, y_test)

    print("\nBaseline training complete.")


if __name__ == "__main__":
    main()
