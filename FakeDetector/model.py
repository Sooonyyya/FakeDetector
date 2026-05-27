from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def train_and_evaluate(X, y, dataset_name=""):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if dataset_name == "Reviews":
        vectorizer = TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 4),
            min_df=2,
            sublinear_tf=True
        )
    else:
        vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 3)
        )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": LinearSVC(random_state=42)
    }

    results = {}
    best_model = None
    best_f1 = 0

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "model": model
        }

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

        print(f"\n=== {name} ({dataset_name}) ===")
        print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"]
        )
        plt.title(f"Confusion Matrix - {name} ({dataset_name})")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(f'cm_{name.replace(" ", "_")}_{dataset_name}.png')
        plt.show()

    plot_comparison(results, dataset_name)

    return results, vectorizer, best_model


def plot_comparison(results, dataset_name):
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results.keys())

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_name in enumerate(model_names):
        values = [results[model_name][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=model_name)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison - {dataset_name}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-score"])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"comparison_{dataset_name}.png")
    plt.show()