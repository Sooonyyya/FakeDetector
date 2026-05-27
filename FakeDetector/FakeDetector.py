import pandas as pd
import pickle

from preprocessing import preprocess_text, preprocess_review_text
from model import train_and_evaluate


def load_news_data():
    print("Loading news dataset...")

    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")

    fake["label"] = 1
    real["label"] = 0

    df = pd.concat([fake, real], ignore_index=True)
    df = df[["title", "text", "label"]].dropna()
    df["content"] = df["title"] + " " + df["text"]

    print(
        f"Total: {len(df)} | "
        f"Fake: {df['label'].sum()} | "
        f"Real: {(df['label'] == 0).sum()}"
    )

    return df


def load_reviews_data():
    print("\nLoading reviews dataset...")

    df = pd.read_csv("data/fake_reviews_fixed.csv")

    df = df.rename(columns={"text_": "content", "label": "label"})
    df = df[["content", "label"]].dropna()

    # CG = fake, OR = real
    df["label"] = (df["label"] == "CG").astype(int)

    print(
        f"Total: {len(df)} | "
        f"Fake: {df['label'].sum()} | "
        f"Real: {(df['label'] == 0).sum()}"
    )

    return df


def save_model(vectorizer, model, filename):
    with open(filename, "wb") as f:
        pickle.dump((vectorizer, model), f)

    print(f"Model saved to {filename}")


def load_model(filename):
    with open(filename, "rb") as f:
        vectorizer, model = pickle.load(f)

    print(f"Model loaded from {filename}")

    return vectorizer, model


def predict_text(text, vectorizer, model, mode="news"):
    if mode == "review":
        processed = preprocess_review_text(text)
    else:
        processed = preprocess_text(text)

    tfidf = vectorizer.transform([processed])
    prediction = model.predict(tfidf)[0]

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(tfidf)[0]
        confidence = int(max(proba) * 100)
    elif hasattr(model, 'decision_function'):
        score = abs(model.decision_function(tfidf)[0])
        confidence = min(int(50 + score * 15), 99)
    else:
        confidence = None

    result = "FAKE" if prediction == 1 else "REAL"
    return result, confidence


def interactive_mode(news_vec, news_model, reviews_vec, reviews_model):
    print("\n" + "=" * 60)
    print("INTERACTIVE PREDICTION MODE")
    print("Type any news or review text to classify it.")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        print("\n1 — News")
        print("2 — Review")
        print("exit — Quit")

        choice = input("Your choice: ").strip().lower()

        if choice == "exit":
            print("Goodbye!")
            break

        elif choice == "1":
            text = input("Enter news text: ").strip()

            if text:
                result, confidence = predict_text(text, news_vec, news_model, mode="news")
                conf_text = f" (Confidence: {confidence}%)" if confidence else ""
                print(f"\nResult: >>> {result} <<<{conf_text}")

        elif choice == "2":
            text = input("Enter review text: ").strip()

            if text:
                result, confidence = predict_text(text, reviews_vec, reviews_model, mode="review")
                conf_text = f" (Confidence: {confidence}%)" if confidence else ""
                print(f"\nResult: >>> {result} <<<{conf_text}")

        else:
            print("Invalid choice. Please enter 1, 2, or exit.")


def main():
    df_news = load_news_data()

    print("Preprocessing news text...")
    df_news["processed"] = df_news["content"].apply(preprocess_text)

    print("Training models for news...")
    news_results, news_vec, news_model = train_and_evaluate(
        df_news["processed"],
        df_news["label"],
        dataset_name="News"
    )

    save_model(news_vec, news_model, "model_news.pkl")

    df_reviews = load_reviews_data()

    print("Preprocessing review text...")
    df_reviews["processed"] = df_reviews["content"].apply(preprocess_review_text)

    print("Training models for reviews...")
    reviews_results, reviews_vec, reviews_model = train_and_evaluate(
        df_reviews["processed"],
        df_reviews["label"],
        dataset_name="Reviews"
    )

    save_model(reviews_vec, reviews_model, "model_reviews.pkl")

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    for dataset_name, results in [
        ("News", news_results),
        ("Reviews", reviews_results)
    ]:
        print(f"\n{dataset_name}:")
        print(
            f"{'Model':<25} "
            f"{'Accuracy':>10} "
            f"{'Precision':>10} "
            f"{'Recall':>10} "
            f"{'F1':>10}"
        )
        print("-" * 65)

        for model_name, metrics in results.items():
            print(
                f"{model_name:<25} "
                f"{metrics['accuracy']:>10.4f} "
                f"{metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f} "
                f"{metrics['f1']:>10.4f}"
            )

    interactive_mode(news_vec, news_model, reviews_vec, reviews_model)


if __name__ == "__main__":
    main()