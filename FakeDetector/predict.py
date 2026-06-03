import pickle
from preprocessing import preprocess_text, preprocess_review_text

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def predict(text, vectorizer, model, mode="news"):
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

news_vec, news_model = load_model('model_news.pkl')
reviews_vec, reviews_model = load_model('model_reviews.pkl')

print("Models loaded. Ready to classify.\n")

while True:
    print("1 - News  |  2 - Review  |  exit - Quit")
    choice = input("Choice: ").strip().lower()

    if choice == "exit":
        break
    elif choice == "1":
        text = input("Enter news text: ").strip()
        if text:
            result, confidence = predict(text, news_vec, news_model, mode="news")
            conf_text = f" (Confidence: {confidence}%)" if confidence else ""
            print(f"Result: {result}{conf_text}\n")
    elif choice == "2":
        text = input("Enter review text: ").strip()
        if text:
            result, confidence = predict(text, reviews_vec, reviews_model, mode="review")
            conf_text = f" (Confidence: {confidence}%)" if confidence else ""
            print(f"Result: {result}{conf_text}\n")
    else:
        print("Invalid choice.\n")