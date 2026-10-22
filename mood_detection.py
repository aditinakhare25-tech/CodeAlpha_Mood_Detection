import joblib

model = joblib.load("mood_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

text = input("Enter a sentence: ")
prediction = model.predict(vectorizer.transform([text]))[0]

print("Detected mood:", prediction)
