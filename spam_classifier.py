import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "text": [
        "Win a free iPhone now",
        "Hey, are we meeting today?",
        "Limited offer, click now",
        "Let's have lunch tomorrow",
        "Congratulations, you won a prize"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam"]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Custom message
sample_message = ["Free money offer just for you"]
sample_vector = vectorizer.transform(sample_message)
prediction = model.predict(sample_vector)

print("Sample Message Prediction:", prediction[0])

