# Script to train the ML model
# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# Load data
true_news = pd.read_csv("data/True.csv")
fake_news = pd.read_csv("data/Fake.csv")

# Add labels
true_news["label"] = 1
fake_news["label"] = 0

# Combine datasets
data = pd.concat([true_news, fake_news], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

# Features and Labels
X = data["text"]
y = data["label"]

# Vectorization
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
X_tfidf = tfidf.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
