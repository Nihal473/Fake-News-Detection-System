# Flask app main file
# app.py
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        news_text = request.form["news"]
        vectorized_text = vectorizer.transform([news_text])
        result = model.predict(vectorized_text)[0]
        prediction = "Real News ✅" if result == 1 else "Fake News ❌"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's port or default to 5000 locally
    app.run(host="0.0.0.0", port=port, debug=True)