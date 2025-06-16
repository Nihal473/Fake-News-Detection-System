# 📰 Fake News Detection System

A machine learning-powered web application that detects whether a news article is **real** or **fake** based on its content. Built using Python, Scikit-learn, and Flask.

---

## 🚀 Demo

Paste your news article text into the web interface and find out if it's likely **Real** or **Fake**.

> Want a hosted version? You can deploy this on **Render**, **Railway**, or **Heroku** easily.

---

## 🧠 Features

- Binary classification of news as **real** or **fake**
- Trained on labeled news articles from Kaggle
- Uses **TF-IDF + Logistic Regression** (baseline)
- Optional upgrade to **BERT** for more accuracy
- Simple web interface with Flask
- Extensible and beginner-friendly structure

---

## 📂 Dataset

Used: [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

- `Fake.csv`: Fake news articles  
- `True.csv`: Real news articles  
- Each article includes: `title`, `text`, `subject`, `date`

---

## 🛠 Tech Stack

| Tool / Framework | Usage |
|------------------|-------|
| Python           | Programming language |
| Pandas, NumPy    | Data manipulation |
| Scikit-learn     | Model training |
| Flask            | Web app |
| TfidfVectorizer  | Text vectorization |
| LogisticRegression | Classification model |
| HTML/CSS         | Frontend (basic) |
| (Optional) BERT  | Deep NLP model (via HuggingFace Transformers) |

---

## 📦 Installation

### Clone the Repo

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

### Install Requirements

```bash
pip install -r requirements.txt
```

> Make sure Python 3.8+ is installed.

---

## ▶️ Running the App

### Step 1: Train the model

```bash
python train_model.py
```

### Step 2: Run the Flask App

```bash
python app.py
```

Go to `http://127.0.0.1:5000` in your browser.

---

## 🧪 Model Performance

Baseline Model: **TF-IDF + Logistic Regression**

| Metric | Score |
|--------|-------|
| Accuracy | ~95% |
| Precision | High |
| Recall | High |
| F1-Score | Balanced |

*Optional*: Try `bert_model.py` to fine-tune a transformer for even better accuracy.

---

## 💡 Future Enhancements

- ✅ Add BERT transformer model support  
- 🌐 Deploy to the web (Render/Heroku/Vercel)  
- 📊 Add SHAP/LIME for model explainability  
- 🎨 Improve UI with Streamlit or React  
- 🔐 Add user authentication for personalized history

---

## 🤝 Contributions

Pull requests are welcome! If you find a bug or would like to suggest an improvement, please feel free to open an issue.

---

## 📜 License

This project is licensed under the MIT License.

---

## ✨ Author

Built by [Nihal473](https://github.com/Nihal473) — passionate about machine learning and AI for good.

