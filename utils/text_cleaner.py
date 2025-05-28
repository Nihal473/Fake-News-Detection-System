
# Helper functions for text preprocessing
# utils/text_cleaner.py
import re
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem.porter import PorterStemmer

stop_words = set(nltk_stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return " ".join(text)