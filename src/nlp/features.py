import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_demo(path="data/threat_text_demo.csv"):
    df = pd.read_csv(path)  # columns: text,label (0/1)
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(int).values
    return X, y

def make_features(X):
    vec = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    Xv = vec.fit_transform(X)
    return Xv, vec
