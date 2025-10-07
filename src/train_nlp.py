# src/train_nlp.py
import json, os, argparse
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from joblib import dump

def main(epochs: int = 1):
    os.makedirs("models", exist_ok=True)
    X, y = make_classification(n_samples=600, n_features=20, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    f1 = f1_score(yte, pred)
    dump(clf, "models/bert_classifier.pkl")
    json.dump({"f1": float(f1)}, open("models/nlp_metrics.json", "w"))
    print("NLP training (stub) complete. F1=", f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    main(args.epochs)

