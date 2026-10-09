"""Real UCI SMS Spam Collection NLP baseline; no fabricated security messages.

The UCI labels are HAM versus SMS SPAM, not phishing, malware or attacks.
Data CC BY 4.0, Almeida and Hidalgo (2011), DOI 10.24432/C5CC84.
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import joblib
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SOURCE = "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip"
CITATION = "Almeida, T. & Hidalgo, J. (2011). UCI SMS Spam Collection. https://doi.org/10.24432/C5CC84"


def read_uci_spam(source=SOURCE, session=None):
    """Download genuine labeled messages; never silently substitute fake samples."""
    r = (session or requests).get(source, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as archive:
        candidates = [n for n in archive.namelist() if n.rsplit("/", 1)[-1].lower() == "smsspamcollection"]
        if not candidates:
            raise ValueError("UCI archive lacks SMSSpamCollection")
        content = archive.read(candidates[0]).decode("latin1")
    data = []
    for line in content.splitlines():
        if not line.strip():
            continue
        label, sep, message = line.partition("\t")
        if not sep or label not in {"ham", "spam"} or not message.strip():
            raise ValueError("UCI source contains malformed label or message")
        data.append((label, message))
    if len(data) < 5000 or len(set(row[0] for row in data)) != 2:
        raise ValueError("The original labeled SMS corpus is incomplete")
    return pd.DataFrame(data, columns=["label", "text"])


def train_real_nlp():
    df = read_uci_spam()
    # Identical text can occur multiple times. Deduplicate before splitting.
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=15000)),
        ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
    ])
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    result = {
        "source": SOURCE,
        "citation": CITATION,
        "license": "CC BY 4.0",
        "task": "SMS spam / ham classification (NOT malware, phishing or cyberattacks)",
        "n_deduplicated": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "heldout_spam_f1": float(f1_score(y_test, pred, pos_label="spam")),
        "heldout_report": classification_report(y_test, pred, output_dict=True),
    }
    return clf, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="models")
    args = ap.parse_args()
    model, result = train_real_nlp()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "uci_sms_spam_baseline.joblib")
    (out / "nlp_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "heldout_report"}, indent=2))


if __name__ == "__main__":
    main()
