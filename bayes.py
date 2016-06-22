import os

import numpy as np

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


class Document(object):
    labels = None
    text = None

    def __init__(self, labels, text, path):
        self.labels = labels
        self.text = text
        self.path = path


def read_corpora(in_path):
    _corpora = []
    for root, dirs, files in os.walk(in_path):
        for f in files:
            topic, subtopic = root.split("/")[-2:]
            if f.endswith(".txt"):
                path = os.path.join(root, f)

                with open(path, "r") as _in:
                    text = " ".join([item for item in _in.read().split("\n") if len(item) > 0])

                if len(text) > 1:
                    document = Document(subtopic, text, path)
                    _corpora.append(document)
    return _corpora


if __name__ == "__main__":
    IN_PATH = "/Users/charlie/shitbox/gensim-stuff/output"

    corpora = read_corpora(IN_PATH)

    _labels = [doc.labels for doc in corpora]
    _documents = [doc.text for doc in corpora]

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(_documents, _labels)

    classifier = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC())
    ])

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    print(metrics.classification_report(y_test, predictions))
    print(metrics.confusion_matrix(y_test, predictions))
