"""Intent classifier: the ML half of the intelligent automation.

TF-IDF on words and character n-grams (character n-grams survive typos and
Pidgin spellings) into a logistic regression. Small, fast, explainable, and
easy to retrain when the service team relabels a few hundred tickets.

The bot only acts on its own when confidence is high. Everything else goes to
a person, which is the right default for a regulated business.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

CONFIDENCE_THRESHOLD = 0.70


@dataclass
class Prediction:
    intent: str
    confidence: float
    second_intent: str
    second_confidence: float

    @property
    def auto(self) -> bool:
        return self.confidence >= CONFIDENCE_THRESHOLD


class IntentModel:
    def __init__(self) -> None:
        self.pipe = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(max_iter=3000, C=4.0)),
            ]
        )
        self.report: str = ""

    def fit(self, history: pd.DataFrame) -> "IntentModel":
        tr, te = train_test_split(history, test_size=0.2, random_state=1, stratify=history["intent"])
        self.pipe.fit(tr["text"], tr["intent"])
        self.report = classification_report(te["intent"], self.pipe.predict(te["text"]), digits=3)
        self.pipe.fit(history["text"], history["intent"])  # refit on everything once evaluated
        return self

    def predict(self, text: str) -> Prediction:
        proba = self.pipe.predict_proba([text])[0]
        order = np.argsort(-proba)
        classes = self.pipe.classes_
        return Prediction(classes[order[0]], float(proba[order[0]]), classes[order[1]], float(proba[order[1]]))
