
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# regex, izvlacenje url i ip adresas
URL_RE = re.compile(r"https?://[^\s'\"<>]+|www\.[^\s'\"<>]+", re.IGNORECASE)
IP_RE  = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}", re.IGNORECASE)

def extract_urls(text):
    if not isinstance(text, str):
        return []
    return URL_RE.findall(text)

# transformer, daje url feature
class UrlFeatureExtractor(BaseEstimator, TransformerMixin):
    """num i tekst osobine u data frame."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["urls"] = df["body"].apply(extract_urls)
        df["num_urls"] = df["urls"].apply(len)
        df["num_ip_urls"] = df["urls"].apply(lambda lst: sum(1 for u in lst if IP_RE.search(u)))
        df["avg_url_len"] = df["urls"].apply(lambda lst: np.mean([len(u) for u in lst]) if lst else 0)
        df["max_url_len"] = df["urls"].apply(lambda lst: np.max([len(u) for u in lst]) if lst else 0)
        df["urls_concat"] = df["urls"].apply(lambda lst: " ".join(lst))
        df["combined_text"] = (
            df["subject"].fillna("") + " " +
            df["body"].fillna("") + " " +
            df["urls_concat"].fillna("")

        )
        
        return df[["combined_text", "num_urls", "num_ip_urls", "avg_url_len", "max_url_len"]]

# main function
def build_text_url_featurizer():
    """pipeline sa tf-idf + skaliranim url num osobinama."""
    text_vec = TfidfVectorizer(
        sublinear_tf=True,
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=3,
        max_df=0.9
    )

    preproc = ColumnTransformer(transformers=[
        ("tfidf", text_vec, "combined_text"),
        ("num", StandardScaler(), ["num_urls", "num_ip_urls", "avg_url_len", "max_url_len"])
    ])

    pipe = Pipeline([
        ("url_features", UrlFeatureExtractor()),
        ("preproc", preproc)
    ])

    return pipe
