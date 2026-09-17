"""
retrieval.py
------------
Builds an offline TF-IDF index over historical (customer_text -> agent_reply)
pairs and retrieves the top-k most similar past customer messages for a new
incoming message. This is the "grounded in how the brand has historically
resolved similar issues" mechanism (deliverable #2).

We use TF-IDF + cosine similarity rather than a downloaded embedding model
because (a) it needs zero network access / model downloads, keeping the
15-minute reproducibility promise honest, and (b) on short, templated
customer-support tweets, TF-IDF over word 1-2 grams is a strong, well
understood, easily-explained baseline for near-duplicate complaint matching.
report.md's failure analysis discusses where this breaks down (paraphrases
with no lexical overlap).
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoricalCaseIndex:
    def __init__(self, pairs_df: pd.DataFrame):
        self.pairs = pairs_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.6, stop_words="english"
        )
        self.matrix = self.vectorizer.fit_transform(self.pairs["customer_text"].fillna(""))

    def query(self, text: str, k: int = 3, exclude_id: str = None):
        vec = self.vectorizer.transform([text])
        sims = cosine_similarity(vec, self.matrix)[0]
        order = np.argsort(-sims)
        results = []
        for idx in order:
            row = self.pairs.iloc[idx]
            if exclude_id is not None and row.get("customer_tweet_id") == exclude_id:
                continue
            if sims[idx] <= 0:
                break
            results.append({
                "similarity": float(sims[idx]),
                "customer_text": row["customer_text"],
                "agent_reply": row["agent_reply"],
                "customer_tweet_id": row["customer_tweet_id"],
            })
            if len(results) >= k:
                break
        return results
