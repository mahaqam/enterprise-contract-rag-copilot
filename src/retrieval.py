from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SearchResult:
    filename: str
    category: str
    text: str
    score: float


def _safe_list(value) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()]


def load_clause_corpus(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    evidence_cols = [
        c for c in df.columns
        if c != "Filename" and "Answer" not in c
    ]
    rows = []
    for _, row in df.iterrows():
        for category in evidence_cols:
            for text in _safe_list(row[category]):
                rows.append({
                    "filename": row["Filename"],
                    "category": category,
                    "text": text,
                })
    return pd.DataFrame(rows)


class HybridClauseRetriever:
    """Transparent hybrid retriever using word and character TF-IDF."""

    def __init__(self, corpus: pd.DataFrame, word_weight: float = 0.7):
        if not 0 <= word_weight <= 1:
            raise ValueError("word_weight must be between 0 and 1")
        self.corpus = corpus.reset_index(drop=True).copy()
        self.word_weight = word_weight
        self.char_weight = 1.0 - word_weight

        self.word_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=60_000,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=60_000,
            sublinear_tf=True,
        )
        texts = self.corpus["text"].fillna("").tolist()
        self.word_matrix = self.word_vectorizer.fit_transform(texts)
        self.char_matrix = self.char_vectorizer.fit_transform(texts)

    def score(self, query: str) -> np.ndarray:
        q_word = self.word_vectorizer.transform([query])
        q_char = self.char_vectorizer.transform([query])
        word_scores = cosine_similarity(q_word, self.word_matrix)[0]
        char_scores = cosine_similarity(q_char, self.char_matrix)[0]
        return self.word_weight * word_scores + self.char_weight * char_scores

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        scores = self.score(query)
        order = np.argsort(-scores)[:k]
        out = []
        for idx in order:
            row = self.corpus.iloc[int(idx)]
            out.append(SearchResult(
                filename=str(row["filename"]),
                category=str(row["category"]),
                text=str(row["text"]),
                score=float(scores[idx]),
            ))
        return out

    def grounded_response(self, query: str, k: int = 5) -> dict:
        results = self.search(query, k=k)
        return {
            "query": query,
            "answer_type": "extractive_grounded_response",
            "note": "No external LLM is required; the response returns ranked source evidence.",
            "evidence": [r.__dict__ for r in results],
        }


def build_retriever(csv_path: str | Path) -> HybridClauseRetriever:
    return HybridClauseRetriever(load_clause_corpus(csv_path))
