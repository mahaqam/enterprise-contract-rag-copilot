from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from retrieval import HybridClauseRetriever, load_clause_corpus


def evaluate(csv_path: str | Path, output_dir: str | Path = "results") -> dict:
    source = pd.read_csv(csv_path)
    categories = [c for c in source.columns if c != "Filename" and "Answer" not in c]
    corpus = load_clause_corpus(csv_path)
    retriever = HybridClauseRetriever(corpus, word_weight=0.7)

    rows = []
    reciprocal_ranks = []
    hits = {1: [], 3: [], 5: [], 10: []}
    labels = corpus["category"].to_numpy()

    for category in categories:
        scores = retriever.score(category)
        order = np.argsort(-scores)
        relevant_positions = np.where(labels[order] == category)[0]
        rank = int(relevant_positions[0] + 1) if len(relevant_positions) else None
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
        for k in hits:
            hits[k].append(rank is not None and rank <= k)
        top = int(order[0])
        rows.append({
            "query_category": category,
            "first_relevant_rank": rank,
            "top1_category": str(labels[top]),
            "top1_score": float(scores[top]),
        })

    metrics = {
        "contracts": int(source["Filename"].nunique()),
        "clause_categories": int(len(categories)),
        "clause_snippets": int(len(corpus)),
        "duplicate_clause_snippets": int(corpus["text"].duplicated().sum()),
        "mrr": float(np.mean(reciprocal_ranks)),
        **{f"hit@{k}": float(np.mean(v)) for k, v in hits.items()},
        "method": "0.7 word TF-IDF + 0.3 character TF-IDF cosine similarity",
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_query.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    evaluate(args.csv_path, args.output_dir)
