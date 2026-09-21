# Enterprise Contract RAG Copilot

[Live Demo](https://enterprise-contract-rag-copilot.streamlit.app/)

Grounded contract-clause retrieval prototype built from the CUAD-style `master_clauses.csv` dataset.

## Dataset
- 510 contracts
- 41 clause categories
- 13,101 non-empty clause snippets
- 1,955 duplicate snippets identified during validation

## What the project does
The project builds a searchable clause corpus and benchmarks a hybrid retrieval pipeline using word-level TF-IDF and character-level TF-IDF similarity. It returns source contract filenames, clause categories, ranked evidence snippets, and an extractive grounded response. The architecture is intentionally transparent and evaluation-first; it does not claim a production LLM deployment.

## Verified retrieval benchmark
Using one category-name query for each of the 41 CUAD clause types:
- Hit@1: **61.0%**
- Hit@3: **78.0%**
- Hit@5: **82.9%**
- Hit@10: **85.4%**
- MRR: **0.703**

The benchmark uses `0.7 * word TF-IDF + 0.3 * character TF-IDF` cosine similarity. Results are in `results/`.

## Files
- `src/retrieval.py` — corpus loading, normalization, hybrid retrieval, grounded response
- `src/evaluate.py` — reproducible 41-query retrieval benchmark
- `api/app.py` — FastAPI search endpoint
- `app/dashboard.py` — Streamlit demo
- `results/metrics.json` — verified summary metrics
- `results/per_query.csv` — per-clause-type ranking results

## Run
```bash
pip install -r requirements.txt
python src/evaluate.py master_clauses.csv
uvicorn api.app:app --reload
streamlit run app/dashboard.py
```

## Limitations
- `master_clauses.csv` contains annotated clause snippets rather than full raw contracts, so this project benchmarks retrieval over extracted evidence, not PDF parsing.
- Query evaluation uses clause-category names, so it is a controlled retrieval benchmark rather than a production user-query benchmark.
- The current grounded response is extractive. A production GenAI layer should add an LLM only with citation checks, refusal/fallback behavior, regression evaluation, privacy controls, and human review.
