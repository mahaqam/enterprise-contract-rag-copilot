from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.retrieval import build_retriever

app = FastAPI(title="Enterprise Contract Retrieval Copilot")
_retriever = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5


def get_retriever():
    global _retriever
    if _retriever is None:
        path = os.getenv("CONTRACT_DATA", "master_clauses.csv")
        if not Path(path).exists():
            raise HTTPException(
                status_code=503,
                detail="Dataset not found. Set CONTRACT_DATA to master_clauses.csv.",
            )
        _retriever = build_retriever(path)
    return _retriever


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(payload: SearchRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    k = min(max(payload.k, 1), 20)
    return get_retriever().grounded_response(payload.query, k=k)
