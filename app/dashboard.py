from pathlib import Path
import os
import streamlit as st

from src.retrieval import build_retriever

st.set_page_config(page_title="Enterprise Contract Retrieval Copilot", layout="wide")
st.title("Enterprise Contract RAG Copilot")
st.caption("Grounded contract-clause retrieval with transparent hybrid ranking and source citations.")

path = os.getenv("CONTRACT_DATA", "master_clauses.csv")
if not Path(path).exists():
    st.warning("Dataset not found in the deployment. Add `master_clauses.csv` to the app or set CONTRACT_DATA.")
    st.stop()

@st.cache_resource
def load_retriever():
    return build_retriever(path)

retriever = load_retriever()
query = st.text_input("Ask for a contract clause", value="termination for convenience")
k = st.slider("Evidence results", 1, 10, 5)
if st.button("Retrieve evidence", type="primary") and query.strip():
    results = retriever.search(query, k=k)
    for i, r in enumerate(results, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {r.category}** — score `{r.score:.3f}`")
            st.caption(r.filename)
            st.write(r.text)

st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Contracts", "510")
c2.metric("Clause snippets", "13,101")
c3.metric("Hit@5", "82.9%")
c4.metric("MRR", "0.703")
st.info("This version returns extractive grounded evidence. It does not claim production LLM generation or full-contract PDF parsing.")
