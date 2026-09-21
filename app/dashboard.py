from pathlib import Path
import os
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval import HybridClauseRetriever, build_retriever

st.set_page_config(
    page_title="Enterprise Contract Retrieval Copilot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Enterprise Contract Retrieval Copilot")
st.caption(
    "Grounded contract-clause retrieval with transparent hybrid ranking and source-linked evidence."
)

FULL_DATA_PATH = Path(os.getenv("CONTRACT_DATA", ROOT / "master_clauses.csv"))

# Small deployment fallback built from real snippets in the uploaded CUAD-style dataset.
# It keeps the public demo functional without committing the full source dataset.
DEMO_ROWS = [
    {
        "filename": "CybergyHoldingsInc_20140520_10-Q_EX-10.27_Affiliate Agreement.pdf",
        "category": "Agreement Date",
        "text": "8th day of May 2014",
    },
    {
        "filename": "CybergyHoldingsInc_20140520_10-Q_EX-10.27_Affiliate Agreement.pdf",
        "category": "Anti-Assignment",
        "text": "MA may not assign, sell, lease or otherwise transfer in whole or in part any of the rights granted pursuant to this Agreement without prior written approval of Company.",
    },
    {
        "filename": "CybergyHoldingsInc_20140520_10-Q_EX-10.27_Affiliate Agreement.pdf",
        "category": "Audit Rights",
        "text": "MA shall keep accurate records of the sales of the Technology and Maintenance and shall make these records available for review by a representative of Company.",
    },
    {
        "filename": "GopageCorp_20140221_10-K_EX-10.1_Content License Agreement.pdf",
        "category": "Change Of Control",
        "text": "Any merger, consolidation or reorganization involving Licensee will be deemed to be a transfer of rights for which Licensor's prior written consent is required.",
    },
    {
        "filename": "FulucaiProductionsLtd_20131223_10-Q_EX-10.9_Content License Agreement.pdf",
        "category": "Exclusivity",
        "text": "During the License Term, Producer agrees that ConvergTV has the exclusive right to exercise the rights granted to it under this Agreement within the Licensed Territory.",
    },
    {
        "filename": "CybergyHoldingsInc_20140520_10-Q_EX-10.27_Affiliate Agreement.pdf",
        "category": "Governing Law",
        "text": "This Agreement is accepted by Company in the State of Nevada and shall be governed by and construed in accordance with the laws thereof.",
    },
    {
        "filename": "MusclepharmCorp_20170208_10-KA_EX-10.38_Co-Branding Agreement.pdf",
        "category": "Insurance",
        "text": "MusclePharm shall obtain and maintain a commercial general liability insurance policy including coverage for contractual liability, product liability, personal injury liability, and advertiser's liability.",
    },
    {
        "filename": "CybergyHoldingsInc_20140520_10-Q_EX-10.27_Affiliate Agreement.pdf",
        "category": "License Grant",
        "text": "Company hereby grants MA, during the term of this Agreement, the right to use Company trade names, trademarks or service marks on Technology or in advertising or promotion relating directly to these products.",
    },
    {
        "filename": "CybergyHoldingsInc_20140520_10-Q_EX-10.27_Affiliate Agreement.pdf",
        "category": "Minimum Commitment",
        "text": "MA commits to purchase a minimum of 100 Units in aggregate within the Territory within the first six months of the term of this Agreement.",
    },
    {
        "filename": "PareteumCorp_20081001_8-K_EX-99.1_Hosting Agreement.pdf",
        "category": "No-Solicit Of Employees",
        "text": "Without the prior written consent of the other Party, a Party shall not solicit any employee of the other Party while this Agreement is in force and for a one-year period after termination.",
    },
    {
        "filename": "EdietsComInc_20001030_10QSB_EX-10.4_Co-Branding Agreement.pdf",
        "category": "Non-Compete",
        "text": "During the Term, Women.com will not enter into a relationship with a Competitive Company involving specified integrated promotional arrangements.",
    },
    {
        "filename": "EuromediaHoldingsCorp_20070215_10SB12G_EX-10.B(01)_Content License Agreement.pdf",
        "category": "Most Favored Nation",
        "text": "If Licensor grants another VOD or PPV service provider an earlier availability date, Licensor shall also grant Rogers the right to distribute and exhibit the Licensed Program on that earlier date.",
    },
]

@st.cache_resource
def load_retriever():
    if FULL_DATA_PATH.exists():
        return build_retriever(FULL_DATA_PATH), "full"
    demo = pd.DataFrame(DEMO_ROWS)
    return HybridClauseRetriever(demo), "demo"

retriever, mode = load_retriever()

with st.sidebar:
    st.header("About this demo")
    st.write(
        "The live site demonstrates the retrieval and grounding layer from the portfolio project."
    )
    if mode == "full":
        st.success("Full project dataset loaded")
    else:
        st.info("Public demo mode: uses a small subset of real uploaded clause snippets.")
    st.markdown("[View source on GitHub](https://github.com/mahaqam/enterprise-contract-rag-copilot)")

left, right = st.columns([2, 1])
with left:
    query = st.text_input(
        "Ask for a contract clause",
        value="termination or assignment restrictions",
        help="Try: governing law, non-compete, insurance, audit rights, exclusivity",
    )
with right:
    k = st.slider("Evidence results", 1, 8, 5)

if st.button("Retrieve evidence", type="primary", use_container_width=True) and query.strip():
    results = retriever.search(query, k=min(k, len(retriever.corpus)))
    st.subheader("Ranked evidence")
    for i, result in enumerate(results, 1):
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{i}. {result.category}**")
                st.caption(result.filename)
            with c2:
                st.metric("Similarity", f"{result.score:.3f}")
            st.write(result.text)

st.divider()
metrics = st.columns(4)
metrics[0].metric("Contracts analysed", "510")
metrics[1].metric("Clause snippets", "13,101")
metrics[2].metric("Verified Hit@5", "82.9%")
metrics[3].metric("Verified MRR", "0.703")

st.markdown("### How it works")
st.write(
    "The benchmark combines word-level TF-IDF and character-level TF-IDF similarity, then ranks source clauses for a query. "
    "The verified project benchmark used 41 expert-defined clause categories."
)
st.warning(
    "Scope: this is an extractive retrieval/grounding prototype, not a production LLM deployment. "
    "The uploaded project dataset contains annotated clause snippets rather than raw contract PDFs."
)
