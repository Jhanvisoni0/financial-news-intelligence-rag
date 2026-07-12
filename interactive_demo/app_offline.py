# interactive_demo/app_offline.py
# Fully offline version - reads from a local parquet cache (created once by
# export_once.py) instead of querying Databricks live. Works with no Azure/
# Databricks connection at all - only needs an OpenAI key for generation
# (retrieval works with no key needed).
#
# First run export_once.py while your cluster is running (one time only),
# then this app works forever after, independent of Databricks/Azure.
#
# Run with: streamlit run app_offline.py

import os
import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI

st.set_page_config(page_title="Financial RAG - Offline Demo", layout="wide")

# Resolve the cache file path relative to THIS SCRIPT's location, not the
# current working directory - Streamlit Cloud runs scripts with the repo
# root as the working directory, not the script's own folder, so a plain
# relative path like "gold_chunks_cache.parquet" fails to find the file
# even though it exists right next to this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SCRIPT_DIR, "gold_chunks_cache.parquet")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def load_cached_data():
    if not os.path.exists(CACHE_FILE):
        return None
    return pd.read_parquet(CACHE_FILE)


@st.cache_resource
def build_embedding_matrix(_df: pd.DataFrame, cache_key: str) -> np.ndarray:
    """
    Pre-compute a single, stacked, normalized embedding matrix ONCE, instead of
    converting each row's embedding from a Python list to a numpy array on
    every single query. The original per-query Python loop (iterating ~5,000+
    rows and calling np.array() on each, every time "Ask" was clicked) caused
    repeated memory churn that likely contributed to an out-of-memory crash
    (segfault) after the corpus grew from ~1,094 to ~5,000+ chunks (11 tickers).
    This vectorized approach computes the matrix once, cached across the whole
    session, and reuses it for every query - far less memory pressure and
    much faster (single matrix multiply instead of a per-row Python loop).

    NOTE: the leading underscore on _df tells Streamlit's cache to skip
    hashing the (large, slow-to-hash) DataFrame itself; cache_key is the
    small, cheap value Streamlit actually uses to decide whether to recompute.
    """
    matrix = np.stack(_df["embedding"].values).astype(np.float32)
    # Pre-normalize once, so retrieval becomes a plain dot product (cosine
    # similarity without repeated norm calculations per query).
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  # avoid division by zero for any empty embeddings
    return matrix / norms


def retrieve(query: str, df: pd.DataFrame, embedding_matrix: np.ndarray, model, k: int = 5) -> list:
    """
    Vectorized retrieval: one matrix-vector multiply against the pre-normalized
    embedding matrix, instead of a Python loop recomputing every row's array
    conversion and cosine similarity individually on every single query.
    """
    query_embedding = model.encode(query).astype(np.float32)
    query_norm = np.linalg.norm(query_embedding)
    if query_norm > 0:
        query_embedding = query_embedding / query_norm

    similarities = embedding_matrix @ query_embedding  # single vectorized operation
    top_k_idx = np.argsort(similarities)[::-1][:k]

    return [(float(similarities[i]), df.iloc[i]) for i in top_k_idx]


def build_prompt(query: str, retrieved: list) -> str:
    context_blocks = []
    for i, (score, row) in enumerate(retrieved):
        source_label = f"[Source {i+1}: {row['ticker']} - {row['source_type']} - {row.get('doc_subtype', 'N/A')}]"
        context_blocks.append(f"{source_label}\n{row['chunk_text']}")
    context_text = "\n\n".join(context_blocks)

    return f"""You are a financial research assistant. Answer the question using ONLY the information in the sources below. Do not use outside knowledge.

CITATION RULES (strict):
- EVERY sentence that states a fact must end with a citation like [Source 1].
- Cite ONLY the specific source(s) that actually contain that exact fact.
- Most sentences should have exactly ONE citation.
- If the sources don't contain enough information to answer the question, say so explicitly rather than guessing.

SOURCES:
{context_text}

QUESTION: {query}

ANSWER (cite only the specific source for each individual claim):"""


st.title("Financial RAG — Offline Demo")
st.caption(
    "Runs entirely locally against a cached snapshot of the real ~1,390-chunk "
    "Gold corpus from Databricks - no Azure/Databricks connection required. "
    "The snapshot was exported once via export_once.py while the cluster was "
    "running; retrieval works fully offline, generation needs an OpenAI key."
)

df = load_cached_data()

if df is None:
    st.error(
        f"No cached data found ({CACHE_FILE} missing). Run `python export_once.py` "
        "once first, while your Databricks cluster is running, to create the cache."
    )
    st.stop()

st.success(f"Loaded {len(df)} real chunks from local cache (no live connection needed).")
st.caption(
    "Note: this is a static snapshot exported once from the live Databricks "
    "pipeline, not a live-updating feed. It reflects the corpus as of the "
    "export date, not real-time data."
)

openai_key = st.sidebar.text_input("OpenAI API Key (only needed to generate answers):", type="password")
st.sidebar.caption("Retrieval works without this. It's only used to generate the final cited answer.")

REAL_EVAL_QUESTIONS = [
    "What are the key risk factors in Apple's latest 10-K?",
    "How is Microsoft's Azure cloud business performing?",
    "What are Realty Income's dividend distribution policies?",
    "What intellectual property risks does Apple face?",
    "What recent news has come out about Apple and Chinese memory chip suppliers?",
    "What is Microsoft's revenue growth in Intelligent Cloud?",
    "How does Realty Income describe its business model?",
    "What macroeconomic factors does Apple cite as risks?",
    "What tariff-related risks does Apple disclose?",
    "What does Microsoft say about its Productivity and Business Processes segment?",
    "What is Apple's approach to third-party software developers?",
    "What is Realty Income's Adjusted EBITDAre and why does it matter?",
    "What is the outlook for Apple's stock price this year?",
    "What legal proceedings is Apple currently involved in?",
    "How has foreign currency impacted Microsoft's reported revenue?",
]

question_choice = st.selectbox("Pick a real evaluation question:", ["(type my own)"] + REAL_EVAL_QUESTIONS)
query = st.text_input("Your question:") if question_choice == "(type my own)" else question_choice

ticker_filter = st.selectbox("Filter by ticker (optional):", ["(all)"] + sorted(df["ticker"].unique().tolist()))

if st.button("Ask", type="primary") and query:
    # Build the embedding matrix ONCE against the full dataset (cached across
    # the whole session - not rebuilt on every query or every filter change).
    model = load_embedding_model()
    full_embedding_matrix = build_embedding_matrix(df, cache_key=CACHE_FILE)

    if ticker_filter == "(all)":
        filtered_df = df.reset_index(drop=True)
        filtered_matrix = full_embedding_matrix
    else:
        # Slice both the dataframe AND the matrix using the same boolean mask,
        # so row positions stay aligned between the two.
        mask = (df["ticker"] == ticker_filter).to_numpy()
        filtered_df = df[mask].reset_index(drop=True)
        filtered_matrix = full_embedding_matrix[mask]

    retrieved = retrieve(query, filtered_df, filtered_matrix, model, k=5)

    st.subheader("Retrieved sources (from local cache of your real corpus)")
    for i, (score, row) in enumerate(retrieved):
        with st.expander(f"[{i+1}] {row['ticker']} - {row['source_type']} - {row.get('doc_subtype','N/A')} (similarity: {score:.3f})"):
            st.write(row["chunk_text"])

    if openai_key:
        client = OpenAI(api_key=openai_key)
        prompt = build_prompt(query, retrieved)
        with st.spinner("Generating cited answer..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
        st.subheader("Answer")
        st.write(response.choices[0].message.content)
    else:
        st.info("Enter an OpenAI API key in the sidebar to generate a cited answer from these real retrieved chunks.")
