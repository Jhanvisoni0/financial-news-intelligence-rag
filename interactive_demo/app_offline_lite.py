# interactive_demo/app_offline_lite.py
# Lightweight version: uses OpenAI's embedding API instead of a local
# sentence-transformers/torch model - removing the heaviest dependency from
# the deployed app entirely. Built after confirming (via repeated testing
# and memory diagnostics) that torch + transformers' baseline import footprint
# alone exceeded Streamlit Community Cloud's 1GB free-tier memory limit,
# regardless of dataset size or dependency version pinning (see
# ENGINEERING_LOG.md entry #14 for the full diagnosis).
#
# Requires gold_chunks_cache_openai.parquet (created by reembed_with_openai.py)
# - NOT the original gold_chunks_cache.parquet, which has sentence-transformers
# embeddings in a different vector space and isn't compatible with OpenAI
# query embeddings.
#
# Run with: streamlit run app_offline_lite.py

import os
import streamlit as st
import numpy as np
import pandas as pd
from openai import OpenAI

st.set_page_config(page_title="Financial RAG — Ask Your Filings", layout="wide")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SCRIPT_DIR, "gold_chunks_cache_openai.parquet")
EMBEDDING_MODEL = "text-embedding-3-small"


@st.cache_data
def load_cached_data():
    if not os.path.exists(CACHE_FILE):
        return None
    return pd.read_parquet(CACHE_FILE)


@st.cache_resource
def build_embedding_matrix(_df: pd.DataFrame, cache_key: str) -> np.ndarray:
    """Pre-compute a single, normalized embedding matrix once - same vectorized
    approach as the original app, just with OpenAI-sourced embeddings."""
    matrix = np.stack(_df["embedding"].values).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return matrix / norms


def embed_query(query: str, client: OpenAI) -> np.ndarray:
    """Embed the user's query via OpenAI's API - no local model needed at all."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    embedding = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def retrieve(query_embedding: np.ndarray, df: pd.DataFrame, embedding_matrix: np.ndarray, k: int = 5) -> list:
    similarities = embedding_matrix @ query_embedding
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


st.title("💰 Financial RAG — Ask Your Filings")
st.caption(
    "Ask a real question about Apple, Microsoft, or Realty Income and get a "
    "cited answer pulled directly from their actual SEC filings and news — "
    "not a general-knowledge guess. Grounded, verifiable, and honest when it "
    "doesn't know."
)

df = load_cached_data()

if df is None:
    st.error(
        "No cached data found. Run reembed_with_openai.py first "
        "to create gold_chunks_cache_openai.parquet from your existing data."
    )
    st.stop()

st.success(f"Loaded {len(df)} real chunks from local cache.")

openai_key = st.sidebar.text_input("OpenAI API Key (required):", type="password")
st.sidebar.caption("Needed for both retrieval (query embedding) and answer generation.")
st.sidebar.markdown("---")
st.sidebar.caption(
    "🔧 Runs on OpenAI's embedding API rather than a local model, keeping "
    "this deployment lightweight — [see the engineering write-up](https://github.com/Jhanvisoni0/financial-news-intelligence-rag/blob/main/ENGINEERING_LOG.md) "
    "for why."
)

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
    if not openai_key:
        st.warning("Please enter your OpenAI API key in the sidebar first.")
        st.stop()

    client = OpenAI(api_key=openai_key)

    full_embedding_matrix = build_embedding_matrix(df, cache_key=CACHE_FILE)

    if ticker_filter == "(all)":
        filtered_df = df.reset_index(drop=True)
        filtered_matrix = full_embedding_matrix
    else:
        mask = (df["ticker"] == ticker_filter).to_numpy()
        filtered_df = df[mask].reset_index(drop=True)
        filtered_matrix = full_embedding_matrix[mask]

    with st.spinner("Embedding your question..."):
        query_embedding = embed_query(query, client)

    retrieved = retrieve(query_embedding, filtered_df, filtered_matrix, k=5)

    st.subheader("Retrieved sources")
    for i, (score, row) in enumerate(retrieved):
        with st.expander(f"[{i+1}] {row['ticker']} - {row['source_type']} - {row.get('doc_subtype','N/A')} (similarity: {score:.3f})"):
            st.write(row["chunk_text"])

    prompt = build_prompt(query, retrieved)
    with st.spinner("Generating cited answer..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
    st.subheader("Answer")
    st.write(response.choices[0].message.content)
