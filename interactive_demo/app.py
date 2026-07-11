# interactive_demo/app.py
# Interactive visualization of chunking and embedding - the two pipeline
# stages between raw text and vector search. Paste any text, adjust chunk
# size/overlap, and see the real embedding model (same one used in the
# project's Gold layer) place each chunk in 2D space based on meaning.
#
# Run with: streamlit run app.py
# (requires: pip install streamlit sentence-transformers tiktoken scikit-learn plotly)

import streamlit as st
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.graph_objects as go


st.set_page_config(page_title="Chunking & Embedding Explorer", layout="wide")


# =========================================================
# Core logic (same functions used in the actual project pipeline)
# =========================================================

@st.cache_resource
def load_embedding_model():
    """Load once and cache across reruns - avoids reloading the model on every interaction."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """Token-aware chunking, identical logic to the project's Silver layer."""
    if not text.strip():
        return []

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        if end == len(tokens):
            break
        start = end - overlap

    return chunks


def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# =========================================================
# UI
# =========================================================

st.title("Chunking & Embedding Explorer")
st.caption(
    "Visualizes the two pipeline stages between raw text and vector search: "
    "splitting text into token-sized chunks, then embedding each chunk into "
    "a point in meaning-space. Uses the same tokenizer (tiktoken) and "
    "embedding model (all-MiniLM-L6-v2) as the main project pipeline."
)

DEFAULT_TEXT = """Apple Inc. faces several risk factors including intellectual property disputes, where the company's products may be alleged to infringe existing patents held by competitors. The company also faces significant tariff-related risks, as new tariffs on imports from China, India, and other countries could materially impact its supply chain and gross margins. Additionally, Apple relies heavily on third-party software developers, and any decision by these developers to discontinue support for Apple's platforms could negatively affect customer demand. Microsoft's Azure cloud business has shown strong growth, with Intelligent Cloud revenue increasing significantly, driven by demand across all workloads including AI consumption services. Realty Income, as a REIT, must distribute at least 90% of its taxable income as dividends to maintain its tax-advantaged status, and its business model focuses on generating stable, predictable revenue through long-term commercial real estate leases."""

text_input = st.text_area(
    "Paste text to chunk and embed (or use the sample SEC-style text below):",
    value=DEFAULT_TEXT,
    height=150,
)

col1, col2 = st.columns(2)
with col1:
    chunk_size = st.slider("Chunk size (tokens)", min_value=20, max_value=200, value=60, step=10)
with col2:
    overlap = st.slider("Overlap (tokens)", min_value=0, max_value=50, value=10, step=5)

if st.button("Chunk & Embed", type="primary"):
    chunks = chunk_text(text_input, chunk_size, overlap)

    if not chunks:
        st.warning("Please enter some text first.")
    else:
        st.subheader(f"Step 1: Chunking — {len(chunks)} chunk(s) created")
        st.caption(
            "Text is split by TOKEN count (using tiktoken, the same tokenizer OpenAI "
            "models use) rather than by words or characters - so chunk boundaries "
            "reflect what the model actually 'sees', not just character counts."
        )

        colors = ["#e8f0fe", "#fef3e8", "#e8fef0", "#fee8f0", "#f0e8fe", "#fefce8"]
        for i, chunk in enumerate(chunks):
            token_count = count_tokens(chunk)
            with st.expander(f"Chunk {i+1} — {token_count} tokens", expanded=(len(chunks) <= 3)):
                st.markdown(
                    f"<div style='background-color:{colors[i % len(colors)]}; "
                    f"padding:10px; border-radius:5px;'>{chunk}</div>",
                    unsafe_allow_html=True,
                )

        st.subheader("Step 2: Embedding — chunks placed in meaning-space")
        st.caption(
            "Each chunk is converted into a 384-dimensional vector capturing its "
            "meaning. Since humans can't visualize 384 dimensions, PCA compresses "
            "this down to 2D for display - chunks discussing similar topics should "
            "land closer together."
        )

        with st.spinner("Loading embedding model and computing embeddings..."):
            model = load_embedding_model()
            embeddings = model.encode(chunks)

        if len(chunks) >= 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(embeddings)

            fig = go.Figure()
            for i, (x, y) in enumerate(coords):
                preview = chunks[i][:80] + ("..." if len(chunks[i]) > 80 else "")
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode="markers+text",
                    marker=dict(size=20, color=colors[i % len(colors)], line=dict(width=2, color="black")),
                    text=[f"Chunk {i+1}"],
                    textposition="top center",
                    hovertext=preview,
                    hoverinfo="text",
                    name=f"Chunk {i+1}",
                ))
            fig.update_layout(
                height=500,
                xaxis_title="PCA dimension 1",
                yaxis_title="PCA dimension 2",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add more text (producing 2+ chunks) to see the 2D embedding plot.")

        st.subheader("Step 3: Try a query — see which chunk is most relevant")
        query = st.text_input("Ask a question about the text above:", placeholder="e.g. What risks does Apple face?")

        if query:
            query_embedding = model.encode(query)
            similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
            ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

            st.write("**Chunks ranked by similarity to your question:**")
            for rank, (idx, score) in enumerate(ranked):
                preview = chunks[idx][:150] + ("..." if len(chunks[idx]) > 150 else "")
                st.markdown(f"**#{rank+1} - Chunk {idx+1}** (similarity: `{score:.3f}`)")
                st.markdown(f"> {preview}")
