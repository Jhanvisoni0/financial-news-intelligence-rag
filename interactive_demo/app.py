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
st.info(
    "The default text below is a real excerpt combining actual passages from "
    "Apple's 10-Q, Microsoft's 10-Q, and Realty Income's 10-K/10-Q filings "
    "(the same documents used in the full project). This tool runs fully "
    "locally and demonstrates the chunking/embedding mechanism on a small "
    "sample - it is not connected live to the full 1,390-chunk Databricks "
    "corpus, which requires the live Databricks environment (see the main "
    "README for the full RAG system results)."
)

DEFAULT_TEXT = """UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 10-Q Apple Inc. The Company's business, results of operations and financial condition could be materially adversely affected by acts of litigation or government investigations. The outcome of litigation or government investigations is inherently uncertain. Apple discloses that new tariffs were announced on imports to the U.S. beginning in the second quarter of 2025, which include additional tariffs on imports from several countries such as China, India, Japan, South Korea, Taiwan, Vietnam, and the European Union. In response to these U.S. tariffs, several countries have imposed or threatened to impose reciprocal tariffs on imports from the U.S. Tariffs and other measures applied to Apple's products or their components can have a material adverse impact on its business, results of operations, and financial condition, affecting the supply chain, availability of raw materials, pricing, and gross margin. The Company is exposed to risks related to the infringement of intellectual property rights, which can require modifications to products or result in significant licensing costs. Apple relies heavily on third-party software developers, and any decision by these developers to discontinue support for Apple's platforms could negatively affect customer purchasing behavior. OVERVIEW Microsoft is a technology company committed to making digital technology and artificial intelligence available broadly and responsibly. Azure and other cloud services revenue increased by 40% driven by demand for Microsoft's portfolio of services with continued growth across all workloads. Intelligent Cloud Revenue increased $14.2 billion or 29%, driven by Azure. Realty Income, as a REIT, must distribute at least 90% of its taxable income as dividends to maintain its tax-advantaged status. Our Adjusted EBITDAre is generally consistent with the Nareit definition. Together with our analytics-supported underwriting, our business model is designed to generate a stable and predictable revenue stream through long-term commercial real estate leases."""

# The actual 15-question evaluation set used to test the real RAG system
# (see ../ENGINEERING_LOG.md and the Evaluation section of ../README.md for
# the full scored results against the real 1,390-chunk corpus in Databricks).
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
        st.caption(
            "Choose from the actual 15-question evaluation set used to test the "
            "real RAG system (see ENGINEERING_LOG.md / README.md for the full "
            "scored results against the real 1,390-chunk corpus), or type your own."
        )
        question_choice = st.selectbox(
            "Pick a real evaluation question:",
            ["(type my own instead)"] + REAL_EVAL_QUESTIONS,
        )
        if question_choice == "(type my own instead)":
            query = st.text_input("Your question:", placeholder="e.g. What risks does Apple face?")
        else:
            query = question_choice
            st.write(f"**Question:** {query}")

        if query:
            query_embedding = model.encode(query)
            similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
            ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

            st.write("**Chunks ranked by similarity to your question:**")
            for rank, (idx, score) in enumerate(ranked):
                preview = chunks[idx][:150] + ("..." if len(chunks[idx]) > 150 else "")
                st.markdown(f"**#{rank+1} - Chunk {idx+1}** (similarity: `{score:.3f}`)")
                st.markdown(f"> {preview}")
