# interactive_demo/app_live.py
# Live-connected version: queries the REAL Gold table (gold.embedded_chunks)
# directly from Databricks over SQL, and runs the real RAG chain (retrieval +
# citation-grounded generation) against the actual ~1,390-chunk corpus -
# not a hardcoded sample. Requires a running Databricks cluster and valid
# credentials (see the sidebar inputs).
#
# Run with: streamlit run app_live.py
# (requires: pip install -r requirements_live.txt)

import streamlit as st
import numpy as np
from databricks import sql as databricks_sql
from sentence_transformers import SentenceTransformer
from openai import OpenAI

st.set_page_config(page_title="Financial RAG - Live Demo", layout="wide")


# =========================================================
# Sidebar: connection credentials (never hardcoded, never logged)
# =========================================================

st.sidebar.header("Databricks Connection")
db_hostname = st.sidebar.text_input("Server Hostname", value="adb-7405612624543070.10.azuredatabricks.net")
db_http_path = st.sidebar.text_input("HTTP Path", value="sql/protocolv1/o/7405612624543070/0703-031906-eib1v6oy")
db_token = st.sidebar.text_input("Databricks Access Token", type="password")

st.sidebar.header("OpenAI")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")

st.sidebar.caption(
    "Credentials are used only for this session and are never saved to disk "
    "or sent anywhere other than Databricks/OpenAI directly."
)


# =========================================================
# Core logic
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner="Querying real data from Databricks...")
def fetch_gold_data(hostname: str, http_path: str, token: str):
    """
    Connects directly to the Databricks SQL endpoint and pulls the real
    Gold table - the same ~1,390 chunks used in the actual project - rather
    than a local export or hardcoded sample.
    """
    connection = databricks_sql.connect(
        server_hostname=hostname,
        http_path=http_path,
        access_token=token,
    )
    cursor = connection.cursor()
    cursor.execute("""
        SELECT chunk_id, ticker, source_type, doc_subtype, date,
               chunk_text, sentiment, risk_density, embedding
        FROM gold.embedded_chunks
    """)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    cursor.close()
    connection.close()

    data = [dict(zip(columns, row)) for row in rows]
    return data


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(query: str, data: list, model, k: int = 5) -> list:
    """Real retrieval against the full live corpus - same logic as the project's retrieve()."""
    query_embedding = model.encode(query)
    scored = []
    for row in data:
        chunk_embedding = np.array(row["embedding"], dtype=np.float32)
        score = cosine_similarity(query_embedding, chunk_embedding)
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


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


# =========================================================
# UI
# =========================================================

st.title("Financial RAG — Live Demo")
st.caption(
    "This connects live to the real Databricks Gold table (~1,390 chunks from "
    "SEC filings and news, the same corpus used in the full project) and runs "
    "the actual retrieval + citation-grounded generation pipeline against it."
)

if not (db_hostname and db_http_path and db_token):
    st.warning("Enter your Databricks connection details in the sidebar to continue.")
    st.stop()

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

ticker_filter = st.selectbox("Filter by ticker (optional):", ["(all)", "AAPL", "MSFT", "O"])

if st.button("Ask", type="primary") and query:
    try:
        data = fetch_gold_data(db_hostname, db_http_path, db_token)
        st.success(f"Loaded {len(data)} real chunks from Databricks.")
    except Exception as e:
        st.error(f"Databricks connection failed: {e}")
        st.stop()

    if ticker_filter != "(all)":
        data = [row for row in data if row["ticker"] == ticker_filter]

    model = load_embedding_model()
    retrieved = retrieve(query, data, model, k=5)

    st.subheader("Retrieved sources (real chunks from your corpus)")
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
