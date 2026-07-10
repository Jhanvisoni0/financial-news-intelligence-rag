# Databricks notebook source
# MAGIC %pip uninstall -y sentence-transformers huggingface_hub transformers tokenizers
# MAGIC %pip install sentence-transformers==3.0.1 huggingface_hub==0.23.4 transformers==4.42.4 tokenizers==0.19.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install chromadb
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Vector Store & Retrieval
# Loads Gold-layer embeddings into ChromaDB and builds a retrieval function.
# This is the "R" in RAG - given a query, find the most relevant chunks.

# ---- Install (run once per cluster session) ----
# %pip install chromadb
# dbutils.library.restartPython()

import chromadb
from sentence_transformers import SentenceTransformer

# =========================================================
# PART 1: Load Gold data into ChromaDB
# =========================================================

# ChromaDB running in local-persistent mode on the cluster's local disk.
# NOTE: DBFS (/dbfs/...) does NOT work here - it doesn't support the low-level
# file operations (mmap, file locking) that Chroma's SQLite backend needs.
# Local disk (/local_disk0 or /tmp) works fine. This means the collection
# won't survive a cluster restart, but since we rebuild it from the Gold
# Delta table each run anyway, that's not a problem at this project's scale.
chroma_client = chromadb.PersistentClient(path="/local_disk0/tmp/financial_rag_chroma")

collection = chroma_client.get_or_create_collection(
    name="financial_documents",
    metadata={"hnsw:space": "cosine"}  # cosine similarity - standard for text embeddings
)

# Pull Gold data out of Spark into a regular pandas DataFrame.
# (Chroma isn't Spark-native, so we collect data locally - fine at this scale;
#  a production system with millions of chunks would batch this differently.)
gold_df = spark.table("gold.embedded_chunks").toPandas()

print(f"Loading {len(gold_df)} chunks into ChromaDB...")

# Chroma needs: ids (unique strings), embeddings (lists of floats),
# documents (the actual text), and metadatas (extra searchable fields).
collection.add(
    ids=[str(x) for x in gold_df["chunk_id"].tolist()],
    embeddings=gold_df["embedding"].tolist(),
    documents=gold_df["chunk_text"].tolist(),
    metadatas=gold_df[["ticker", "source_type", "doc_subtype", "date", "sentiment", "risk_density"]]
        .astype(str)  # Chroma metadata must be str/int/float/bool - cast dates etc. to str
        .to_dict(orient="records"),
)

print(f"ChromaDB collection now has {collection.count()} documents")


# =========================================================
# PART 2: Retrieval function
# =========================================================

# Load the SAME embedding model used to build the Gold embeddings.
# Critical: the query must be embedded with the same model as the documents,
# or the similarity comparison is meaningless (different vector spaces).
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query: str, k: int = 5, ticker_filter: str = None) -> list:
    """
    Given a natural language query, return the top-k most relevant chunks.

    Args:
        query: the user's question
        k: how many chunks to retrieve
        ticker_filter: optionally restrict results to one ticker (e.g. "AAPL")
    """
    query_embedding = embed_model.encode(query).tolist()

    where_clause = {"ticker": ticker_filter} if ticker_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_clause,
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],  # lower = more similar
        })
    return retrieved


# =========================================================
# PART 3: Test retrieval with hand-picked queries (no LLM yet - just check relevance)
# =========================================================

test_queries = [
    "What are Apple's key risk factors?",
    "How is Microsoft's cloud business performing?",
    "What is the sentiment around REITs and dividends?",
]

for q in test_queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {q}")
    print('='*60)
    results = retrieve(q, k=3)
    for r in results:
        print(f"\n  [ticker: {r['metadata']['ticker']}, source: {r['metadata']['source_type']}, distance: {r['distance']:.3f}]")
        print(f"  {r['text'][:200]}...")

# COMMAND ----------

