# Databricks notebook source
# MAGIC %pip uninstall -y sentence-transformers huggingface_hub transformers tokenizers
# MAGIC %pip install sentence-transformers==3.0.1 huggingface_hub==0.23.4 transformers==4.42.4 tokenizers==0.19.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from sentence_transformers import SentenceTransformer
print("Import successful")

# COMMAND ----------

# MAGIC %pip install sentence-transformers huggingface_hub
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Gold Layer — Embeddings + Feature Engineering
# Uses sentence-transformers (local, free, no API key required) for embeddings.
# NOTE: Once your OpenAI key is verified, see the commented section at the bottom
# for how to swap in OpenAI's text-embedding-3-small instead - the rest of the
# pipeline (Chroma, retrieval, RAG) works identically regardless of which you use.

# ---- Install (run once per cluster session) ----
# %pip install sentence-transformers
# dbutils.library.restartPython()

import re
from sentence_transformers import SentenceTransformer
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType

# =========================================================
# PART 1: Embeddings (local model, no API key needed)
# =========================================================

# all-MiniLM-L6-v2 is a small, fast, well-regarded general-purpose embedding model.
# It produces 384-dimensional vectors (vs. OpenAI's 1536) - smaller but perfectly
# usable for a project this size, and completely free to run.
model = SentenceTransformer("all-MiniLM-L6-v2")

# Broadcast the model to all Spark workers so it's loaded once, not per-row
broadcast_model = spark.sparkContext.broadcast(model)


def embed_text(text: str) -> list:
    """Generate an embedding vector for a chunk of text using the local model."""
    if not text:
        return []
    vec = broadcast_model.value.encode(text)
    return vec.tolist()


embed_udf = udf(embed_text, ArrayType(FloatType()))


# =========================================================
# PART 2: Feature Engineering (sentiment, risk keywords, entities)
# =========================================================

# Simple keyword-based sentiment (a real project might use a proper sentiment
# model, but a transparent keyword approach is a reasonable, explainable v1).
POSITIVE_WORDS = ["growth", "profit", "beat", "strong", "record", "increase", "gain", "surge", "upgrade"]
NEGATIVE_WORDS = ["loss", "decline", "risk", "lawsuit", "investigation", "weak", "drop", "downgrade", "concern"]

RISK_KEYWORDS = [
    "litigation", "material weakness", "going concern", "regulatory action",
    "investigation", "restatement", "breach", "default", "impairment",
]


def compute_sentiment(text: str) -> str:
    """Basic keyword-count sentiment: compares positive vs negative word hits."""
    if not text:
        return "neutral"
    text_lower = text.lower()
    pos_count = sum(text_lower.count(word) for word in POSITIVE_WORDS)
    neg_count = sum(text_lower.count(word) for word in NEGATIVE_WORDS)
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    return "neutral"


def compute_risk_density(text: str) -> float:
    """
    Risk-keyword density: count of risk-related terms per 1000 words.
    Higher = more risk-language-heavy text (useful for filtering/ranking later).
    """
    if not text:
        return 0.0
    text_lower = text.lower()
    word_count = max(len(text_lower.split()), 1)
    risk_hits = sum(text_lower.count(term) for term in RISK_KEYWORDS)
    return round((risk_hits / word_count) * 1000, 2)


def compute_word_count(text: str) -> int:
    """Simple word count feature."""
    if not text:
        return 0
    return len(text.split())


sentiment_udf = udf(compute_sentiment, StringType())
risk_density_udf = udf(compute_risk_density, FloatType())
word_count_udf = udf(compute_word_count, IntegerType())


# =========================================================
# PART 3: Apply to chunked Silver table, write to Gold
# =========================================================

chunked = spark.table("silver.documents_chunked")

gold = (
    chunked
    .withColumn("embedding", embed_udf(col("chunk_text")))
    .withColumn("sentiment", sentiment_udf(col("chunk_text")))
    .withColumn("risk_density", risk_density_udf(col("chunk_text")))
    .withColumn("word_count", word_count_udf(col("chunk_text")))
)

spark.sql("CREATE SCHEMA IF NOT EXISTS gold")
gold.write.format("delta").mode("overwrite").saveAsTable("gold.embedded_chunks")

print(f"Gold table written: {gold.count()} chunks with embeddings + features")

display(
    gold.select("ticker", "source_type", "sentiment", "risk_density", "word_count")
    .orderBy("ticker", "source_type")
)

# =========================================================
# OPTIONAL: Swap in OpenAI embeddings once your key is verified
# =========================================================
# import openai
# openai_key = dbutils.secrets.get(scope="financial-rag", key="openai-api-key")
# client = openai.OpenAI(api_key=openai_key)
#
# def embed_text_openai(text: str) -> list:
#     if not text:
#         return []
#     response = client.embeddings.create(
#         model="text-embedding-3-small",
#         input=text
#     )
#     return response.data[0].embedding
#
# embed_openai_udf = udf(embed_text_openai, ArrayType(FloatType()))
# Then re-run the gold write with .withColumn("embedding", embed_openai_udf(col("chunk_text")))

# COMMAND ----------

from pyspark.sql.functions import col

sample = spark.table("gold.embedded_chunks").filter(col("source_type") == "sec_filing").limit(1)
row = sample.collect()[0]

print("chunk_token_count:", row["chunk_token_count"])
print("word_count:", row["word_count"])
print("---- chunk_text preview ----")
print(row["chunk_text"][:500])


# COMMAND ----------

from pyspark.sql.functions import col

risk_check = (
    spark.table("gold.embedded_chunks")
    .filter((col("source_type") == "sec_filing") & (col("risk_density") > 0))
    .count()
)

total_sec_chunks = spark.table("gold.embedded_chunks").filter(col("source_type") == "sec_filing").count()

print(f"SEC chunks with risk_density > 0: {risk_check} out of {total_sec_chunks}")

# COMMAND ----------

