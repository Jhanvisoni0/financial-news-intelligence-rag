# interactive_demo/reembed_with_openai.py
# Re-embeds the ALREADY-CACHED chunk text (from gold_chunks_cache.parquet)
# using OpenAI's embedding API instead of the local sentence-transformers
# model. This needs NO Databricks/Azure connection at all - it only reads
# the chunk_text column from data you already exported, and calls OpenAI.
#
# Why: the deployed Streamlit app segfaults on Streamlit Cloud's free tier
# because torch + sentence-transformers' baseline memory footprint alone
# exceeds the 1GB limit (confirmed via repeated testing - see
# ENGINEERING_LOG.md entry #14). Switching to API-based embeddings removes
# torch/transformers from the deployed app entirely.
#
# Run with: python reembed_with_openai.py

import getpass
import pandas as pd
from openai import OpenAI

INPUT_FILE = "gold_chunks_cache.parquet"
OUTPUT_FILE = "gold_chunks_cache_openai.parquet"
EMBEDDING_MODEL = "text-embedding-3-small"

print("=== Re-embedding cached data with OpenAI (no Databricks needed) ===\n")

openai_key = getpass.getpass("OpenAI API Key (hidden as you type): ")
client = OpenAI(api_key=openai_key)

print(f"\nLoading {INPUT_FILE}...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df)} chunks.")

new_embeddings = []
batch_size = 100  # OpenAI allows batching multiple texts per request - much faster/cheaper than one-at-a-time

for start in range(0, len(df), batch_size):
    batch_texts = df["chunk_text"].iloc[start:start + batch_size].tolist()
    print(f"Embedding chunks {start} to {min(start + batch_size, len(df))} of {len(df)}...")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=batch_texts,
    )
    batch_embeddings = [item.embedding for item in response.data]
    new_embeddings.extend(batch_embeddings)

df["embedding"] = new_embeddings

df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nSaved {len(df)} chunks with OpenAI embeddings to: {OUTPUT_FILE}")
print("Update app_offline.py's CACHE_FILE to point to this new file.")

# Rough cost estimate for reference (text-embedding-3-small pricing, approx)
total_tokens_estimate = sum(len(t.split()) for t in df["chunk_text"]) * 1.3  # rough word-to-token conversion
cost_estimate = (total_tokens_estimate / 1000) * 0.00002
print(f"\nApprox. cost for this run: ${cost_estimate:.4f}")
