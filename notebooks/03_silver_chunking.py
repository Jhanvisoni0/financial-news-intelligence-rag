import tiktoken
from pyspark.sql.functions import udf, col, explode
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, IntegerType


CHUNK_SIZE_TOKENS = 600    
CHUNK_OVERLAP_TOKENS = 100  



def _get_encoding():
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> list:
    """
    Split text into overlapping chunks based on token count (not word/character count).

    Why overlap matters: if a sentence or idea spans a chunk boundary, overlap ensures
    it still appears in full within at least one chunk, rather than being cut in half
    and losing meaning in both pieces.
    """
    if not text:
        return []

    encoding = _get_encoding()
    tokens = encoding.encode(text)
    if len(tokens) <= chunk_size:
        
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_str = encoding.decode(chunk_tokens)
        chunks.append(chunk_str)

        if end == len(tokens):
            break
        start = end - overlap 

    return chunks


chunk_text_udf = udf(chunk_text, ArrayType(StringType()))


def count_tokens(text: str) -> int:
    """Return the token count for a given piece of text."""
    if not text:
        return 0
    encoding = _get_encoding()
    return len(encoding.encode(text))


count_tokens_udf = udf(count_tokens, IntegerType())

# Apply chunking to the unified Silver table

silver_docs = spark.table("silver.documents")

chunked = (
    silver_docs
    .withColumn("chunks", chunk_text_udf(col("text")))
    .withColumn("chunk_text", explode(col("chunks")))
    .withColumn("chunk_token_count", count_tokens_udf(col("chunk_text")))
    .drop("chunks", "text")
)

from pyspark.sql.functions import monotonically_increasing_id

chunked = chunked.withColumn("chunk_id", monotonically_increasing_id())

chunked.write.format("delta").mode("overwrite").saveAsTable("silver.documents_chunked")

total_chunks = chunked.count()
print(f"Total chunks created: {total_chunks}")

display(
    chunked.select("ticker", "source_type", "doc_subtype", "chunk_token_count")
    .orderBy("ticker", "source_type")
)

# COMMAND ----------

