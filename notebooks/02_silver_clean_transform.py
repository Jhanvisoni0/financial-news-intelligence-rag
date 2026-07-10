# Databricks notebook source
# MAGIC %pip install beautifulsoup4

# COMMAND ----------

# Silver Layer — Cleaning, Filtering, and Schema Unification
# Reads from bronze.sec_filings and bronze.news_articles,
# cleans/filters both, and writes a single unified silver table.

import re
from bs4 import BeautifulSoup
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import StringType, BooleanType

# =========================================================
# PART 1: Clean SEC filings (strip HTML down to readable text)
# =========================================================

def clean_html_to_text(raw_html: str) -> str:
    """
    Strip HTML/XBRL tags from a raw SEC filing document, returning plain text.

    Modern SEC filings use Inline XBRL (iXBRL): human-readable text and
    machine-readable XBRL facts are interleaved throughout the SAME document,
    often in many small <ix:...> tagged elements scattered everywhere (not
    just one hidden block). BeautifulSoup's default HTML parser doesn't always
    handle these namespaced tags cleanly, so we strip them with regex FIRST,
    before handing the remainder to BeautifulSoup for final cleanup.
    """
    if not raw_html:
        return ""
    try:
        # Step 1: Remove the entire <ix:header>...</ix:header> block, which
        # typically contains a large concentration of hidden XBRL facts.
        html = re.sub(r"<ix:header\b.*?</ix:header>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)

        # Step 2: Remove any remaining ix:-namespaced tags anywhere in the document
        # (these wrap individual XBRL-tagged facts embedded inline in the text).
        html = re.sub(r"</?ix:[a-zA-Z]+[^>]*>", "", html)

        # Step 3: Now parse the cleaned HTML normally
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        # Step 4: Still remove any explicitly hidden elements as a backup measure
        for tag in soup.find_all(style=lambda v: v and "display:none" in v.replace(" ", "")):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""


clean_html_udf = udf(clean_html_to_text, StringType())

sec_bronze = spark.table("bronze.sec_filings")

sec_silver = (
    sec_bronze
    .withColumn("clean_text", clean_html_udf(col("raw_text")))
    .withColumn("source_type", lit("sec_filing"))
    .withColumn("section", lit(None).cast(StringType()))  # placeholder; section extraction can be refined later
    .select(
        col("ticker"),
        col("source_type"),
        col("form_type").alias("doc_subtype"),
        col("filing_date").alias("date"),
        col("clean_text").alias("text"),
        col("accession_number").alias("source_id"),
        lit(None).cast(StringType()).alias("url"),
    )
)

print(f"SEC Silver records: {sec_silver.count()}")


# =========================================================
# PART 2: Clean + filter news articles (this is where we fix the noise problem)
# =========================================================

# Ticker -> the core company name(s) that must appear as a whole word.
# A word-boundary match on the plain company name works better than requiring
# specific phrases (e.g. "apple inc") - real headlines just say "Apple".
# This still excludes false positives (Camaro, SpaceX, etc.) since those
# articles never mention the company name at all.
RELEVANCE_KEYWORDS = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "O": ["realty income", "o stock"],
}


def is_relevant(ticker: str, title: str, description: str) -> bool:
    """
    Check whether a news article is actually about the company it was tagged with,
    by requiring the company name to appear as a whole word (word-boundary match)
    in the title or description - not just a phrase requiring extra context.
    """
    if not title and not description:
        return False
    combined = f"{title or ''} {description or ''}".lower()
    keywords = RELEVANCE_KEYWORDS.get(ticker, [])
    return any(re.search(r"\b" + re.escape(keyword) + r"\b", combined) for keyword in keywords)


is_relevant_udf = udf(is_relevant, BooleanType())


def clean_article_text(text: str) -> str:
    """Basic cleanup: strip NewsAPI's truncation artifacts and excess whitespace."""
    if not text:
        return ""
    # NewsAPI truncates content with a "[+XXXX chars]" suffix - strip that noise
    text = re.sub(r"\[\+\d+ chars\]$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


clean_article_udf = udf(clean_article_text, StringType())

news_bronze = spark.table("bronze.news_articles")

news_silver = (
    news_bronze
    .withColumn("is_relevant", is_relevant_udf(col("ticker"), col("title"), col("description")))
    .filter(col("is_relevant") == True)  # noqa: E712 - explicit for clarity with Spark boolean column
    .withColumn("clean_text", clean_article_udf(col("raw_text")))
    .select(
        col("ticker"),
        col("source_type"),
        col("source_name").alias("doc_subtype"),
        col("published_at").alias("date"),
        col("clean_text").alias("text"),
        col("title").alias("source_id"),  # using title as a rough identifier for news
        col("url"),
    )
)

print(f"News Silver records (after relevance filter): {news_silver.count()}")
print(f"News Bronze records (before filter): {news_bronze.count()}")


# =========================================================
# PART 3: Union both into one unified Silver table
# =========================================================

silver_unified = sec_silver.unionByName(news_silver)

spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
silver_unified.write.format("delta").mode("overwrite").saveAsTable("silver.documents")

print(f"\nWritten to silver.documents — total records: {silver_unified.count()}")
display(spark.table("silver.documents").select("ticker", "source_type", "doc_subtype", "date"))

# COMMAND ----------

