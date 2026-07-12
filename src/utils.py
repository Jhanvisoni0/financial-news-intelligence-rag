# src/utils.py
# Standalone, testable versions of the core pipeline logic used in this project.
# These are extracted from the Databricks notebooks (which wrap them in Spark UDFs)
# so they can be unit tested in plain Python, without needing a running cluster.
# The notebook versions call these same functions inside Spark UDFs.

import re
import tiktoken


# =========================================================
# Chunking logic (used in 03_silver_chunking)
# =========================================================

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list:
    """
    Split text into overlapping chunks based on token count (not word/character count).
    Uses tiktoken's cl100k_base encoding, matching OpenAI's embedding models.
    """
    if not text:
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
        chunk_str = encoding.decode(chunk_tokens)
        chunks.append(chunk_str)

        if end == len(tokens):
            break
        start = end - overlap

    return chunks


def count_tokens(text: str) -> int:
    """Return the token count for a given piece of text."""
    if not text:
        return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# =========================================================
# News relevance filtering logic (used in 02_silver_clean_transform)
# =========================================================

RELEVANCE_KEYWORDS = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "O": ["realty income", "o stock"],
    "NVDA": ["nvidia", "nvda"],
    "AMZN": ["amazon", "amzn"],
    "GOOGL": ["google", "alphabet", "googl"],
    "JPM": ["jpmorgan", "jp morgan", "chase"],
    "XOM": ["exxon", "exxonmobil"],
    "JNJ": ["johnson & johnson", "johnson and johnson"],
    "DIS": ["disney"],
    "BRK-B": ["berkshire hathaway", "berkshire"],
}


def is_relevant(ticker: str, title: str, description: str) -> bool:
    """
    Check whether a news article is actually about the company it was tagged
    with, using a word-boundary match on the plain company name.
    """
    if not title and not description:
        return False
    combined = f"{title or ''} {description or ''}".lower()
    keywords = RELEVANCE_KEYWORDS.get(ticker, [])
    return any(re.search(r"\b" + re.escape(keyword) + r"\b", combined) for keyword in keywords)


# =========================================================
# Feature engineering logic (used in 04_gold_embeddings_features)
# =========================================================

RISK_KEYWORDS = [
    "litigation", "material weakness", "going concern", "regulatory action",
    "investigation", "restatement", "breach", "default", "impairment",
]

POSITIVE_WORDS = ["growth", "profit", "beat", "strong", "record", "increase", "gain", "surge", "upgrade"]
NEGATIVE_WORDS = ["loss", "decline", "risk", "lawsuit", "investigation", "weak", "drop", "downgrade", "concern"]


def compute_risk_density(text: str) -> float:
    """Risk-keyword density: count of risk-related terms per 1000 words."""
    if not text:
        return 0.0
    text_lower = text.lower()
    word_count = max(len(text_lower.split()), 1)
    risk_hits = sum(text_lower.count(term) for term in RISK_KEYWORDS)
    return round((risk_hits / word_count) * 1000, 2)


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
