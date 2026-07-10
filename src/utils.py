import re
import tiktoken


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list:
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
    if not text:
        return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


RELEVANCE_KEYWORDS = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "O": ["realty income", "o stock"],
}


def is_relevant(ticker: str, title: str, description: str) -> bool:
    if not title and not description:
        return False
    combined = f"{title or ''} {description or ''}".lower()
    keywords = RELEVANCE_KEYWORDS.get(ticker, [])
    return any(re.search(r"\b" + re.escape(keyword) + r"\b", combined) for keyword in keywords)


RISK_KEYWORDS = [
    "litigation", "material weakness", "going concern", "regulatory action",
    "investigation", "restatement", "breach", "default", "impairment",
]

POSITIVE_WORDS = ["growth", "profit", "beat", "strong", "record", "increase", "gain", "surge", "upgrade"]
NEGATIVE_WORDS = ["loss", "decline", "risk", "lawsuit", "investigation", "weak", "drop", "downgrade", "concern"]


def compute_risk_density(text: str) -> float:
    if not text:
        return 0.0
    text_lower = text.lower()
    word_count = max(len(text_lower.split()), 1)
    risk_hits = sum(text_lower.count(term) for term in RISK_KEYWORDS)
    return round((risk_hits / word_count) * 1000, 2)


def compute_sentiment(text: str) -> str:
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
