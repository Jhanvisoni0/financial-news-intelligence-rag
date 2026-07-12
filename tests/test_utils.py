# tests/test_utils.py
# Smoke tests for the core pipeline logic - not exhaustive unit tests, but
# enough to catch obvious regressions (e.g., someone breaks chunking or
# the relevance filter without realizing it) before they reach the pipeline.

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import (
    chunk_text,
    count_tokens,
    is_relevant,
    compute_risk_density,
    compute_sentiment,
)


# =========================================================
# Chunking tests
# =========================================================

def test_chunk_text_short_text_returns_single_chunk():
    """Text under the chunk size limit should not be split at all."""
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=600)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text(None) == []


def test_chunk_text_long_text_produces_multiple_chunks():
    """Text well over the chunk size should be split into multiple pieces."""
    long_text = "word " * 2000  # ~2000 words, well over 600 tokens
    chunks = chunk_text(long_text, chunk_size=600, overlap=100)
    assert len(chunks) > 1


def test_count_tokens_matches_expected_range():
    """A short, known sentence should have a small, non-zero token count."""
    text = "Hello world, this is a test."
    token_count = count_tokens(text)
    assert token_count > 0
    assert token_count < 20  # sanity check - this short sentence shouldn't be huge


# =========================================================
# Relevance filter tests (regression test for the bug we found and fixed)
# =========================================================

def test_is_relevant_matches_plain_company_name():
    """
    Regression test: this exact case (plain 'Apple' mention, no 'Apple Inc'
    phrase) was the real bug found during development - the original filter
    was too strict and missed this legitimate article.
    """
    assert is_relevant("AAPL", "Apple weighs buying RAM from Chinese suppliers", "") is True


def test_is_relevant_excludes_unrelated_articles():
    """An article that never mentions the company name should be excluded."""
    assert is_relevant("AAPL", "38k-Mile 2016 Chevrolet Camaro 2SS Coupe", "A classic car listing") is False


def test_is_relevant_handles_empty_text():
    assert is_relevant("AAPL", "", "") is False
    assert is_relevant("AAPL", None, None) is False


# =========================================================
# Regression tests for the 8 newly-added tickers (NVDA, AMZN, GOOGL, JPM,
# XOM, JNJ, DIS, BRK-B). Same pattern as the AAPL bug fix: catch both
# "misses a real match" and "matches something it shouldn't" cases.
# =========================================================

def test_is_relevant_new_tickers_match_real_headlines():
    """Each new ticker should correctly match a realistic headline mentioning it."""
    cases = [
        ("NVDA", "NVIDIA unveils new AI chip architecture", ""),
        ("AMZN", "Amazon expands same-day delivery to more cities", ""),
        ("GOOGL", "Google announces update to search algorithm", ""),
        ("JPM", "JPMorgan Chase reports quarterly earnings beat", ""),
        ("XOM", "Exxon Mobil to increase capital spending on refining", ""),
        ("JNJ", "Johnson & Johnson faces new product liability lawsuit", ""),
        ("DIS", "Disney reports strong streaming subscriber growth", ""),
        ("BRK-B", "Berkshire Hathaway increases stake in energy sector", ""),
    ]
    for ticker, title, description in cases:
        assert is_relevant(ticker, title, description) is True, f"{ticker} should match: {title}"


def test_is_relevant_new_tickers_exclude_unrelated_articles():
    """Same false-positive check as the original AAPL/Camaro bug, for new tickers."""
    cases = [
        ("NVDA", "Local bakery wins county fair pie contest", ""),
        ("AMZN", "New hiking trail opens in national park", ""),
        ("GOOGL", "City council approves new parking regulations", ""),
        ("JPM", "Weather forecast predicts mild autumn temperatures", ""),
    ]
    for ticker, title, description in cases:
        assert is_relevant(ticker, title, description) is False, f"{ticker} should NOT match: {title}"


def test_is_relevant_unknown_ticker_returns_false_not_error():
    """A ticker not in RELEVANCE_KEYWORDS should safely return False, not crash -
    this is the exact silent-bug scenario caught during the 11-ticker expansion:
    forgetting to add a ticker here means its news gets silently filtered out
    entirely, with no error raised. This test documents that behavior explicitly
    so it's a conscious, visible fact rather than a silent gap."""
    assert is_relevant("UNKNOWN_TICKER", "Some company news headline", "") is False


# =========================================================
# Feature engineering tests
# =========================================================

def test_compute_risk_density_detects_risk_language():
    text = "The company faces significant litigation and a material weakness in controls."
    density = compute_risk_density(text)
    assert density > 0


def test_compute_risk_density_zero_for_neutral_text():
    text = "The company sells consumer electronics and software products."
    density = compute_risk_density(text)
    assert density == 0.0


def test_compute_sentiment_detects_positive():
    text = "Strong growth and record profit this quarter, with significant gains."
    assert compute_sentiment(text) == "positive"


def test_compute_sentiment_detects_negative():
    text = "Significant decline in revenue, with major losses and an ongoing lawsuit."
    assert compute_sentiment(text) == "negative"


def test_compute_sentiment_defaults_to_neutral():
    text = "The company is headquartered in California."
    assert compute_sentiment(text) == "neutral"
