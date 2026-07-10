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


def test_chunk_text_short_text_returns_single_chunk():
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=600)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text(None) == []


def test_chunk_text_long_text_produces_multiple_chunks():
    long_text = "word " * 2000
    chunks = chunk_text(long_text, chunk_size=600, overlap=100)
    assert len(chunks) > 1


def test_count_tokens_matches_expected_range():
    text = "Hello world, this is a test."
    token_count = count_tokens(text)
    assert token_count > 0
    assert token_count < 20


def test_is_relevant_matches_plain_company_name():
    assert is_relevant("AAPL", "Apple weighs buying RAM from Chinese suppliers", "") is True


def test_is_relevant_excludes_unrelated_articles():
    assert is_relevant("AAPL", "38k-Mile 2016 Chevrolet Camaro 2SS Coupe", "A classic car listing") is False


def test_is_relevant_handles_empty_text():
    assert is_relevant("AAPL", "", "") is False
    assert is_relevant("AAPL", None, None) is False


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
