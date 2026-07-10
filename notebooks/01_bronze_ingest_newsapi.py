# Databricks notebook source
# NewsAPI Ingestion — Bronze Layer
import requests
import time
from datetime import datetime, timezone

NEWSAPI_KEY = dbutils.secrets.get(scope="financial-rag", key="newsapi-key")
NEWSAPI_URL = "https://newsapi.org/v2/everything"

COMPANIES = {
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft",
    "O": "Realty Income REIT",
}

ARTICLES_PER_COMPANY = 20


def fetch_news_for_company(company_name: str, page_size: int = 20) -> list:
    params = {
        "q": company_name,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY,
    }
    resp = requests.get(NEWSAPI_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        print(f"  API returned non-ok status: {data.get('message', 'unknown error')}")
        return []

    return data.get("articles", [])


def ingest_all_news() -> list:
    all_records = []

    for ticker, company_name in COMPANIES.items():
        print(f"Fetching news for {ticker} ({company_name})...")
        try:
            articles = fetch_news_for_company(company_name, page_size=ARTICLES_PER_COMPANY)
        except Exception as e:
            print(f"  FAILED to fetch news for {ticker}: {e}")
            continue

        print(f"  Retrieved {len(articles)} articles")

        for article in articles:
            record = {
                "ticker": ticker,
                "source_type": "news_article",
                "source_name": article.get("source", {}).get("name"),
                "title": article.get("title"),
                "description": article.get("description"),
                "raw_text": article.get("content") or article.get("description") or "",
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            all_records.append(record)

        time.sleep(1)

    return all_records


# ---- RUN IT ----
news_records = ingest_all_news()
print(f"\nIngested {len(news_records)} news articles total.")
if news_records:
    print(news_records[0]["ticker"], "-", news_records[0]["title"])

# COMMAND ----------

df_news = spark.createDataFrame(news_records)
df_news.write.format("delta").mode("overwrite").saveAsTable("bronze.news_articles")

print("Written to bronze.news_articles")
display(spark.table("bronze.news_articles").select("ticker", "source_name", "title", "published_at"))

# COMMAND ----------

