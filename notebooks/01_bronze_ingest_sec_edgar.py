# Databricks notebook source
# SEC EDGAR Ingestion — Bronze Layer
# Pulls company filings (10-K, 10-Q) from SEC EDGAR's public API.
# No API key required — SEC only requires a descriptive User-Agent header.
# Docs: https://www.sec.gov/os/webmaster-faq#developers

import requests
import time
from datetime import datetime, timezone

# ---- CONFIG ----
# IMPORTANT: SEC requires a real identifying User-Agent (your name/email).
# Requests without this, or with a generic one, can get you rate-limited or blocked.
USER_AGENT = "Jhanvi Soni jhanvisoni0@gmail.com"  # <-- already edited for you, double check it's right

HEADERS = {"User-Agent": USER_AGENT}

# Tickers to track -> SEC CIK numbers (Central Index Key, SEC's internal company ID)
TICKERS_TO_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "O": "0000726728",       # Realty Income (REIT example)
}

FORM_TYPES = ["10-K", "10-Q"]


def get_company_filings(cik: str) -> dict:
    """Fetch the filing history for a company from SEC's submissions API."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def extract_recent_filings(filings_json: dict, form_types: list) -> list:
    """Parse the submissions JSON and extract metadata for matching filings."""
    recent = filings_json.get("filings", {}).get("recent", {})
    results = []

    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_documents = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate", [])

    for i, form in enumerate(forms):
        if form in form_types:
            results.append({
                "form_type": form,
                "accession_number": accession_numbers[i],
                "filing_date": filing_dates[i],
                "report_date": report_dates[i],
                "primary_document": primary_documents[i],
            })
    return results


def get_filing_text(cik: str, accession_number: str, primary_document: str) -> str:
    """Fetch the actual filing document text (raw HTML, cleaned later in Silver)."""
    acc_no_clean = accession_number.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_clean}/{primary_document}"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.text


def ingest_all_filings() -> list:
    """Main ingestion loop: pull recent 10-K/10-Q filings + raw text for each ticker."""
    all_records = []

    for ticker, cik in TICKERS_TO_CIK.items():
        print(f"Fetching filing list for {ticker} (CIK {cik})...")
        filings_json = get_company_filings(cik)
        recent_filings = extract_recent_filings(filings_json, FORM_TYPES)

        # Limit to 4 most recent filings per ticker to keep this manageable for now
        recent_filings = recent_filings[:4]

        for filing in recent_filings:
            print(f"  Downloading {filing['form_type']} filed {filing['filing_date']}...")
            try:
                text = get_filing_text(cik, filing["accession_number"], filing["primary_document"])
            except Exception as e:
                print(f"  FAILED to fetch {filing['accession_number']}: {e}")
                continue

            record = {
                "ticker": ticker,
                "cik": cik,
                "source_type": "sec_filing",
                "form_type": filing["form_type"],
                "filing_date": filing["filing_date"],
                "report_date": filing["report_date"],
                "accession_number": filing["accession_number"],
                "raw_text": text,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            all_records.append(record)

            time.sleep(0.5)  # be polite to SEC's servers

    return all_records


# ---- RUN IT ----
records = ingest_all_filings()
print(f"\nIngested {len(records)} filings total.")
print(records[0]["ticker"], records[0]["form_type"], records[0]["filing_date"])

# COMMAND ----------

records = ingest_all_filings()
print(f"Ingested {len(records)} filings total.")
print(records[0]["ticker"], records[0]["form_type"], records[0]["filing_date"])

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")

df = spark.createDataFrame(records)
df.write.format("delta").mode("overwrite").saveAsTable("bronze.sec_filings")

print("Written to bronze.sec_filings")
display(spark.table("bronze.sec_filings").select("ticker", "form_type", "filing_date", "report_date"))

# COMMAND ----------

