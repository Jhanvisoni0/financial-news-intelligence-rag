# Financial News Intelligence RAG Platform

A Retrieval-Augmented Generation (RAG) system that answers natural language questions about SEC filings and financial news, with citations back to the exact source document. Built on Azure Databricks using a Medallion architecture, with a production-style layer of MLOps, IaC, and CI/CD wrapped around the core RAG pipeline.

**Example query:** *"What are the key risk factors in Apple's latest 10-K?"* → a grounded, cited answer pulling directly from Apple's actual filing text — not a general-knowledge guess.

---

## Why this project

Most "RAG demo" projects retrieve from a handful of clean text files. This one intentionally uses messy, real-world data instead: SEC filings (which bury human-readable text inside machine-readable Inline XBRL markup) and live news APIs (which return plenty of irrelevant noise alongside real matches). Getting from raw data to trustworthy, cited answers required real data engineering, not just an API call — and that's the actual point of the project.

---

## Architecture

```
                    ┌─────────────────┐         ┌──────────────────┐
                    │   SEC EDGAR API  │         │    NewsAPI        │
                    └────────┬─────────┘         └────────┬─────────┘
                             │                             │
                             ▼                             ▼
                    ┌─────────────────────────────────────────────┐
                    │              BRONZE (raw ingestion)          │
                    │     bronze.sec_filings, bronze.news_articles │
                    └────────────────────┬──────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │       SILVER (cleaned, unified schema)       │
                    │  - iXBRL tag-data stripped from filings      │
                    │  - Word-boundary relevance filter for news   │
                    │  - Token-aware chunking (tiktoken)           │
                    └────────────────────┬──────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │      GOLD (embeddings + feature engineering) │
                    │  - sentence-transformers embeddings          │
                    │  - Sentiment, risk-density, word count        │
                    └────────────────────┬──────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │              ChromaDB (vector store)         │
                    └────────────────────┬──────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │     RAG CHAIN: retrieve → prompt → generate  │
                    │        (OpenAI gpt-4o-mini, cited answers)   │
                    └────────────────────┬──────────────────────────┘
                                         ▼
                              User question → cited answer
```

**Supporting layers around the core pipeline:** MLflow (experiment tracking), Databricks Workflows (scheduled orchestration), a BI Dashboard (Gold-layer analytics), Terraform (IaC for storage + secrets), and GitHub Actions (CI/CD).

---

## Tech stack

| Layer | Tool |
|---|---|
| Ingestion | Python (`requests`), SEC EDGAR API, NewsAPI |
| Storage | Delta Lake on Azure Databricks (Medallion architecture) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| Generation | OpenAI `gpt-4o-mini` |
| RAG orchestration | Custom retrieval + prompt-construction chain |
| Experiment tracking | MLflow |
| Orchestration | Databricks Workflows |
| Serving / analytics | Databricks AI/BI Dashboard |
| IaC | Terraform (`azurerm` provider) |
| CI/CD | GitHub Actions |
| Secrets management | Azure Key Vault + Databricks secret scopes |
| AI-assisted development | Claude (Anthropic) - used throughout for code review, debugging, and architecture discussion; all design decisions and code understood and validated by the author |

---

## Data pipeline: what actually happens at each stage

**Bronze:** Raw ingestion. SEC EDGAR filings (10-K/10-Q for AAPL, MSFT, O) pulled via the public submissions API (no key required, just a descriptive User-Agent). News articles pulled via NewsAPI for the same tickers.

**Silver:** Two real data-quality problems were found and fixed here (see `ENGINEERING_LOG.md` for full detail):
1. Modern SEC filings use **Inline XBRL** - machine-readable tag data embedded directly inside the human-readable HTML. Naive HTML stripping let this tag data pollute the extracted text, causing near-zero "risk density" scores and undersized word counts. Fixed with regex-based removal of `ix:`-namespaced elements before parsing.
2. A first-pass news relevance filter (designed to exclude noise like an unrelated Chevrolet Camaro listing matching "Apple" as a stray word) was initially *too* strict, silently dropping legitimate articles that used the plain company name instead of a longer phrase. Fixed with a word-boundary regex match.

Documents are then chunked using `tiktoken` (~600 tokens per chunk, 100-token overlap), matching the tokenizer used by the embedding/generation models downstream.

**Gold:** Each chunk is embedded and enriched with engineered features - a keyword-based sentiment score, a risk-keyword density score (counts of terms like "litigation," "material weakness," "going concern" per 1,000 words), and word count. These features are also used for retrieval filtering.

**Retrieval + Generation:** Chunks are loaded into ChromaDB. A query is embedded with the same model used for the documents, the top-k most similar chunks are retrieved, and an LLM generates an answer using *only* the retrieved context - with an explicit instruction to cite the specific source for each factual sentence, and to say so plainly if the sources don't contain an answer rather than guessing.

---

## Evaluation

A 15-question test set was built covering both tickers and both data sources, including one intentionally unanswerable question ("What is the outlook for Apple's stock price this year?") to test hallucination resistance.

| Metric | Result |
|---|---|
| Retrieval - correct ticker | 100% |
| Retrieval - correct source type | 100% |
| Avg. groundedness (1-5, LLM-as-judge) | 4.00 |
| Avg. relevance (1-5) | 4.80 |
| Avg. citation quality (1-5) | 3.87 |
| Questions correctly declined (hallucination avoidance) | 1 / 15 |

**Note on the citation quality score:** this average includes the intentionally unanswerable question, which the system correctly declined to answer - a pass on hallucination-avoidance, but scored as low "quality" by the judge since it's a short refusal rather than a full answer. Excluding that one deliberately-unanswerable question, quality scores on the 14 answerable questions are meaningfully higher.

**Citation quality was iterated on twice:**
1. Initial prompt scored 3.73/5 - the judge flagged answers making factual claims without a citation on every sentence.
2. After tightening the prompt to require a citation on every factual sentence, the score barely moved (3.87) - digging into the low scorers revealed the model was now citing every sentence, but blanket-citing *all* retrieved sources on every sentence rather than the specific one that supported each fact.
3. A second prompt revision explicitly instructed the model to cite only the specific source containing each fact, with a worked example showing discriminating citation. This produced qualitatively much better output (each citation now traceable to a distinct, correct source) - though the aggregate score didn't move further, since remaining judge complaints shifted to a different, harder problem: bare citation numbers don't self-describe what they point to. The next real fix here would be structured output (e.g., OpenAI function calling with a citation schema), not another prompt iteration - noted below as a concrete next step.

---

## MLOps, IaC, and CI/CD (the production layer)

These exist to support the RAG system above - they are not the centerpiece of the project, but they demonstrate that the system is operable, not just a notebook that happened to work once.

- **MLflow:** Every RAG query is logged as an experiment run (question, retrieved sources, latency, answer, tagged by ticker), making the system's behavior over time reviewable rather than opaque.
- **Databricks Workflows:** The ingestion pipeline is configured as a scheduled job (currently paused in this development environment to conserve compute credits, but fully configured and runnable on demand).
- **BI Dashboard:** Three charts built directly on the Gold layer - sentiment distribution by ticker, average risk density by ticker, and document volume by source type. The volume chart revealed a real, honest finding: news coverage is a small fraction of the corpus compared to SEC filing text, once irrelevant articles are filtered out.
- **Terraform:** Core infrastructure (storage account, Key Vault) is codified as Terraform, validated with `terraform init` / `terraform validate`. This infrastructure was originally provisioned manually during initial development, then codified afterward - a common real-world pattern. Role assignments and the Databricks workspace itself are intentionally excluded from the Terraform config (documented in `terraform/README.md`) since they involve tenant-specific identifiers not portable across environments.
- **CI/CD:** A GitHub Actions workflow lints the codebase and runs a suite of unit tests (including a regression test for the news-relevance-filter bug described above) on every push.

---

## Real engineering problems solved

This project hit a number of genuine, non-trivial technical problems along the way - full write-ups, root causes, and fixes for all of them are in [`ENGINEERING_LOG.md`](./ENGINEERING_LOG.md). Highlights:
- Diagnosing and fixing Inline XBRL data pollution in SEC filing text extraction
- Azure RBAC control-plane vs. data-plane access separation (Owner role does not grant Key Vault secret access)
- Identifying the wrong Azure identity (human user vs. `AzureDatabricks` service principal) blocking Key Vault access
- A PySpark UDF pickling failure caused by a Rust-based tokenizer library
- ChromaDB's incompatibility with DBFS-mounted storage (SQLite requires file operations DBFS's FUSE mount doesn't support)

---

## What I'd improve with more time

- **Structured output for citations.** The remaining citation-quality gap (see Evaluation section) would likely be better solved with OpenAI function calling / a JSON schema enforcing citation structure, rather than further prompt engineering.
- **A second embedding model comparison.** OpenAI's `text-embedding-3-small` was set up but not directly A/B tested against the local `sentence-transformers` model used here - a natural next experiment.
- **Streaming/real-time news ingestion**, rather than batch pulls, to keep the news corpus current.
- **A formal data quality framework** (e.g., Great Expectations) instead of ad hoc checks, to catch issues like the iXBRL pollution bug automatically rather than by manual inspection.
- **Expanding the Terraform scope** to include the Databricks workspace itself, using the `databricks` Terraform provider.

---

## Repository structure

```
notebooks/          Databricks notebooks (ingestion, transformation, RAG chain)
src/                 Standalone, testable core logic (chunking, filtering, feature engineering)
tests/               Unit tests for src/ (run via GitHub Actions CI)
terraform/           Infrastructure as Code for core Azure resources
.github/workflows/   CI/CD pipeline definition
ENGINEERING_LOG.md   Detailed write-ups of every real technical problem solved
requirements.txt     Python dependencies (with version pins where compatibility issues were hit)
```
