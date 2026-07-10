# Databricks notebook source
# MAGIC %pip install -U mlflow protobuf
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -U openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip uninstall -y sentence-transformers huggingface_hub transformers tokenizers
# MAGIC %pip install sentence-transformers==3.0.1 huggingface_hub==0.23.4 transformers==4.42.4 tokenizers==0.19.1 chromadb openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./05_vector_store_retrieval

# COMMAND ----------

# RAG Chain with Citation Tracking
# Combines retrieval (Day 6) + generation (LLM) to answer questions
# grounded in the actual filing/news data, with citations back to source.

# ---- Install (run once per cluster session) ----
# %pip install openai
# dbutils.library.restartPython()

from openai import OpenAI

# =========================================================
# PART 1: Set up the LLM client
# =========================================================

openai_key = dbutils.secrets.get(scope="financial-rag", key="open-api-key")
client = OpenAI(api_key=openai_key)


# =========================================================
# PART 2: Build the RAG prompt with citation instructions
# =========================================================

def build_prompt(query: str, retrieved_chunks: list) -> str:
    """
    Construct a prompt that gives the LLM the retrieved context and
    explicit instructions to cite which source each part of its answer
    comes from - this is what makes the answer trustworthy/checkable
    instead of just a confident-sounding guess.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        meta = chunk["metadata"]
        source_label = f"[Source {i+1}: {meta['ticker']} - {meta['source_type']} - {meta.get('doc_subtype', 'N/A')}]"
        context_blocks.append(f"{source_label}\n{chunk['text']}")

    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a financial research assistant. Answer the question using ONLY the information in the sources below. Do not use outside knowledge.

CITATION RULES (strict):
- EVERY sentence that states a fact must end with a citation like [Source 1].
- Cite ONLY the specific source(s) that actually contain that exact fact - do not cite every source on every sentence.
- Most sentences should have exactly ONE citation. Only use multiple citations [Source 1][Source 2] if the sentence genuinely combines distinct facts from different sources.
- Do not write any factual sentence without a citation attached.
- If the sources don't contain enough information to answer the question, say so explicitly rather than guessing.

Example of correctly formatted output (note each sentence cites only its actual source, not all sources):
"Apple's revenue grew 5% year over year [Source 1]. This growth was driven primarily by services [Source 2]. The company also announced a new product line [Source 1]."

SOURCES:
{context_text}

QUESTION: {query}

ANSWER (cite only the specific source for each individual claim):"""

    return prompt


# =========================================================
# PART 3: The full RAG function - retrieve + generate
# =========================================================

def rag_answer(query: str, k: int = 5, ticker_filter: str = None) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks, then generate a grounded,
    cited answer using an LLM.

    Returns a dict with the answer text AND the sources used, so the citation
    can actually be verified/displayed - not just claimed by the model.
    """
    retrieved_chunks = retrieve(query, k=k, ticker_filter=ticker_filter)

    if not retrieved_chunks:
        return {
            "answer": "No relevant information found in the knowledge base for this question.",
            "sources": [],
        }

    prompt = build_prompt(query, retrieved_chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cost-efficient, strong enough for this use case
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # low temperature - we want grounded, consistent answers, not creative ones
    )

    answer_text = response.choices[0].message.content

    return {
        "answer": answer_text,
        "sources": [
            {
                "source_num": i + 1,
                "ticker": chunk["metadata"]["ticker"],
                "source_type": chunk["metadata"]["source_type"],
                "doc_subtype": chunk["metadata"].get("doc_subtype"),
                "text_preview": chunk["text"][:200],
            }
            for i, chunk in enumerate(retrieved_chunks)
        ],
    }


# =========================================================
# PART 4: Test against the brief's example questions
# =========================================================

def print_rag_result(result: dict):
    print("\nANSWER:")
    print(result["answer"])
    print("\nSOURCES USED:")
    for s in result["sources"]:
        print(f"  [{s['source_num']}] {s['ticker']} - {s['source_type']} - {s['doc_subtype']}")
        print(f"      \"{s['text_preview']}...\"")


test_questions = [
    "What are the key risk factors in Apple's latest 10-K?",
    "Summarize analyst sentiment on REITs this quarter",
    "How is Microsoft's Azure cloud business performing?",
]

for q in test_questions:
    print(f"\n{'='*70}")
    print(f"QUESTION: {q}")
    print('='*70)
    result = rag_answer(q, k=5)
    print_rag_result(result)

# COMMAND ----------

# RAG Evaluation
# Runs a fixed set of test questions through the RAG system and scores them
# two ways: (1) LLM-as-judge for answer quality/groundedness, (2) simple
# retrieval checks (did we get chunks from the expected ticker/source?).
# This turns ad-hoc "looks good" spot checks into documented, defensible numbers.

import json

# =========================================================
# PART 1: The evaluation test set
# =========================================================
# Each question has an expected_ticker (what company it should retrieve about)
# and expected_source_type (sec_filing or news_article, or "any" if either is fine).
# This lets us check retrieval correctness automatically, separate from whether
# the final generated answer reads well.

EVAL_QUESTIONS = [
    {"question": "What are the key risk factors in Apple's latest 10-K?", "expected_ticker": "AAPL", "expected_source_type": "sec_filing"},
    {"question": "How is Microsoft's Azure cloud business performing?", "expected_ticker": "MSFT", "expected_source_type": "sec_filing"},
    {"question": "What are Realty Income's dividend distribution policies?", "expected_ticker": "O", "expected_source_type": "sec_filing"},
    {"question": "What intellectual property risks does Apple face?", "expected_ticker": "AAPL", "expected_source_type": "sec_filing"},
    {"question": "What recent news has come out about Apple and Chinese memory chip suppliers?", "expected_ticker": "AAPL", "expected_source_type": "news_article"},
    {"question": "What is Microsoft's revenue growth in Intelligent Cloud?", "expected_ticker": "MSFT", "expected_source_type": "sec_filing"},
    {"question": "How does Realty Income describe its business model?", "expected_ticker": "O", "expected_source_type": "sec_filing"},
    {"question": "What macroeconomic factors does Apple cite as risks?", "expected_ticker": "AAPL", "expected_source_type": "sec_filing"},
    {"question": "What tariff-related risks does Apple disclose?", "expected_ticker": "AAPL", "expected_source_type": "sec_filing"},
    {"question": "What does Microsoft say about its Productivity and Business Processes segment?", "expected_ticker": "MSFT", "expected_source_type": "sec_filing"},
    {"question": "What is Apple's approach to third-party software developers?", "expected_ticker": "AAPL", "expected_source_type": "sec_filing"},
    {"question": "What is Realty Income's Adjusted EBITDAre and why does it matter?", "expected_ticker": "O", "expected_source_type": "sec_filing"},
    {"question": "What is the outlook for Apple's stock price this year?", "expected_ticker": "AAPL", "expected_source_type": "any"},  # intentionally hard - likely not answerable from filings/news alone
    {"question": "What legal proceedings is Apple currently involved in?", "expected_ticker": "AAPL", "expected_source_type": "sec_filing"},
    {"question": "How has foreign currency impacted Microsoft's reported revenue?", "expected_ticker": "MSFT", "expected_source_type": "sec_filing"},
]


# =========================================================
# PART 2: Run each question through the RAG system, capture results
# =========================================================

def run_evaluation(questions: list) -> list:
    """
    Runs every eval question through rag_answer() (defined in the RAG chain
    notebook - must be run in the same session) and records the answer,
    sources used, and basic retrieval-correctness checks.
    """
    results = []

    for item in questions:
        q = item["question"]
        expected_ticker = item["expected_ticker"]
        expected_source = item["expected_source_type"]

        result = rag_answer(q, k=5)

        retrieved_tickers = [s["ticker"] for s in result["sources"]]
        retrieved_source_types = [s["source_type"] for s in result["sources"]]

        # Retrieval check: did we get AT LEAST ONE chunk matching the expected ticker?
        ticker_match = expected_ticker in retrieved_tickers

        # Source-type check: only meaningful if we specified something other than "any"
        source_match = (
            expected_source == "any" or expected_source in retrieved_source_types
        )

        # Did the model decline to answer (a sign of correct hallucination avoidance,
        # not necessarily a failure - flagged separately for manual review)
        declined = any(
            phrase in result["answer"].lower()
            for phrase in ["cannot provide", "do not contain", "not available in the sources", "cannot answer"]
        )

        results.append({
            "question": q,
            "expected_ticker": expected_ticker,
            "expected_source_type": expected_source,
            "answer": result["answer"],
            "retrieved_tickers": retrieved_tickers,
            "retrieved_source_types": retrieved_source_types,
            "ticker_match": ticker_match,
            "source_match": source_match,
            "declined_to_answer": declined,
            "num_sources": len(result["sources"]),
        })

    return results


# =========================================================
# PART 3: LLM-as-judge scoring for answer quality
# =========================================================

def judge_answer(question: str, answer: str) -> dict:
    """
    Uses the LLM itself to score whether the answer is well-grounded,
    relevant, and appropriately cited - a second, independent quality check
    beyond just "did retrieval find the right ticker."
    """
    judge_prompt = f"""You are evaluating the quality of an AI-generated answer to a financial research question.

QUESTION: {question}

ANSWER: {answer}

Score the answer on a scale of 1-5 for each of the following, and respond ONLY in valid JSON format (no other text):
{{
  "groundedness": <1-5, does the answer appear to be based on cited sources rather than general knowledge>,
  "relevance": <1-5, does the answer actually address the question asked>,
  "citation_quality": <1-5, are claims properly attributed to numbered sources>,
  "brief_reasoning": "<one sentence explaining the scores>"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"groundedness": None, "relevance": None, "citation_quality": None, "brief_reasoning": "JSON parse error"}


# =========================================================
# PART 4: Run it all and summarize
# =========================================================

print("Running evaluation on all questions...\n")
eval_results = run_evaluation(EVAL_QUESTIONS)

print("Scoring answers with LLM-as-judge...\n")
for r in eval_results:
    judge_scores = judge_answer(r["question"], r["answer"])
    r.update(judge_scores)

# ---- Summary statistics ----
total = len(eval_results)
ticker_match_rate = sum(r["ticker_match"] for r in eval_results) / total * 100
source_match_rate = sum(r["source_match"] for r in eval_results) / total * 100
avg_groundedness = sum(r["groundedness"] for r in eval_results if r["groundedness"]) / total
avg_relevance = sum(r["relevance"] for r in eval_results if r["relevance"]) / total
avg_citation = sum(r["citation_quality"] for r in eval_results if r["citation_quality"]) / total
decline_count = sum(r["declined_to_answer"] for r in eval_results)

print("="*60)
print("EVALUATION SUMMARY")
print("="*60)
print(f"Total questions evaluated:     {total}")
print(f"Retrieval - correct ticker:    {ticker_match_rate:.1f}%")
print(f"Retrieval - correct source:    {source_match_rate:.1f}%")
print(f"Avg groundedness (1-5):        {avg_groundedness:.2f}")
print(f"Avg relevance (1-5):           {avg_relevance:.2f}")
print(f"Avg citation quality (1-5):    {avg_citation:.2f}")
print(f"Questions declined (no hallucination): {decline_count}")

# Save full results to a Delta table for reference / README screenshots
results_df = spark.createDataFrame(eval_results)
spark.sql("CREATE SCHEMA IF NOT EXISTS eval")
results_df.write.format("delta").mode("overwrite").saveAsTable("eval.rag_evaluation_results")

print("\nFull results written to eval.rag_evaluation_results")
display(results_df.select("question", "ticker_match", "source_match", "groundedness", "relevance", "citation_quality"))

# COMMAND ----------

from pyspark.sql.functions import col

low_citation = (
    spark.table("eval.rag_evaluation_results")
    .filter(col("citation_quality") <= 3)
    .select("question", "citation_quality", "brief_reasoning")
)
display(low_citation)

# COMMAND ----------

tariff_row = spark.table("eval.rag_evaluation_results").filter(
    col("question") == "What tariff-related risks does Apple disclose?"
).select("answer", "brief_reasoning").collect()[0]

print("ANSWER:", tariff_row["answer"])
print("\nJUDGE REASONING:", tariff_row["brief_reasoning"])


# COMMAND ----------

# MLflow Experiment Tracking for the RAG System
# Wraps rag_answer() so every query gets logged as an MLflow run - capturing
# the question, retrieved sources, the exact prompt sent to the LLM, the
# model used, the generated answer, and (optionally) eval scores.
# This is what makes the system OBSERVABLE - you can look back at any past
# query and see exactly what happened, not just the final answer.

import mlflow
import time

# Set/create an experiment to group all RAG runs together
mlflow.set_experiment("/Shared/financial_rag_experiments")


def rag_answer_tracked(query: str, k: int = 5, ticker_filter: str = None) -> dict:
    """
    Same as rag_answer(), but wraps the whole call in an MLflow run that logs:
    - Parameters: the query, k, ticker_filter, model name
    - Metrics: latency, number of sources retrieved
    - Artifacts: the full prompt sent to the LLM, and the full answer received
    This turns every RAG query into a reviewable, comparable experiment run.
    """
    with mlflow.start_run(run_name=f"rag_query_{int(time.time())}"):
        start_time = time.time()

        # Log the inputs before we even call the model
        mlflow.log_param("query", query)
        mlflow.log_param("k", k)
        mlflow.log_param("ticker_filter", ticker_filter or "none")
        mlflow.log_param("model", "gpt-4o-mini")
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")

        # Run the actual RAG pipeline (reuses your existing function)
        result = rag_answer(query, k=k, ticker_filter=ticker_filter)

        latency = time.time() - start_time

        # Log metrics - things you'd want to track/compare across many runs
        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_metric("num_sources_retrieved", len(result["sources"]))

        # Log the full answer and prompt as text artifacts - lets you go back
        # and read exactly what happened for any specific past query
        mlflow.log_text(result["answer"], "answer.txt")

        sources_summary = "\n".join(
            f"[{s['source_num']}] {s['ticker']} - {s['source_type']} - {s['doc_subtype']}"
            for s in result["sources"]
        )
        mlflow.log_text(sources_summary, "sources.txt")

        # Tag the run with the tickers actually retrieved, for easy filtering
        # later in the MLflow UI (e.g., "show me all AAPL-related runs")
        retrieved_tickers = list(set(s["ticker"] for s in result["sources"]))
        mlflow.set_tag("tickers_used", ",".join(retrieved_tickers))

        return result


# =========================================================
# Test: run a few queries through the tracked version
# =========================================================

test_queries = [
    "What are the key risk factors in Apple's latest 10-K?",
    "How is Microsoft's Azure cloud business performing?",
    "What are Realty Income's dividend distribution policies?",
]

for q in test_queries:
    print(f"\nRunning tracked query: {q}")
    result = rag_answer_tracked(q, k=5)
    print(f"  -> Answer length: {len(result['answer'])} chars, Sources: {len(result['sources'])}")

print("\nAll runs logged to MLflow experiment: /Shared/financial_rag_experiments")
print("View them in the Databricks UI under the 'Experiments' icon in the left sidebar.")

# COMMAND ----------

