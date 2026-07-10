# Engineering Log — Real Issues Solved (README + Interview Prep)

This document tracks every real technical problem hit while building this project,
the root cause, the fix, and why it's a genuinely good thing to talk about in an
interview. Debugging stories like these are usually MORE convincing to an interviewer
than "it just worked" - they prove you understand the systems, not just the happy path.

---

## 1. Azure "exhausted credits" error was actually a resource group mismatch
**What happened:** Cluster creation failed with "cannot run Cluster because you've
exhausted your available credits," despite the account showing $174+ available.
**Root cause:** Was creating the cluster in the wrong resource group - the error
message was misleading and didn't point directly at the actual cause.
**Fix:** Switched to the correct resource group; cluster created successfully.
**Interview angle:** Shows you don't take error messages at face value - you verify
against actual account state (checked Cost Management directly) before assuming
the obvious explanation is correct.

---

## 2. Azure RBAC: "Owner" role does NOT grant Key Vault secret access
**What happened:** Had subscription-level "Owner" role, but still got a 403
Forbidden error trying to add a secret to Key Vault.
**Root cause:** Azure RBAC on Key Vault separates control-plane access
(managing the vault itself - settings, policies) from data-plane access
(reading/writing actual secret values). Owner covers control-plane; it does
NOT automatically grant data-plane secret operations.
**Fix:** Explicitly assigned the "Key Vault Secrets Officer" role (read/write)
to my own user account, separate from the Owner role already held.
**Interview angle:** A genuinely subtle, real-world cloud security concept - most
junior engineers assume "Owner = full access to everything." Being able to explain
control-plane vs. data-plane separation shows real IAM understanding, not just
clicking through wizards.

---

## 3. Wrong identity assigned Key Vault access - human user vs. service principal
**What happened:** Databricks still couldn't read secrets even after assigning
"Key Vault Secrets User" to a "Databricks-admin" account.
**Root cause:** The actual caller trying to fetch secrets was a distinct
first-party service principal (AzureDatabricks, a Microsoft-owned Enterprise
Application), not the human admin account. The error message's appid field
revealed the real caller identity.
**Fix:** Found and assigned the role to the correct AzureDatabricks service
principal instead.
**Interview angle:** Debugging by reading the actual appid/oid in a raw API
error, rather than guessing, is a real diagnostic skill. Also a good example of
"human identity vs. machine identity" - a distinction that matters a lot in any
cloud IAM setup.

---

## 4. RBAC role propagation delay
**What happened:** Correct role assigned, but still got 403 errors immediately after.
**Root cause:** Azure RBAC role assignments can take several minutes to propagate,
even though the portal shows them as already active.
**Fix:** Waited ~5 minutes, refreshed auth session (closed/reopened portal), retried.
**Interview angle:** Real distributed systems have propagation delay/eventual
consistency even in "instant"-looking admin UIs - a broadly transferable lesson,
not just an Azure quirk.

---

## 5. SEC filings extraction: Inline XBRL polluting document text
**What happened:** After chunking and embedding, risk_density was 0 across
every SEC filing chunk, and word counts were suspiciously low (~89 words per
600-token chunk vs. expected ~450).
**Root cause:** Modern SEC filings use Inline XBRL (iXBRL) - machine-readable
financial tag data embedded directly inside the same HTML document as the
human-readable filing text, often scattered in many small ix:... tagged
elements rather than one clearly hidden block. Naive BeautifulSoup tag-stripping
extracted both the real text AND all this tag data, and the tag data dominated
the early chunks.
**Fix:** Added regex-based pre-processing to strip ix:header blocks and all
ix:-namespaced tags before HTML parsing, plus removal of explicitly
display:none elements as a backup.
**Diagnosis method:** Didn't just guess - printed the actual chunk text, saw
literal XBRL tag names (xbrli:shares, us-gaap:...) in what should have been
prose, and traced it to the iXBRL structure.
**Interview angle:** This is probably the single best story in the whole project.
It's a real, non-obvious data engineering problem (most people don't know SEC
filings work this way), you diagnosed it from evidence rather than guessing, and
the fix required understanding a real document format, not just calling a library
function. Good answer to "tell me about a time you debugged a tricky data issue."

---

## 6. NewsAPI relevance filter was too strict, silently dropping valid data
**What happened:** After building a keyword-based relevance filter to cut news
noise (e.g., a Chevrolet Camaro listing showing up under "AAPL"), the AAPL ticker
ended up with ZERO news articles passing the filter.
**Root cause:** The filter required specific phrases like "apple inc" or "tim cook,"
but real headlines just used the plain word "Apple" (e.g., "Apple weighs buying
RAM from Chinese suppliers") - too strict a match, so real articles failed it.
**Fix:** Switched to a word-boundary regex match on the plain company name
(\bapple\b), which still correctly excluded true noise (since irrelevant
articles never mentioned the company name at all) while no longer over-filtering.
**Interview angle:** A good example of iterating on a first-pass solution after
seeing real output, not just shipping the first idea - and knowing how to verify
a filter isn't silently breaking things (checking per-ticker counts, not just
"did the count go down").

---

## 7. PySpark UDF pickling error with tiktoken (Rust-based library)
**What happened:** TypeError: cannot pickle 'builtins.CoreBPE' object when
running the chunking UDF.
**Root cause:** tiktoken's core tokenizer is implemented in Rust. When it was
instantiated once at module level and referenced inside a Spark UDF, Spark tried
to serialize (pickle) that object to ship to worker processes - but Rust objects
generally aren't picklable in Python.
**Fix:** Moved the tokenizer instantiation INSIDE each function call, so every
worker creates its own local instance instead of receiving a shared object.
**Interview angle:** Understanding how Spark distributes work across workers
(and what that means for object serialization) is a real distributed-computing
concept, not just a Python syntax fix.

---

## 8. huggingface_hub / sentence-transformers version incompatibility
**What happened:** ImportError: cannot import name 'cached_download' from
huggingface_hub.
**Root cause:** The installed sentence-transformers version called a
huggingface_hub function (cached_download) that was removed in newer
huggingface_hub releases - a dependency version mismatch.
**Fix:** Pinned specific, mutually-compatible versions of sentence-transformers,
huggingface_hub, transformers, and tokenizers together.
**Interview angle:** Dependency management / version pinning is an unglamorous
but very real and common production issue - worth mentioning as evidence you
know how to read a stack trace back to its actual cause (a removed API) rather
than just re-running the install blindly.

---

## 9. ChromaDB fails on DBFS-mounted storage
**What happened:** InternalError: Operation not supported (os error 95) when
initializing a persistent ChromaDB client pointed at a /dbfs/... path.
**Root cause:** ChromaDB's persistent storage uses SQLite, which requires
low-level file operations (memory-mapping, file locking) that DBFS's FUSE-based
mount doesn't support.
**Fix:** Pointed ChromaDB at local disk (/local_disk0/...) on the cluster's
driver node instead of DBFS. Tradeoff: the vector index doesn't persist across
cluster restarts, but since it's rebuilt from the Gold Delta table each run,
this is an acceptable tradeoff at this project's scale.
**Interview angle:** Good example of knowing the right storage tool for the job -
being able to explain WHY DBFS and SQLite are incompatible (not just "I moved
the path and it worked") shows you understand the underlying filesystem
mechanics, not just trial-and-error fixing.

---

## 10. RAG system correctly refused to answer when data didn't support it
**What happened:** When asked "Summarize analyst sentiment on REITs this quarter,"
the RAG system responded that it could not answer, since the retrieved chunks
only contained financial metrics (FFO, EBITDA) rather than analyst sentiment.
**Why this matters:** This is the citation/grounding design working exactly as
intended - the system was explicitly prompted to only use retrieved sources and
say so if the sources don't support an answer, rather than fabricating a
plausible-sounding response.
**Interview angle:** This is strong, concrete evidence of actively testing for
and preventing hallucination, not just building a RAG system and hoping it
behaves. A good direct answer to "how did you handle/test for hallucination in
your RAG system?" - most candidates can only describe this in the abstract;
you have a real example with a screenshot.

---

## 11. OpenAI billing: Codex credits and API billing are separate wallets
**What happened:** Had $100 in "Codex" credits (from a student promotion) but
still got insufficient_quota errors calling the OpenAI API directly.
**Root cause:** Codex credits are scoped exclusively to OpenAI's Codex/ChatGPT
products, not the underlying developer API platform - two separate billing
systems despite both being "OpenAI." A Visa gift card added directly to
platform.openai.com's billing was the actual fix.
**Interview angle:** Minor but real example of reading vendor documentation
carefully rather than assuming "credit" means the same thing across a
company's entire product line - a genuinely common mistake with cloud billing.

---

## 12. Terraform installed via winget but not found on PATH
**What happened:** winget install HashiCorp.Terraform completed successfully
(confirmed via winget list), but terraform -v returned
"command not recognized" even after reopening PowerShell.
**Root cause:** winget installed the binary into a package-specific folder
under AppData\Local\Microsoft\WinGet\Packages\... but did not add that
folder to the system PATH automatically.
**Fix:** Located the actual terraform.exe path via
Get-ChildItem -Recurse -Filter "terraform.exe", then called it directly
using its full path rather than relying on PATH resolution - a reliable
workaround when PATH configuration is unclear or inconsistent across environments.
**Interview angle:** A small, honest example of not getting stuck on a
tooling/environment problem - finding a working alternative (full path
invocation) rather than spending excessive time debugging PATH configuration
that wasn't essential to the actual task (validating the Terraform config).

---

## Pattern across all of these (good closing point for an interview)

Almost none of these were "write code, it works." Every one required: reading
the actual error message/evidence closely, forming a specific hypothesis about
root cause, and verifying the fix rather than assuming it worked. That iterative
diagnose-hypothesize-verify loop is arguably the most transferable skill this
project demonstrates - more than any single tool in the stack.
