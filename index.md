---
title: "I Parsed 10,000 Complex Technical Docs for €50: A Multimodal RAG Survival Guide"
description: "How I built a practically free, serverless knowledge base on GCP using Gemini 1.5 Flash, BigQuery, Langchain and Cloud Run"
---

# I Parsed 10,000 Complex Technical Docs for €50: A Multimodal RAG Survival Guide

**How I built a practically free, serverless knowledge base on GCP using Gemini 1.5 Flash, BigQuery, Langchain and Cloud Run**

![Cover image]({{ '/images/1-cover.jpeg' | relative_url }})

## The Backstory
While I’m writing this in early 2026, this project actually dates back to late 2024 and early 2025. Consequently, the architecture was shaped by the specific tools available during that window. I had to innovate under strict constraints: the entire pipeline needed to run on GCP, adhere to data‑sovereignty requirements, and stick to a tight budget. Back then, I didn’t have the luxury of today’s parsing solutions like Google LangExtract or Mistral’s Document AI. Even Google’s native RAG Engine wasn't available in European regions yet. I realize the landscape has shifted dramatically; for those deploying cutting-edge agentic RAG systems in 2026, the tech stack might look a bit retro. However, this article is less about the specific tools and more about the architectural decision-making process. If you are interested in how to navigate constraints or need to build an AI-powered proprietary document library with minimal processing and maintenance costs, this is still very much worth reading. 
---

## The Problem: 10,000 Unstructured Documents
In 2024, I was approached by the marketing team of a leading industrial manufacturer. They were sitting on a treasure trove of unique data: specialized industry articles, technical journals, and presentations buried in a chaotic labyrinth of Google Drive folders containing roughly 10,000 documents. The corpus was multimodal, consisting of primarily PDFs, but also Word documents, PPTs, and Excel files, all dense with text, tables, charts, and diagrams. Besides that, a considerable portion of the presentations consisted of slide images (photos) without OCR, making the text non-searchable and harder to extract. The ingestion pipeline was immutable: subject-matter experts (non-technical users) manually uploaded files monthly, and this workflow could not be altered. The good news was that along with Drive, I was provisioned a GCP environment. The catch was that I couldn't step outside it. I was restricted strictly to native services: no external APIs and no new vendors. Finally, I had to make it all work on a shoestring budget.
---

## Deliverables

* **Vectorization for RAG:** Convert the entire corpus into embeddings and store it in a vector database for retrieval-augmented generation.
* **Rich metadata extraction: Generate detailed metadata: keywords, summaries, and structured text descriptions of embedded visuals (graphs, charts, and tables) to enable grouping, organization, and document classification.
* **Recognition of photo-based presentations: Parse slide decks made of images (no OCR / non-selectable text) add the extracted content to the vector database.
* **Seamless ingestion (no workflow changes):** Users keep working as usual. New files land in Google Drive, and the system automatically ingests and processes them.
* **Multimodal technical chatbot:** Provide a chatbot that can answer precise technical questions by understanding not only the text, but also charts, tables, and diagrams within the documents.
* **Source-first answers with direct links:** Return citations with links to the relevant pages in GCS, plus the full document, so specialists can jump straight to the original material.
* **Automatic re-ingestion for higher accuracy:** When a relevant document is retrieved, the system can optionally re-ingest it on demand using newer Gemini models integrated into the chatbot to parse it in real time and compensate for any imperfections from batch processing.
* **Scalability as usage grows:** Designed to scale smoothly as adoption increases and the corpus expands, keeping ingestion, retrieval, and response quality performant as the database grows.

**The Result:** I built a fully automated pipeline that parsed everything for **€50 total**. Maintenance costs? Practically zero, thanks to the GCP Free Tier.

Here is the breakdown of how I did it.
---

## Parsing tool

### How I Arrived at a Custom Parsing Solution

Traditional RAG tutorials usually rely on splitting documents into fixed-size, overlapping chunks. While this might work for simple continuous text, it is imperfect even for works like War and Peace, where Tolstoy’s shifts between Russian and French result in disjointed, mixed-language chunks. But how on earth can this method be applied to technical documents? Real-world docs are not linear streams; they are dynamic layouts of text, diagrams, and tables that standard chunking ignores.

![Layout parsing concept]({{ '/images/2-layout.jpeg' | relative_url }})

I quickly realized that to get decent results I needed to stick with **object-centric chunking**, an approach where the parser can identify distinct regions on a page and treat each one as a separate object, complete with the metadata I care about. That meant I needed a tool that could reliably understand page layout. And there was another catch: a chunk of the presentations were **image-based**, with **no OCR**.

As expected, the usual Python suspects like PyPDF2, PyMuPDF, PDFPlumber were basically useless for this. The only library that looked somewhat promising at the time was **unstructured.io**, which was getting a lot of hype. But it still didn’t solve my core requirement: generating **textual descriptions of graphs, diagrams, and images**. For that, you had to build a separate pipeline.

It was a similar story with **Google Cloud Document AI Layout Parser**. The object detection and JSON output were excellent. Honestly, it was probably the best in terms of parsing quality, but at the end of the day, it’s *just* a parser. And then there was the price: **$0.01 per page**. For 100-page documents at scale, the math gets ugly fast. With 10,000 documents:

* 100 pages × 10,000 docs = **1,000,000 pages**
* 1,000,000 pages × $0.01 = **$10,000** just for parsing

Meanwhile, the landscape shifted: Google **slashed the price of Gemini 1.5 Flash**. Input token costs dropped by **78%** to about **$0.075 per million tokens**, and output costs fell to about **$0.30 per million tokens** (for prompts under 128k tokens). Even better, Google treated **each document page as ~258 tokens**.

So the per-page cost looks like this:

**Input cost per page**

```
Cost = 258 × (0.075 / 1,000,000) = $0.00001935 per page (≈ $0.000019)
```

**Output cost**

```
output_tokens × (0.30 / 1,000,000)
```

For a typical page (~650 output tokens):

* Input: $0.00001935
* Output: 650 × (0.30 / 1,000,000) = $0.000195
* **Total per page:** $0.00021435

Now scale it:

* Total pages: **1,000,000**
* Total cost: 1,000,000 × $0.00021435 = **$214.35**

Feel the difference.

And pricing wasn’t the only reason a custom parser built on **Gemini 1.5 Flash** started to look like the right plan.


## What made Gemini 1.5 Flash the best parser beyond affordability?

**Long context.** Gemini 1.5 Flash supports context windows up to **1 million tokens**, which effectively future-proofed my pipeline. In practical terms, that’s roughly **1,000,000 ÷ 258 ≈ 3,876 pages** of PDF content per request. Wildly convenient when you’re dealing with long reports.

**Strong multilingual handling.** The model can process prompts in **100+ languages**, which mattered because our documents weren’t just in one language: we had Chinese, French, English, and others. While most PDF text extractors are technically language-agnostic (they pull Unicode text), real PDFs still break in messy ways: strange font encodings or missing Unicode mappings can produce garbled characters even when the document *looks* fine visually. Gemini handled these edge cases far more gracefully than the typical extraction stack.

**Native multimodal document understanding.** This was the real differentiator. Gemini doesn’t just extract text, it can interpret **images, diagrams, charts, and tables** inside PDFs, pulling actual values from visuals (like bar charts) rather than relying on nearby text.

That multimodal capability unlocked a few practical wins for the pipeline:

* **Direct PDF ingestion** (using the PDF MIME type) with no preprocessing.
* **LLM features out of the box:** summaries, keyword extraction, chart/diagram descriptions, and table understanding.
* **No need for a special pipeline for photo-based slides.** Even when slides were embedded as images **without any OCR preprocessing**, Gemini handled them like any other PDF.
* **A foundation for object-oriented chunking**, where the model can **classify and separate** page elements (text blocks, tables, figures, captions) instead of treating everything as one flat stream.

**Structured output with controlled generation.** Gemini’s schema-guided responses let me define a JSON structure and reliably get output that conforms to it. That removed a lot of brittle post-processing and made the extraction step feel closer to deterministic parsing. I could spend time on business logic instead of cleaning model output.

**A “good” kind of hallucination (in my use case).** One unexpected benefit: when PDFs contained line breaks that split words or phrases across lines, the model often reconstructed them correctly. It wasn’t always character-perfect, but it preserved the meaning and the resulting chunks were cleaner for retrieval. Since our workflow already required specialists to verify LLM answers against the source document, the tradeoff was worth it: I ended up vectorizing more readable text.

**Surprisingly good math handling.** Gemini also did a nice job normalizing math expressions that used special Unicode symbols, converting them into clean, readable formulas rather than leaving behind broken glyphs.


### Is Gemini 1.5 Flash Truly an Out-of-the-Box Parser?

It would be a developer’s **idyll** if everything stayed as smooth as the performance described above. Of course, **real life is rarely a bed of roses**, and I eventually encountered a single technical constraint that significantly shaped the parser's architecture: **the Output Token Limit.**

While Gemini 1.5 Flash has a massive *input* context window, its *output* is capped at **8,192 tokens**. 

8192 output tokens ÷ 650 ≈ 12.60 pages 

If I were to submit a 50-page technical manual for a full JSON extraction of every chart and table, the output would be substantial. It would inevitably exceed the 8,192-token limit, resulting in either a truncated response or an oversimplified summary rather than a complete parse.

Even if the document is closer to upper limit of this 12.60 pages, I noticed that sending the whole file caused the model to "compress" information - it would summarize too aggressively and skip the minute technical details I needed. So it literally refuses to work as a parser.

To address the token limits, I designed a **Two-Pass Approach**:

**Pass A: Global Context Extraction (Whole Document)**
I submitted the full document to Gemini in a single call to extract high-level, document-wide metadata. This step was crucial for establishing context that isolated pages lack. For example, "Page 57" does not know the document's author, creation date, or type (e.g., research paper vs. manual), information typically found only on the cover. This pass also generated a comprehensive summary of the file.

**Pass B: Granular Page Parsing**
For detailed data extraction, I divided the document into individual pages. By treating each page as a discrete unit, Gemini 1.5 Flash worked as an ideal parser, delivering results in line with all the powerful features described above. This granular approach captures the specific object chunks and strict JSON schema adherence that can be lost or truncated when attempting to process a massive document in a single shot.

**Parallel Execution Strategy**
To operationalize this, the "Whole Document" context request and the individual "Page-by-Page" requests were all treated as distinct API calls, each accompanied by its own prompt. These requests were executed concurrently to maximize throughput, while being carefully throttled to respect the Vertex AI rate limit of ~200 requests per minute (RPM).

### From Concept to Implementation: The Parsing + RAG Pipeline

Now that we’ve covered the conceptual rationale, let’s shift into the engineering: the end-to-end parsing and RAG pipeline: how files are discovered, normalized into PDFs, parsed in layers with Gemini, and ultimately indexed into BigQuery for retrieval.

![Pipeline diagram]({{ '/images/3-pipeline.jpeg' | relative_url }})

1. Discover files in Google Drive folders.
2. Compare against file registry table (new vs already tracked).
3. Copy new files to GCS and convert supported formats to PDF (with concurrent Drive->GCS copying).
4. Parse each document in three layers (with concurrent document/page ingestion to Gemini within API limits):
   - Programmatic prep layer (`metadata["properties"]`): IDs, hashes, links, layout, and session metadata
   - Document LLM layer (`gen_result`): title/author/type/keywords/summary
   - Page layer (`all_chunks`): object-centric chunk extraction with page-level and session fields
5. Save intermediate JSON artifacts to GCS (document-level, page-level, merged).
6. Merge per-document outputs into session NDJSON.
7. Batch load NDJSON into BigQuery chunks table.
8. Update file registry statuses (`copied_to_gcs`, `parsed`, `added_to_bq`, timestamps).
9. Add new chunks to BigQuery vector store via LangChain `BigQueryVectorStore` (filtering out very short chunks, e.g. <300 chars).


### Implementation Walkthrough: Code-Level Highlights

## 1) Control Plane First: `FILE_LIST`

The pipeline starts with a state table, so each file can be tracked across discovery, copy, parsing, and corpus update.
Key libraries: `google-cloud-bigquery`

| Field | Type | Used for |
|---|---|---|
| `file_number` | `INTEGER` | Stable numeric identifier across stages |
| `file_id` | `STRING` | Unique Drive ID for deduplication |
| `file_name` | `STRING` | Human-readable source name |
| `file_mime_type` | `STRING` | Conversion branch selection |
| `file_created_time` | `TIMESTAMP` | Source chronology and filtering |
| `file_path` | `STRING` | Folder lineage for traceability |
| `web_view_link` | `STRING` | Direct source link in Drive |
| `file_size` | `FLOAT` | Monitoring and throughput planning |
| `copied_to_gcs` | `BOOL` | Copy stage completion flag |
| `timestamp_copied` | `DATETIME` | Copy stage timestamp |
| `parsed` | `STRING` | Parse status (`false`, `failed`, `parsed`) |
| `timestamp_parsed` | `DATETIME` | Parse stage timestamp |
| `added_to_bq` | `BOOL` | Corpus insertion completion flag |
| `timestamp_added` | `DATETIME` | Corpus insertion timestamp |

Snippet A shows the incremental principle: compare discovered files to known `file_id` values.

```python
# table_id, bq_client, master_files are prepared above
query = f"SELECT file_id, file_number FROM `{table_id}`"
rows = list(bq_client.query(query))

existing_file_ids = {row["file_id"] for row in rows}
new_files = [f for f in master_files if f["file_id"] not in existing_file_ids]
```
Source files: `01_setup_bq_file_list.ipynb (cell 2 lines 20-35; cell 4 lines 21-24)`, `03_parsing_tool.ipynb (cell 10 lines 8-21)`

## 2) Drive Access and Resilient Discovery

Drive access is built from a service account secret; listing is shared-drive aware.
Key libraries: `google-cloud-secret-manager`, `google.oauth2.service_account`, `googleapiclient`

```python
# get_secret(...) is defined earlier
service_account_info = json.loads(get_secret("your-drive-service-account-json"))
credentials = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=SCOPES,
)
drive_service = build("drive", "v3", credentials=credentials)
```
Source files: `03_parsing_tool.ipynb (cell 8 lines 1-33)`

Discovery calls use exponential backoff for transient API failures.

```python
def robust_files_list(service, query, max_retries=5, initial_wait=1):
    wait = initial_wait
    for attempt in range(max_retries + 1):
        try:
            return service.files().list(
                q=query,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            ).execute()
        except HttpError as e:
            if e.resp.status in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(wait); wait *= 2
            else:
                raise
```
Source files: `03_parsing_tool.ipynb (cell 8 lines 34-63)`


## 3) Copy/Convert Stage: Normalize to PDF

All supported formats are normalized to PDF before parsing, so downstream logic has one input type.
Google Workspace MIME types (`application/vnd.google-apps.*`) are exported directly to PDF.
Office MIME types are downloaded, temporarily converted, and then exported to PDF.
Key libraries: `googleapiclient.http (MediaIoBaseDownload/MediaIoBaseUpload)`, `googleapiclient.discovery`

```python
# file, file_id, drive_service, media_body, file_name_no_ext are prepared above
# GOOGLE_WORKSPACE_TYPES / OFFICE_TYPES / target_mime_type prepared above
if file["file_mime_type"] in GOOGLE_WORKSPACE_TYPES:
    request = drive_service.files().export_media(fileId=file_id, mimeType="application/pdf")
elif file["file_mime_type"] in OFFICE_TYPES:
    request = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)  # download source
    uploaded = drive_service.files().create(
        body={"name": file_name_no_ext, "mimeType": target_mime_type},
        media_body=media_body,
        fields="id",
    ).execute()  # temporary Google-native copy
    request = drive_service.files().export_media(fileId=uploaded["id"], mimeType="application/pdf")
    drive_service.files().delete(fileId=uploaded["id"]).execute()
else:
    request = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
```
Source files: `03_parsing_tool.ipynb (cell 12 lines 90-177)`

Parallel workers use thread-local Drive clients to avoid shared mutable client state.

```python
# credentials is prepared above
thread_local = threading.local()

def get_drive_service():
    if not hasattr(thread_local, "drive_service"):
        thread_local.drive_service = build("drive", "v3", credentials=credentials)
    return thread_local.drive_service
```
Source files: `03_parsing_tool.ipynb (cell 12 lines 53-60)`

## 4) Parallelism Model: Copy + Page-Aware Batches

Copying happens concurrently at file level.
Key libraries: `concurrent.futures.ThreadPoolExecutor`

```python
# num_threads, process_file, files_to_process are prepared above
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_file, f) for f in files_to_process]
    total_copied = 0
    for future in as_completed(futures):
        if future.result():
            total_copied += 1
```
Source files: `03_parsing_tool.ipynb (cell 12 lines 252-260)`

Parsing concurrency is shaped by page counts under an explicit `API_LIMIT`.

```python
# file_page_counts is prepared above
API_LIMIT = 100
batches, current_batch, current_pages = [], [], 0

for file_number, num_pages in file_page_counts:
    if current_pages + num_pages <= API_LIMIT:
        current_batch.append(file_number); current_pages += num_pages
    else:
        batches.append(current_batch)
        current_batch, current_pages = [file_number], num_pages

if current_batch:
    batches.append(current_batch)
```
Source files: `03_parsing_tool.ipynb (cell 35 lines 61-187)`

Inside each file, the whole-document task and all page tasks are also executed in parallel.

```python
# gsutil_uri and blobs are prepared above
tasks = [("gen", gsutil_uri)]
tasks.extend(("page", blob) for blob in blobs)

with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
    future_to_task = {}
    for task_type, payload in tasks:
        future = executor.submit(process_pdf_gen if task_type == "gen" else process_page, payload, bucket_name)
        future_to_task[future] = task_type
    for future in as_completed(future_to_task):
        if future_to_task[future] == "gen":
            gen_result, file_base_name, base_folder = future.result()
        else:
            all_chunks.extend(future.result() or [])
```
Source files: `03_parsing_tool.ipynb (cell 33 lines 233-256)`

## 5) Parsing Logic by Derivation Source

#### 1) Programmatic foundation (non-LLM, first)

Programmatic extraction creates the operational backbone of each record: stable IDs, file/page links, hashes, session timestamps, status flags, and token/model telemetry.  
This layer is deterministic and is used for traceability, reruns, debugging, and cost analysis.

#### 2) Whole-document LLM pass (second)

A single pass over the full document extracts global context that page-level parsing cannot reliably infer:

- document name
- author
- type
- document-level keywords
- document-level summary

These fields are then attached to chunk records to improve grouping, thematic search, and retrieval context quality.

#### 3) Per-page LLM chunk pass (third)

Each page is parsed independently to avoid long-document token compression/truncation and to preserve local detail.  
Chunking is object-centric rather than length-centric, with chunk types:

- `text`
- `table`
- `chart`
- `diagram`
- `image` (with filtering of irrelevant visuals when possible)

For each chunk, the model extracts type/title/content/keywords/summary using a fixed JSON schema.

Both LLM passes (whole-document and per-page) use schema-constrained JSON outputs, then programmatic metadata is merged before loading to BigQuery.

For exact field-level mapping (`a_` to `ar_`) and ownership (programmatic vs LLM), see the field map table below.

### Field Map (`a_` to `ar_`)

The table below maps all schema fields to:

- scope: `Document` or `Page`
- derivation: LLM vs programmatic
- intent: what the field is used for

| Field | Scope | Derived By | Purpose |
|---|---|---|---|
| `a_chunk_id` | Page | Programmatic | Generated as UUID (`uuid4`) |
| `b_page_number` | Page | Programmatic | Page-level citation and navigation |
| `c_file_pages` | Document | Programmatic | Completeness checks and pagination validation |
| `d_document_date` | Document | LLM | Time-based filtering and trend analysis |
| `e_chunk_type` | Page | LLM | Modality-aware retrieval and routing |
| `f_chunk_title` | Page | LLM | Better snippet readability and ranking hints |
| `g_chunk_contents` | Page | LLM | Main searchable text for embeddings and retrieval |
| `h_chunk_keywords` | Page | LLM | Recall boost for sparse or domain-specific queries |
| `i_chunk_summary` | Page | LLM | Fast preview and reranking signal |
| `j_file_number` | Document | Programmatic | Generated as sequential numeric prefix |
| `k_file_hash` | Document | Programmatic | Generated as SHA-256 hash |
| `l_file_name` | Document | Programmatic | Human-readable identifier in UI/results |
| `m_file_layout` | Document | Programmatic | Parsing diagnostics by layout/orientation |
| `n_document_name` | Document | LLM | Corpus grouping and source attribution |
| `o_document_author` | Document | LLM | Author-based filtering and credibility context |
| `p_document_type` | Document | LLM | Type-based filtering (`Research Paper`, `Report`, `Presentation`, `Publication`, `Technical Documentation`, `Other`) |
| `q_document_keywords` | Document | LLM | Thematic indexing at document level |
| `r_document_summary` | Document | LLM | High-level context for ranking and answers |
| `s_page_gsutil` | Page | Programmatic | Direct page file access for multimodal calls |
| `t_page_url` | Page | Programmatic | Click-through evidence link for users |
| `u_file_gsutil` | Document | Programmatic | Direct full-document file access |
| `v_file_url` | Document | Programmatic | Browser access to full source |
| `w_file_session_start` | Document | Programmatic | Start marker for runtime tracking |
| `x_file_session_end` | Document | Programmatic | End marker for runtime tracking |
| `y_file_session_duration` | Document | Programmatic | Processing latency monitoring |
| `z_file_session_status` | Document | Programmatic | Retry/control decisions and alerting |
| `aa_document_session_id` | Document | Programmatic | Generated as UUID (`uuid4`) for document-level session tracking |
| `ab_document_session_start` | Document | Programmatic | Timing audit for document-stage inference |
| `ac_document_session_end` | Document | Programmatic | Timing audit for document-stage inference |
| `ad_document_session_duration` | Document | Programmatic | Duration metric for document-stage inference |
| `ae_chunks_session_id` | Page | Programmatic | Generated as UUID (`uuid4`) for page-level session tracking |
| `af_chunks_session_start` | Page | Programmatic | Timing audit for page-stage inference |
| `ag_chunks_session_end` | Page | Programmatic | Timing audit for page-stage inference |
| `ah_chunks_session_duration` | Page | Programmatic | Duration metric for page-stage inference |
| `ai_document_finish_reason` | Document | Programmatic | Diagnose stop conditions in model output |
| `aj_chunks_finish_reason` | Page | Programmatic | Diagnose stop conditions in model output |
| `ak_document_prompt_token_count` | Document | Programmatic | Prompt-side cost tracking |
| `al_chunks_prompt_token_count` | Page | Programmatic | Prompt-side cost tracking |
| `am_document_candidates_token_count` | Document | Programmatic | Output-side cost tracking |
| `an_chunks_candidates_token_count` | Page | Programmatic | Output-side cost tracking |
| `ao_document_total_token_count` | Document | Programmatic | Total token budget monitoring |
| `ap_chunks_total_token_count` | Page | Programmatic | Total token budget monitoring |
| `aq_document_model_version` | Document | Programmatic | Reproducibility and regression comparison |
| `ar_chunks_model_version` | Page | Programmatic | Reproducibility and regression comparison |

## 6) Schema-Constrained Parsing

The pipeline uses one schema principle across passes: force structured JSON outputs.
Key libraries: `vertexai.generative_models (GenerativeModel, GenerationConfig)`

```python
response_schema_chunks = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "e_chunk_type": {"type": "string", "enum": ["text", "table", "chart", "graph", "diagram", "image"]},
            "g_chunk_contents": {"type": "string"},
            "h_chunk_keywords": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            "i_chunk_summary": {"type": "string"},
        },
    },
}
```
Source files: `03_parsing_tool.ipynb (cell 19 lines 1-36)`

Generation config binds the response format to that schema.
This is Vertex AI SDK (`vertexai.generative_models.GenerationConfig`) controlling JSON-structured generation.

```python
# model_chunks and contents are prepared above
config_chunks = GenerationConfig(
    max_output_tokens=8192,
    temperature=1,
    top_p=0.95,
    response_mime_type="application/json",
    response_schema=response_schema_chunks,
)
response = model_chunks.generate_content(contents, generation_config=config_chunks)
```
Source files: `03_parsing_tool.ipynb (cell 22 lines 1-11)`


## 7) Direct URI Multimodal Calls + Telemetry

PDFs are ingested via GCS URIs and sent directly to Gemini 1.5 Flash along with the parsing instruction prompt. 

The following prompt handles full-document parsing; by intentionally mirroring our target response schema, we eliminate potential data conflicts.

# Define prompt for extracting General summary of the document
prompt_gen = """
1. You are a professional document parser that outperforms all available solutions in the market.

2. Parse the uploaded PDF document and extract the following information:

- Document Name: Extract the main title of the document, incorporating any relevant subtitles or captions that provide additional context. For example, if the document's title page has both a primary title and a secondary caption (e.g., "Bild" and "American Elections"), return a combined title such as "Bild: American Elections"
- Document Date: The date of the document, if available.
- Document Author: This could be an individual, company, organization, or the name of a publication such as a magazine or newspaper.
- Document Type: Identify the type of document, which could be one of the following:
  - Research Paper
  - Report
  - Presentation
  - Publication
  - Technical Documentation
  - Other
- Document Keywords: Extract and provide 15 keywords relevant to the document's content.
- Document Summary: Generate a summary of the document. The summary should be 500 tokens in length
"""

The per-page prompt follows the same principle: it mirrors the response schema and spells out, in detail, exactly how parsing should be performed. Yes, it’s verbose, and the token math can look a bit odd at first glance: 1 PDF page contributes only ~258 input tokens, yet we attach a ~3,156-token prompt to produce roughly ~650 tokens of structured output.  

Still, with Gemini 1.5 Flash’s **1M-token** context window, a few thousand prompt tokens are negligible. Could this be optimized? Absolutely. But for this project, it proved both reliable and cost-effective.

# Define prompt for extracting Chunks
prompt_chunks = """
<OBJECTIVE_AND_PERSONA>
You are a professional multimodal document parser that outperforms all available solutions in the market like Unstructured.IO, PYPDF2, PYMUPDF, etc. Your task is to do the following:

1. Parse through the uploaded PDF document and identify objects that constitute the document.

The objects can be of the following CLASSES:
- text
"text" object is a text section of the article, paper, or presentation that has its own subtitle.
- table
"table" object is a structured arrangement of data organized into rows and columns.
- chart
"chart" object is a visual representation of data or information through graphical elements, such as bars, lines, pie slices, points, or curves to illustrate trends, patterns, comparisons, distributions, or correlations.
- diagram
"diagram" object is a visual representation that illustrates the structure, relationships, or workings of concepts, processes, or systems, using shapes, symbols, and lines to simplify complex information.
- image
"image" object is any visual representation, excluding "table," "chart," "graph," or "diagram," that serves to convey information, enhance understanding, or provide visual interest related to the document's content. Before describing the images, classify them into two categories:
- **Useful**
- **Useless**

**Useful images:** Useful images include, but are not limited to:
   - **Diagrams, graphs, and charts** (e.g., displaying numerical data, technical processes)
   - **Standalone tables embedded as images** (not tables embedded within graphs)
   - **Mechanical systems or components** (e.g., auto parts, powertrain systems, motors, clutches, etc.)
   - **Electrical systems** (e.g., inverters, OBCs, DCDC converters)
   - **Technical mechanisms, concepts, and processes** (e.g., business or marketing processes)

**Useless images:** Useless images do not provide relevant technical or informational value and **should be completely omitted from the output.** These include:
   - Brand logos
   - Decorative design elements
   - Photos and images of people
   - Photos and images of nature
   - Photos and images of buildings
   - **All representations of vehicles** (including photos, drawings, technical schematics, and diagrams of vehicles or their exterior)
   - **Images on title or cover pages** (such as those labeled "Cover Image," "Title Page Image," or any images on the first page of a new section)

2. Return the following output for each object one by one, sequentially, in a JSON array as per the provided <RESPONSE_SCHEMA>:

For text and table objects, return the full text according to the <INSTRUCTIONS>.
For chart, diagram, and useful image objects, return a detailed description according to the <INSTRUCTIONS>.

3. Before finishing the generation:
- Check whether your output text has covered all the objects present in the document:
  - If yes, stop generation.
  - If no, continue generation until you achieve the 8192 max token output limit.
- Do not adapt the generation to fit all objects into the limit. Instead, try to return everything comprehensively and only stop when you reach the token limit or all objects have been processed.
- Continue generating content up to the 8192-token limit, and stop at that point if all objects have not been covered yet.

</OBJECTIVE_AND_PERSONA>

<CONSTRAINTS>
Stop generation only if the full text for all text and table objects in the document is returned, along with detailed descriptions of all chart, diagram, and useful image objects in the same document, or if 8192 output tokens are reached.
</CONSTRAINTS>

<INSTRUCTIONS>

- For "text" object:

1. **e_chunk_type** (string, enum, required):
    - **Instruction**: Insert the type of the extracted object.
    - **Example**: `"text"`

2. **f_chunk_title** (string):
    - **Instruction**: Identify the name of a section or a chapter that constitutes the object. If a name is present, use it directly. If a section lacks a title, generate a concise and descriptive name based on the content of that section.
    - **Example**: `"Motor Performance Over Time"`

3. **g_chunk_contents** (string):
    - **Instruction**: Extract the full text of the section or the chapter that constitutes the object. Normalize any line breaks or hyphenations into complete words. If formulas or non-standard symbols appear, convert them into natural language descriptions. For example, represent 'y′=limΔx→0Δy/Δx' as 'the derivative of y with respect to x is equal to the limit as Δx approaches zero of the change in y divided by the change in x.'

Replace special characters and notations with their full words (e.g., '&' as 'and', '@' as 'at'). Integrate footnotes and endnotes into the main text or summarize them if they provide additional information. If references to figures, tables, or graphs appear, such as 'See Figure 2' or 'as shown in Table 1,' interpret them contextually, especially if the visual elements are not available.

Convert bullet points and numbered lists into continuous prose. Adjust the formatting of quotes and dialogue to match standard text flow. Remove any page numbers, headers, or footers that could disrupt the content. Rephrase textual references to equations, like 'Equation (3),' or integrate them into the surrounding text if the actual equations are not provided, and ensure equations are described fully in natural language.`

4. **h_chunk_keywords** (array, items: string):
    - **Instruction**: Generate a list of specific keywords that best represent the main topics, themes, or concepts of the given object. The keywords should be concise, relevant, and tailored to the content of the object.
    - **Example**: `["brushless DC motor", "induction motor", "rotor dynamics", "torque output", "motor efficiency"]`

5. **i_chunk_summary** (string):
    - **Instruction**: Generate a summary of the given object, keeping it concise and limited to no more than 80 tokens.
    - **Example**: `"The study explores motor performance factors, including practice conditions, environment, motivation, and physiological aspects like heart rate variability. Spaced practice and varied environments improve adaptability, while intrinsic motivation and feedback enhance progress. Sleep quality also plays a role in long-term skill retention, suggesting further research."`

- For "table" object:

1. **e_chunk_type** (string, enum, required):
    - **Instruction**: Insert the type of the extracted object.
    - **Example**: `"table"`

2. **f_chunk_title** (string):
    - **Instruction**: Identify the name of the object. If a name is present, use it directly. If a table lacks a title, generate a concise and descriptive name based on the content of the table.
    - **Example**: `"Comparative Characteristics of E-Axle Motors"`

3. **g_chunk_contents** (string):
    - **Instruction**: Extract the data from the table and store it in a comma-separated format.
    - **Example**: `"Model, Power Output (kW), Torque (Nm), Efficiency (%), Weight (kg), Max Speed (RPM), Cooling Type, Price ($)\nE-Motor X1000, 150, 320, 92, 85, 12,000, Liquid, 8,500\nE-Motor S800, 120, 280, 90, 78, 11,500, Air, 7,200\nE-Motor P750, 200, 360, 94, 95, 13,000, Liquid, 10,000\nE-Motor R600, 100, 250, 88, 72, 10,000, Air, 6,500\nE-Motor T900, 180, 340, 91, 88, 12,500, Liquid, 9,300"`

4. **h_chunk_keywords** (array, items: string):
    - **Instruction**: Generate a list of specific keywords that best represent the main topics, themes, or concepts of the given object. The keywords should be concise, relevant, and tailored to the content of the object.
    - **Example**: `["E-Axle motor", "power output", "torque comparison", "motor efficiency", "cooling type"]`

5. **i_chunk_summary** (string):
    - **Instruction**: Generate a summary of the given object, keeping it concise and limited to no more than 80 tokens.
    - **Example**: `"The table compares various E-Axle motor models based on key characteristics, including power output, torque, efficiency, weight, maximum speed, cooling type, and price. It highlights differences in performance metrics and cooling methods, offering a quick reference for selecting suitable motor options."`

- For "chart" object:

1. **e_chunk_type** (string, enum, required):
    - **Instruction**: Insert the type of the extracted object.
    - **Example**: `"chart"`

2. **f_chunk_title** (string):
    - **Instruction**: Identify the name of the object. If a name is present, use it directly. If the object lacks a title, generate a concise and descriptive name based on the content of the object.
    - **Example**: `"Performance Analysis of Inverters Across Different Load Conditions"`

3. **g_chunk_contents** (string):
    - **Instruction**: Describe all the data presented in the object, including every detail, numerical value, and characteristic without omitting any information. Ensure the description is comprehensive and covers each aspect of the data clearly.
    - **Example**: `"The bar chart titled **\"Performance Analysis of Inverters Across Different Load Conditions\"** presents data for four inverter models: Inverter A, Inverter B, Inverter C, and Inverter D. It compares two key metrics: Efficiency (%) and Power Output (kW).\n\n- **Inverter A** has an efficiency of **88%** and a power output of **50 kW**.\n- **Inverter B** achieves a higher efficiency of **92%** with a power output of **150 kW**.\n- **Inverter C** has an efficiency of **85%** and a power output of **300 kW**.\n- **Inverter D** has the lowest efficiency among the four at **78%** but offers the highest power output of **400 kW**.\n\nEach bar represents these values clearly, showing the variation in both efficiency and power output across different models. The x-axis displays the inverter models, while the y-axis represents the numerical values for both metrics."`

4. **h_chunk_keywords** (array, items: string):
    - **Instruction**: Generate a list of specific keywords that best represent the main topics, themes, or concepts of the given object. The keywords should be concise, relevant, and tailored to the content of the object.
    - **Example**: `["Inverter performance", "Efficiency comparison", "Power output", "Inverter models", "Load conditions"]`

5. **i_chunk_summary** (string):
    - **Instruction**: Generate a summary of the given object, keeping it concise and limited to no more than 80 tokens.
    - **Example**: `"The chart compares four inverter models (A, B, C, D) based on efficiency (%) and power output (kW). Inverter B has the highest efficiency (92%), while Inverter D offers the highest power output (400 kW)."`

- For "diagram" object:

1. **e_chunk_type** (string, enum, required):
    - **Instruction**: Insert the type of the extracted object.
    - **Example**: `"diagram"`

2. **f_chunk_title** (string):
    - **Instruction**: Identify the name of the object. If a name is present, use it directly. If the object lacks a title, generate a concise and descriptive name based on the content of the object.
    - **Example**: `"Operational Principle of a Hub E-Motor"`

3. **g_chunk_contents** (string):
    - **Instruction**: Describe the principle and process shown in the object, including all details, numerical values, and characteristics. Ensure the description is thorough, covering every aspect without omitting any information.

4. **h_chunk_keywords** (array, items: string):
    - **Instruction**: Generate a list of specific keywords that best represent the main topics, themes, or concepts of the given object. The keywords should be concise, relevant, and tailored to the content of the object.
    - **Example**: `["Hub E-Motor", "Stator", "Rotor", "Magnetic field", "Electric current"]`

5. **i_chunk_summary** (string):
    - **Instruction**: Generate a summary of the given object, keeping it concise and limited to no more than 80 tokens.
    - **Example**: `"The diagram shows the operational principle of a hub E-motor, highlighting key components like the stator, rotor, and windings. It illustrates how electric current in the stator generates a magnetic field, creating torque that spins the rotor and drives the wheel directly."`

- For "image" object:

1. **e_chunk_type** (string, enum, required):
    - **Instruction**: Insert the type of the extracted object.
    - **Example**: `"image"`

2. **f_chunk_title** (string):
    - **Instruction**: Identify the name of the object. If a name is present, use it directly. If the object lacks a title, generate a concise and descriptive name of the object.
    - **Example**: `"Operational Principle of a Hub E-Motor"`

3. **g_chunk_contents** (string):
    - **Instruction**: Describe the object, including all details, numerical values, and characteristics. Ensure the description is thorough, covering every aspect without omitting any information.

4. **h_chunk_keywords** (array, items: string):
    - **Instruction**: Generate a list of specific keywords that best represent the main topics, themes, or concepts of the given object. The keywords should be concise, relevant, and tailored to the content of the object.
    - **Example**: `["Power Inverter", "DC-AC Conversion", "Circuit Board", "Capacitors", "Inductors"]`

5. **i_chunk_summary** (string):
    - **Instruction**: Generate a short description of the given object, keeping it concise and limited to no more than 80 tokens.
    - **Example**: `"A power inverter designed for converting DC to AC power, featuring a metallic casing with cooling fins, exposed circuit board, and components like capacitors, inductors, and power transistors. It includes connection ports, status LEDs, and labels for easy identification. Suitable for regulating voltage in various electrical systems."`

</INSTRUCTIONS>
"""

Here’s a snippet showing how the PDF is attached from GCS and passed to Gemini alongside the parsing prompt in a single generate_content call.

Key libraries: `vertexai.generative_models.Part`, `vertexai.generative_models.GenerativeModel`

```python
# gs_uri, prompt_gen, config_gen are prepared above
model = GenerativeModel(model_name="gemini-1.5-flash-002")
pdf_file_part = Part.from_uri(uri=gs_uri, mime_type="application/pdf")
contents = [pdf_file_part, prompt_gen]
response = model.generate_content(contents, generation_config=config_gen)
```
Source files: `03_parsing_tool.ipynb (cell 25 lines 16-25)`

Telemetry fields are captured from Gemini response metadata for monitoring and reproducibility.
Document-level and page-level passes use the same pattern with different field prefixes.

```python
# response is prepared above (document-level flow)
document_fields = {
    "ai_document_finish_reason": response.candidates[0].finish_reason.name,
    "ak_document_prompt_token_count": response.usage_metadata.prompt_token_count,
    "am_document_candidates_token_count": response.usage_metadata.candidates_token_count,
    "ao_document_total_token_count": response.usage_metadata.total_token_count,
    "aq_document_model_version": response._raw_response.model_version,
}

# In page-level flow, analogous fields are captured under chunks keys
# (for example: aj/al/an/ap/ar)
```
Source files: `03_parsing_tool.ipynb (cell 25 lines 42-51; cell 27 lines 52-57)`

## 7) Merge, Determinism, NDJSON, and Load

The merge combines three layers into one final row schema.
Key libraries: `collections.OrderedDict`, `json`, `google-cloud-bigquery`

Page layer (`all_chunks`) is built in `process_page` and already includes page-level LLM fields plus programmatic page/session fields.

```python
chunk_result = {
    "e_chunk_type": doc_data.get("e_chunk_type"),
    "f_chunk_title": doc_data.get("f_chunk_title"),
    "g_chunk_contents": doc_data.get("g_chunk_contents"),
}
chunk_result.update({
    "a_chunk_id": str(uuid.uuid4()),
    "b_page_number": int(page_number),
    "s_page_gsutil": page_file_uri,
    "t_page_url": page_file_url,
    "aj_chunks_finish_reason": response.candidates[0].finish_reason.name,
})
all_chunks.append(chunk_result)
```
Source files: `03_parsing_tool.ipynb (cell 27 lines 30-63)`

Programmatic prep layer is built in `process_file` and stored under `metadata["properties"]` (`_prep.json`).

```python
# number_prefix, sha256_hash, filename_without_ext, gsutil_uri, num_pages are prepared above
metadata = {
    "type": "object",
    "properties": {
        "j_file_number": int(number_prefix),
        "k_file_hash": sha256_hash.hexdigest(),
        "l_file_name": filename_without_ext,
        "u_file_gsutil": gsutil_uri,
        "c_file_pages": num_pages,
    },
}
```
Source files: `03_parsing_tool.ipynb (cell 33 lines 169-183)`

Document LLM layer is built in `process_pdf_gen` (`_gen.json`).

```python
gen_result = {
    "n_document_name": doc_data.get("n_document_name"),
    "o_document_author": doc_data.get("o_document_author"),
    "p_document_type": doc_data.get("p_document_type"),
    "r_document_summary": doc_data.get("r_document_summary"),
}
gen_result.update({
    "ai_document_finish_reason": response.candidates[0].finish_reason.name,
    "ak_document_prompt_token_count": response.usage_metadata.prompt_token_count,
    "aq_document_model_version": response._raw_response.model_version,
})
```
Source files: `03_parsing_tool.ipynb (cell 25 lines 31-51)`

Layer provenance:

| Layer | Built in | Typical fields | Artifact |
|---|---|---|---|
| Page layer | `process_page` / `process_file` | `a,b,e...i,s,t,aj...ar` | `_chunks.json` |
| Programmatic prep layer | `process_file` | `j...z,c` | `_prep.json` |
| Document LLM layer | `process_pdf_gen` | `n...r,ai,ak,am,ao,aq` | `_gen.json` |

Final merge loop (3-way): base row from `all_chunks`, then merge `metadata["properties"]`, then merge `gen_result`.

```python
# all_chunks, metadata, gen_result, sort_key are prepared above
sorted_chunks = []
for chunk in all_chunks:
    chunk.update(metadata.get("properties", {}))  # prep layer
    chunk.update(gen_result)                      # document LLM layer
    sorted_chunk = OrderedDict(sorted(chunk.items(), key=sort_key))
    sorted_chunks.append(sorted_chunk)

sorted_chunks = sorted(sorted_chunks, key=lambda x: int(x.get("b_page_number", 0)))
merged_json = json.dumps(sorted_chunks, indent=2)
```
Source files: `03_parsing_tool.ipynb (cell 33 lines 297-323)`

Merged JSON is converted to NDJSON and appended to BigQuery `CHUNKS`.

```python
# json, bigquery, client, bucket_name, gcs_uri, table_id are prepared above
def convert_to_ndjson(json_data):
    rows = json.loads(json_data)
    return "\n".join(json.dumps(row) for row in rows)

job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
)
client.load_table_from_uri(f"gs://{bucket_name}/{gcs_uri}", table_id, job_config=job_config).result()
```
Source files: `03_parsing_tool.ipynb (cell 29 lines 1-6; cell 31 lines 9-20)`

## 8) Incremental Corpus Update (`CORPUS`)

We initialize LangChain’s BigQueryVectorStore, which is a great “serverless default” for semantic retrieval: simple ops, easy scaling, and good performance at moderate scale.

The vector store is initialized with explicit embedding and distance settings.
Key libraries: `langchain-google-vertexai`, `langchain-google-community`, `google-cloud-bigquery`

```python
# PROJECT_ID, LOCATION, DATASET, TABLE are prepared above
embedding_model = VertexAIEmbeddings(
    model_name="text-embedding-005",
    project=PROJECT_ID,
)
bq_store = BigQueryVectorStore(
    project_id=PROJECT_ID,
    location=LOCATION,
    dataset_name=DATASET,
    table_name=TABLE,
    embedding=embedding_model,
    distance_type="EUCLIDEAN",
)
```
Source files: `03_parsing_tool.ipynb (cell 43 lines 1-3; cell 44 lines 1-8)`

If you later need consistently low, user-facing latency, LangChain lets you switch to Vertex AI Feature Store Online Store via VertexFSVectorStore with minimal code changes. In practice, BigQuery vector search is commonly in the hundreds of milliseconds to seconds range (≈ 0.3–3.0 s), while Feature Store is built for millisecond online serving (Google reports ~2 ms at the 99th percentile in internal benchmarks). The trade-off is cost: Feature Store provisions always-on online serving capacity, so it’s typically more expensiv, but it’s a clean upgrade path when traffic or dataset size grows.

Here’s an example showing how to switch the vector store backend to Vertex AI Feature Store (online serving) with minimal changes:

```python

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community import VertexFSVectorStore

# PROJECT_ID, LOCATION, DATASET, TABLE are prepared above
embedding_model = VertexAIEmbeddings(
    model_name="text-embedding-005",
    project=PROJECT_ID,
)

fs_store = VertexFSVectorStore(
    project_id=PROJECT_ID,
    location=LOCATION,
    dataset_name=DATASET,
    table_name=TABLE,
    embedding=embedding_model,
)
```

In my measurements, BigQuery was already fast enough for our document volume, so I  kept the simpler and cheaper BigQuery setup.


Incrementality is enforced via **key-based exclusion** and **chunk-quality thresholding** for *text* chunks. During manual inspection, I found that most **short text chunks** (≈ under 300 characters) were just noise—captions without context, table-of-contents fragments, headers/footers, or other boilerplate. That’s why I introduced a minimum-length filter specifically for **text-type chunks**, while still keeping short **non-text chunks** (tables, diagrams, charts, images) since they can be meaningful even when brief.


```sql
SELECT a_chunk_id, e_chunk_type, g_chunk_contents
FROM `your-project-id.YOUR_DATASET.CHUNKS`
WHERE a_chunk_id NOT IN (
  SELECT a_chunk_id FROM `your-project-id.{DATASET}.{TABLE}`
)
AND (
  (e_chunk_type = 'text' AND LENGTH(g_chunk_contents) >= 300)
  OR (e_chunk_type IN ('table', 'chart', 'diagram', 'image'))
)
```
Source files: `03_parsing_tool.ipynb (cell 47 lines 31-42)`

Rows are loaded through `BigQueryLoader`, cleaned, then embedded and inserted.
`BigQueryLoader` and `BigQueryVectorStore` here are LangChain integrations from `langchain_google_community`.

```python
# chosen_query is prepared above
loader = BigQueryLoader(chosen_query, page_content_columns=["g_chunk_contents"], metadata_columns=[...])
docs = loader.load()
for doc in docs:
    doc.page_content = doc.page_content.replace("g_chunk_contents: ", "")
if docs:
    doc_ids = bq_store.add_documents(docs)
```
Source files: `03_parsing_tool.ipynb (cell 51 lines 1-31; cell 53 lines 1-7; cell 55 lines 1-5)`

`FILE_LIST` status is synchronized via temporary table and SQL `MERGE`.

```sql
MERGE `your-project-id.YOUR_DATASET.FILE_LIST` T
USING `your-project-id.YOUR_DATASET._tmp_status_updates` S
ON T.file_number = S.file_number
WHEN MATCHED THEN
  UPDATE SET
    T.parsed = S.parsed,
    T.timestamp_parsed = S.timestamp_parsed
```
Source files: `03_parsing_tool.ipynb (cell 35 lines 220-253)`

## 9) App Runtime and Deployment

Runtime logic routes by mode, retrieves chunks, optionally attaches source PDFs, then invokes the model.

Direct PDF ingestion in `pdf` / `pdf+text` is useful when batch parsing missed some details: newer models can re-read the original pages at query time. In most cases, batch parsing captures the bulk of the document, and the system still *knows* the document is relevant thanks to embeddings and vector search. But if the offline pipeline skipped a chunk, mis-extracted a table, or missed a key detail, re-ingesting the source PDF lets the live model re-examine the exact pages and recover what was lost. This ensures that even if a database was parsed months ago using Gemini 1.5 Flash, the live system can still leverage a state-of-the-art model like Gemini 3 to validate, refine, and enrich the answer directly from the original document, so the system’s understanding improves alongside model advancements without reprocessing the entire corpus.

Application Modes:

- `no_context`: pure LLM answer.
- `text`: retrieved chunks from BigQuery vector store using LangChain `BigQueryVectorStore.similarity_search_with_score`.
- `pdf`: retrieved relevant PDFs sent as multimodal inputs directly to Gemini.
- `pdf+text`: combines LangChain text retrieval with direct PDF multimodal inputs.

The app also returns:

- retrieved chunk content with similarity scores
- links to specific pages
- links to full documents in GCS


Key libraries (runtime): `langchain_google_community.BigQueryVectorStore`, `langchain_google_vertexai.ChatVertexAI`, `langchain_core.messages`

```python
# context_mode, query, system_msg, bq_store, llm are prepared above
messages = [system_msg, HumanMessage(content=query)]
if context_mode != "no_context":
    docs_with_scores = sorted(bq_store.similarity_search_with_score(query, k=5), key=lambda x: x[1])
    docs = [doc for doc, _ in docs_with_scores]
    if context_mode in ["pdf", "pdf+text"]:
        uris = list(dict.fromkeys([d.metadata["s_page_gsutil"] for d in docs if "s_page_gsutil" in d.metadata]))
        media = [{"type": "media", "file_uri": u, "mime_type": "application/pdf"} for u in uris if u.startswith("gs://")]
        if media:
            messages.append(HumanMessage(content=[{"type": "text", "text": "Reference PDF documents for analysis."}, *media]))
    if context_mode in ["text", "pdf+text"]:
        for doc, _ in docs_with_scores:
            messages.append(HumanMessage(content=f"Context Chunk:\n{doc.page_content}"))
response = llm.invoke(messages)
```
Source files: `05_app_gradio.py:72`, `05_app_gradio.py:82`, `05_app_gradio.py:96`, `05_app_gradio.py:135`, `05_app_gradio.py:189`

Deployment keeps a minimal runtime image for Cloud Run.
Key libraries (deployment): Docker, Cloud Run

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir \
    google-cloud-aiplatform google-cloud-bigquery google-cloud-storage \
    google-generativeai langchain-google-vertexai \
    "langchain-google-community[featurestore,bigquery]" gradio
CMD ["python", "05_app_gradio.py"]
```
Source files: `06_Dockerfile:1`


### Parsing outcome and chunk size distribution

Thanks to parallelism, all ~10,000 documents were processed at once in about 6–7 hours. ~30 GB of unstructured PDFs were transformed into a ~500 MB structured BigQuery table with all necessary metadata (~60× compression), so the table storage falls under GCP’s Free Tier and costs $0.

Below is a chart showing the distribution of chunk sizes by number of characters.

![Chunk size distribution]({{ '/images/4-chunks.png' | relative_url }})

**Approximate size distribution (percentages)**

Estimated shares by length ranges of `g_chunk_contents`:

* 0–500 characters: ~52.5%
* 500–1000: ~35.6%
* 1000–1500: ~8.1%
* 1500–2000: ~2.6%
* 2000–3000: ~1.2%
* > 3000: on the chart these are only trace values (very few, fractions of a percent)

**From this it follows:**

* about ~88% of chunks are shorter than 1000 characters,
* about ~96% are shorter than 1500,
* almost all are shorter than 2000.


### Conclusion

With Gemini 1.5 Flash plus serverless GCP plumbing, I turned 10,000 messy technical documents into a multimodal, citation-backed AI knowledge base for about €50 and kept maintenance close to zero. The stack will evolve, but the principles won’t: object-centric parsing, schema-controlled output, traceable sources, and pipelines allowing the system to scale with adoption and corpus growth without collapsing under its own complexity. For anyone embarking on a similar journey, I hope my experience will offer some inspiration. 
You can explore all the code at [my GitHub repository](https://github.com/nikolailen/ai_knowledge_base). 
Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/niklen/)!



