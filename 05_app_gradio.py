# 05_app_gradio.py

import os
import time
import logging
import gradio as gr
import vertexai
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_community import BigQueryVectorStore, VertexFSVectorStore, BigQueryLoader
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings, ChatVertexAI
from google.cloud import storage
from google.cloud import bigquery
from langchain_core.messages import HumanMessage, SystemMessage

# Set your project & location
PROJECT_ID = "your-project-id"
LOCATION = "europe-west4"

vertexai.init(project=PROJECT_ID, location=LOCATION)

DATASET = "KNOWLEDGE_BASE"
TABLE = "CORPUS"

embedding_model = VertexAIEmbeddings(
    model_name="text-embedding-005", 
    project=PROJECT_ID
)

bq_store = BigQueryVectorStore(
    project_id=PROJECT_ID,
    location=LOCATION,
    dataset_name=DATASET,
    table_name=TABLE,
    embedding=embedding_model,
    distance_type="COSINE"
)

llm = ChatVertexAI(
    model_name="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=8192,
    timeout=None,
    max_retries=2,
)

# If you have code that must run once (e.g., initialization), put it here
storage_client = storage.Client()

def process_query(query: str, context_mode: str):
    # Add a system instruction that the model should follow before any human messages
    system_msg = SystemMessage(content="""
    You are a highly knowledgeable expert in the automotive industry.

    You have been provided with:
    1. Textual documents containing information from my knowledge database.
    2. PDF documents and their embedded elements, including text, charts, diagrams, or images.

    Your task is to accurately and comprehensively answer the user's question by following these steps in order:

    1. First, carefully read and analyze the provided textual documents. If you find a complete and reliable answer within these textual documents, use that information and conclude your answer without analyzing the PDFs.

    2. If the textual documents do not fully resolve the query, then proceed to analyze the PDF documents, including any charts, diagrams, or images embedded within them. If you find the necessary information there, or need to combine it with what you found in the text to form a complete answer, do so and then finalize your answer.

    3. If neither the provided text nor the PDF documents fully address the question, draw upon your extensive automotive domain expertise. Provide a well-grounded answer based on your own knowledge, and clearly state any assumptions. You may enrich the answer with your broader automotive expertise to ensure it is complete and contextually appropriate.

    Always ground your answer in the most reliable information available. Your final goal is to deliver a clear, complete, and contextually appropriate answer, prioritizing the given documents first and supplementing with your automotive expertise only when necessary.
""")


    if context_mode == "no_context":
        # No context mode, just use the query
        # Convert the query into a HumanMessage
        messages = [system_msg, HumanMessage(content=query)]
        response = llm.invoke(messages)
        llm_answer = response.content
        return llm_answer

    # Otherwise, run similarity search
    k = 5
    docs_with_scores = bq_store.similarity_search_with_score(query, k=k)

    # Sort docs by relevance assuming lower score = more relevant
    docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1])

    if not docs_with_scores:
        return "No relevant documents were found."

    docs = [doc for doc, score in docs_with_scores]

    # Prepare the LLM input depending on the context mode
    # Start with a HumanMessage for the user query
    messages = [system_msg, HumanMessage(content=query)]

    if context_mode in ["pdf", "pdf+text"]:
        # Extracting GCS URIs from metadata for PDFs
        gcs_uris = []
        for doc in docs:
            if "s_page_gsutil" in doc.metadata:
                gcs_uris.append(doc.metadata["s_page_gsutil"])

        # Remove duplicates while preserving order
        unique_gcs_uris = list(dict.fromkeys(gcs_uris))

        # Add PDF files as multimodal media parts so Gemini can read them as files
        pdf_media_parts = []
        for gcs_uri in unique_gcs_uris:
            if not gcs_uri.startswith("gs://"):
                continue
            pdf_media_parts.append(
                {
                    "type": "media",
                    "file_uri": gcs_uri,
                    "mime_type": "application/pdf",
                }
            )

        if pdf_media_parts:
            messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Reference PDF documents for analysis."},
                        *pdf_media_parts,
                    ]
                )
            )

    if context_mode in ["text", "pdf+text"]:
        # Add retrieved texts as additional HumanMessages
        # Truncate or limit if you have too many documents
        for doc, score in docs_with_scores:
            messages.append(HumanMessage(content=f"Context Chunk:\n{doc.page_content}"))

    # Invoke the LLM
    response = llm.invoke(messages)
    llm_answer = response.content

    # Collect metadata for output
    t_page_urls = []
    for doc in docs:
        if "t_page_url" in doc.metadata:
            t_page_urls.append(doc.metadata["t_page_url"])
    unique_t_page_urls = list(set(t_page_urls))

    v_file_urls = []
    for doc in docs:
        if "v_file_url" in doc.metadata:
            v_file_urls.append(doc.metadata["v_file_url"])
    unique_v_file_urls = list(set(v_file_urls))

    # Build markdown output
    output_lines = []
    output_lines.append("## Retrieved Documents (Most Relevant First)")
    for (doc, score) in docs_with_scores:
        chunk_id = doc.metadata.get("a_chunk_id", "N/A")
        output_lines.append(f"**Chunk ID:** {chunk_id}")
        output_lines.append(f"**Similarity Score:** {score}")
        output_lines.append("**Document Content:**")
        output_lines.append(doc.page_content)
        output_lines.append("---")

    if unique_t_page_urls:
        output_lines.append("## Page URLs")
        for url in unique_t_page_urls:
            output_lines.append(f"- [Link]({url})")

    if unique_v_file_urls:
        output_lines.append("## Full Documents")
        for url in unique_v_file_urls:
            output_lines.append(f"- [Link to Full Document]({url})")

    output_lines.append("## LLM Response")
    output_lines.append(llm_answer)

    final_output = "\n\n".join(output_lines)
    return final_output

def run_query_with_timer(query, context_mode):
    start_time = time.time()
    result = process_query(query, context_mode)
    end_time = time.time()
    elapsed = end_time - start_time
    return result, f"**Time taken:** {elapsed:.2f} seconds"

with gr.Blocks() as demo:
    gr.Markdown("# Document-based Q&A with Vertex AI")
    query = gr.Textbox(lines=2, placeholder="Enter your query here...", label="Query")
    context_mode = gr.Radio(
        ["pdf+text", "pdf", "text", "no_context"], 
        value="pdf+text", 
        label="Context Mode"
    )
    submit = gr.Button("Submit")
    with gr.Row():
        output = gr.Markdown()
    with gr.Row():
        time_output = gr.Markdown()

    submit.click(
        fn=run_query_with_timer, 
        inputs=[query, context_mode], 
        outputs=[output, time_output]
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    # Bind to 0.0.0.0 so Cloud Run can route traffic
    demo.launch(server_name="0.0.0.0", server_port=port)
