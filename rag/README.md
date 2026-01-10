# RAG (Retrieval-Augmented Generation)

This directory implements the **Knowledge Management System** of AgConnect. While "Tools" handle structured data (prices, dates), RAG handles **unstructured knowledge** (PDF manuals, Research papers, Weather bulletins).

It ensures that the agents don't just "guess" but actually "read" trusted agronomic documents before answering.

## 🏗️ Architecture Pipeline

The RAG system follows a standard advanced pipeline:

### 1. Ingestion (`utils/ingestor.py`)
*   **What it does**: Reads raw PDF/HTML files (`sonagess_pdfs/`, `data/`).
*   **Chunking**: Splits long documents into smaller, meaningful segments (paragraphs) using `rag/utils/chunker.py` to ensure context isn't lost.

### 2. Embedding (`vector_store.py`)
*   **Model**: Uses **SBERT (Sentence-BERT)** or similar efficient models.
*   **Process**: Converts text chunks into dense numerical vectors (arrays of floats).
*   **Storage**: Stores these vectors in a local vector DB (FAISS for speed, or ChromaDB for persistence).
*   **ID Mapping**: Maintains a robust `IndexIDMap` to link vectors back to their source metadata (filename, page number).

### 3. Retrieval (`retriever.py`)
*   **Class**: `AgentRetriever`
*   **Strategy**: **Hybrid Search**.
    *   **Vector Search**: Finds documents that are *semantically* similar to the user query (e.g., "how to treat rust" matches "rust control methods").
    *   **Filtering**: Can filter results by `source_type` (e.g., only search in "Market Reports" if the intent is financial).

### 4. Re-Ranking (`re_ranker.py`)
*   **Why?**: Vector search is fast but sometimes imprecise.
*   **Component**: `CrossEncoderReranker`
*   **Process**:
    1.  The Retriever fetches the top 20 candidate documents.
    2.  The Re-ranker (a slower, more accurate model) reads these 20 candidates closely against the user's question.
    3.  It assigns a high-precision relevance score.
    4.  We keep only the **Top 5** best matches.

## 📚 Key Components

*   **`vector_store.py`**: Wrapper around **FAISS**. Handles index creation, saving, and loading. includes migration logic (`IndexIDMap` checks).
*   **`retriever.py`**: The main interface for Agents. Contains routing logic (e.g., "If Agent is Market, look in Market Docs").
*   **`re_ranker.py`**: The quality control layer. Uses models like `cross-encoder/ms-marco-MiniLM-L-6-v2`.
*   **`weather_orchestrator.py`**: A specialized RAG orchestrator for weather bulletins (Textualizing forecast data).

## 🔄 Data Flow Example

1.  **User**: "How do I fight Striga in Millet?"
2.  **Embedder**: Converts query to `[0.1, 0.4, -0.2, ...]`
3.  **Retriever (FAISS)**: Finds 20 chunks mentioning "Striga", "weeds", "parasites".
4.  **Re-Ranker**: Analysis:
    *   Chunk A ("Striga biology"): Score 0.4
    *   Chunk B ("Striga control method: pulling"): Score **0.9** (Winner!)
5.  **LLM**: Incorporates Chunk B into the final answer.

