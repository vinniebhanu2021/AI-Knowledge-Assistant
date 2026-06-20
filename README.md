# AI-Knowledge-Assistant
# Total RAG Project

This repository presents a step-by-step implementation of a Retrieval-Augmented Generation (RAG) pipeline, covering document processing, keyword and semantic retrieval, hybrid search, reranking, and advanced LLM-powered question answering

## Phases

### Phase 1: Document Processing

This phase is responsible for ingesting source documents (in PDF format) and preparing them for the subsequent phases.

- It loads PDF files from the `source_documents` directory.
- It splits the documents into smaller, manageable text chunks.
- It saves these chunks, along with their metadata, into a `chunks.json` file.

### Phase 2: Keyword Search

This phase implements a traditional keyword-based search engine.

- It uses the `chunks.json` file created in Phase 1 as its corpus.
- It employs the BM25Okapi algorithm to rank documents based on keyword relevance.
- It provides a command-line interface for users to perform keyword searches.

### Phase 3: Semantic Search

This phase introduces semantic search capabilities, allowing for searches based on meaning rather than just keywords.

- It builds a vector database (using ChromaDB) from the document chunks.
- It uses a custom embedding model (`LMStudioEmbeddings`) to generate vector representations of the text.
- It allows users to perform semantic queries to find contextually relevant information.

### Phase 4: RAG Chat

This phase builds a basic RAG chat application.

- It combines the semantic retriever from Phase 3 with a Large Language Model (LLM).
- When a user asks a question, the system retrieves relevant document chunks and passes them to the LLM as context.
- The LLM then generates an answer based on the provided context.

### Phase 5: Advanced RAG Chat

This phase enhances the RAG pipeline with a more sophisticated retrieval strategy.

- It implements a hybrid search approach, combining both keyword (BM25) and semantic search to retrieve an initial set of documents.
- It then uses a cross-encoder model to re-rank these documents for relevance.
- The re-ranked, most relevant documents are then fed to the LLM to generate a more accurate and context-aware answer.
