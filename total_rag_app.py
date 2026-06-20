
import os
import json
from typing import List

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder

# --- 1. Configuration ---
SOURCE_DIR = "source_documents"
CHUNKS_FILE = "chunks.json"
DB_PATH = "chroma_db"
API_BASE = "http://26.186.178.211:1234/v1"  # Replace with your LM Studio API base
LLM_NAME = "mistralai/devstral-small-2507"
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


# --- 2. Embedding Model ---
class LMStudioEmbeddings(Embeddings):
    def __init__(self, api_base: str):
        self.api_base = api_base

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(f"{self.api_base}/embeddings", json={"input": texts})
        response.raise_for_status()
        embeddings_data = response.json()["data"]
        return [item["embedding"] for item in embeddings_data]

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(f"{self.api_base}/embeddings", json={"input": [text]})
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


# --- 3. Document Processing ---
class DocumentProcessor:
    def __init__(self, source_dir: str, output_file: str):
        self.source_dir = source_dir
        self.output_file = output_file

    def process(self, chunk_size=1000, chunk_overlap=200):
        if os.path.exists(self.output_file):
            print("Chunked documents file already exists. Skipping processing.")
            return

        print("Processing source documents...")
        all_chunks = []
        for filename in os.listdir(self.source_dir):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.source_dir, filename))
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunks = text_splitter.split_documents(documents)

                for chunk in chunks:
                    all_chunks.append({
                        "content": chunk.page_content,
                        "metadata": chunk.metadata
                    })

        with open(self.output_file, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        print(f"Successfully processed and saved {len(all_chunks)} chunks to {self.output_file}.")


# --- 4. Retrieval System ---
class Retriever:
    def __init__(self, db_path: str, chunks_file: str, embedding_function: Embeddings):
        # Semantic Retriever
        if not os.path.exists(db_path):
            self._build_chroma_db(db_path, chunks_file, embedding_function)
        self.db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
        self.semantic_retriever = self.db.as_retriever(search_kwargs={"k": 5})

        # Keyword Retriever
        with open(chunks_file, 'r') as f:
            self.chunks_data = json.load(f)
        corpus = [chunk['content'] for chunk in self.chunks_data]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _build_chroma_db(self, db_path: str, chunks_file: str, embedding_function: Embeddings):
        print(f"Building new ChromaDB at {db_path}...")
        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)
        documents = [Document(page_content=chunk['content'], metadata=chunk['metadata']) for chunk in chunks_data]
        Chroma.from_documents(documents, embedding_function, persist_directory=db_path)
        print("Database built successfully.")

    def retrieve(self, query: str, top_n_hybrid=10):
        # Hybrid Search
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indexes = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_n_hybrid]
        keyword_docs = [self.chunks_data[i] for i in bm25_top_indexes]

        semantic_docs_langchain = self.semantic_retriever.invoke(query)
        semantic_docs = [{"content": doc.page_content, "metadata": doc.metadata} for doc in semantic_docs_langchain]

        # Fuse results
        combined_docs = {doc['content']: doc for doc in keyword_docs + semantic_docs}.values()
        return list(combined_docs)


# --- 5. Re-ranking System ---
class ReRanker:
    def __init__(self, model_name: str):
        self.cross_encoder = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[dict], top_n=3):
        pairs = [[query, doc['content']] for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        scored_docs = list(zip(scores, docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_n]]


# --- 6. RAG Pipeline ---
class RAGPipeline:
    def __init__(self, retriever: Retriever, reranker: ReRanker, llm: OpenAI):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.template = """
Answer the question based ONLY on the following context.
If you don't know the answer, just say that you don't know. Do not make up an answer.
Cite the sources used in your answer.

Context:
{context}

Question:
{question}
"""
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def _format_context(self, docs: List[dict]):
        return " ".join([f"Source: {d['metadata']['source']}{d['content']}" for d in docs])

    def invoke(self, query: str):
        # 1. Retrieve
        retrieved_docs = self.retriever.retrieve(query)

        # 2. Re-rank
        reranked_docs = self.reranker.rerank(query, retrieved_docs)

        # 3. Format context
        context_str = self._format_context(reranked_docs)

        # 4. Generate
        chain = (self.prompt | self.llm | StrOutputParser())
        answer = chain.invoke({
            "context": context_str,
            "question": query
        })
        return answer


# --- 7. Main Execution ---
if __name__ == "__main__":
    # Step 1: Process documents if not already done
    doc_processor = DocumentProcessor(source_dir=SOURCE_DIR, output_file=CHUNKS_FILE)
    doc_processor.process()

    # Step 2: Initialize components
    print("Initializing RAG pipeline components...")
    embeddings = LMStudioEmbeddings(api_base=API_BASE)
    retriever = Retriever(db_path=DB_PATH, chunks_file=CHUNKS_FILE, embedding_function=embeddings)
    reranker = ReRanker(model_name=CROSS_ENCODER_MODEL)
    llm = OpenAI(name=LLM_NAME, base_url=API_BASE, api_key='lm-studio')

    # Step 3: Create and run the pipeline
    pipeline = RAGPipeline(retriever, reranker, llm)
    print("RAG Application Ready. Ask a question about your documents.")

    while True:
        user_query = input("> ")
        if user_query.lower() == 'exit':
            break

        answer = pipeline.invoke(user_query)
        print("--- Answer ---")
        print(answer)
        print("--- End of Answer ---")
