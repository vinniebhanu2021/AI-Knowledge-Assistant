# phase3_semantic_search.py
import json
import os
from typing import List

import requests  # Or your preferred HTTP client
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

DB_PATH = "chroma_db"


api_base = "http://26.186.178.211:1234/v1"


class LMStudioEmbeddings(Embeddings):
    def __init__(self, api_base: str = "http://localhost:1234/v1"):
        self.api_base = api_base

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Logic to call LM Studio's embedding API for multiple documents
        response = requests.post(f"{self.api_base}/embeddings", json={"input": texts})
        response.raise_for_status()
        embeddings_data = response.json()["data"]
        return [item["embedding"] for item in embeddings_data]

    def embed_query(self, text: str) -> List[float]:
        # Logic to call LM Studio's embedding API for a single query
        response = requests.post(f"{self.api_base}/embeddings", json={"input": [text]})
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


def build_or_load_db():
    # Check if the database already exists
    if os.path.exists(DB_PATH):
        print("Loading existing database.")
        return Chroma(persist_directory=DB_PATH, embedding_function=LMStudioEmbeddings(api_base))

    print("Building new database.")
    # Load the processed chunks
    with open("chunks.json", 'r') as f:
        chunks_data = json.load(f)

    # Convert dicts to Document objects
    documents = [Document(page_content=chunk['content'], metadata=chunk['metadata']) for chunk in chunks_data]

    # Create embeddings and store them in ChromaDB
    db = Chroma.from_documents(
        documents,
        LMStudioEmbeddings(api_base),
        persist_directory=DB_PATH
    )
    return db

if __name__ == "__main__":
    db = build_or_load_db()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    while True:
        query = input("Enter your semantic query (or 'exit'): ")
        if query.lower() == 'exit':
            break

        results = retriever.invoke(query)
        for i, res in enumerate(results):
            print(f"--- Result {i+1} (Source: {res.metadata['source']}) ---")
            print(res.page_content)
            print("\n")