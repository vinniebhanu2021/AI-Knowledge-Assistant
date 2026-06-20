# phase1_process_docs.py
import os
from langchain_community.document_loaders import PyPDFLoader
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_documents(source_dir="source_documents", output_file="chunks.json"):
    all_chunks = []
    for filename in os.listdir(source_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(source_dir, filename))
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            for chunk in chunks:
                all_chunks.append({
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                })

    with open(output_file, 'w') as f:
        json.dump(all_chunks, f, indent=2)
    print(f"Successfully processed {len(all_chunks)} chunks.")

if __name__ == "__main__":
    process_documents()