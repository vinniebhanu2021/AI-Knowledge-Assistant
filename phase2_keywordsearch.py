# phase2_keyword_search.py
import json
from rank_bm25 import BM25Okapi

# Load the processed chunks
with open("chunks.json", 'r') as f:
    chunks_data = json.load(f)

# Get the content of each chunk
corpus = [chunk['content'] for chunk in chunks_data]
tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

def search_keyword(query, top_n=3):
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)

    top_indexes = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_n]

    results = [chunks_data[i] for i in top_indexes]
    return results

if __name__ == "__main__":
    while True:
        query = input("Enter your keyword query (or 'exit'): ")
        if query.lower() == 'exit':
            break

        results = search_keyword(query)
        for i, res in enumerate(results):
            print(f"--- Result {i+1} (Source: {res['metadata']['source']}) ---")
            print(res['content'])
            print("\n")