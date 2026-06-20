# phase5_advanced_rag.py
import json

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder

from phase3_semantic_search import LMStudioEmbeddings, api_base

# Keyword Search (from Phase 2)
with open("chunks.json", 'r') as f:
    chunks_data = json.load(f)
corpus = [chunk['content'] for chunk in chunks_data]
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Semantic Search (from Phase 3)
db = Chroma(persist_directory="chroma_db", embedding_function=LMStudioEmbeddings(api_base=api_base))
semantic_retriever = db.as_retriever(search_kwargs={"k": 5})

# Re-ranker Model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# LLM (from Phase 4)
llm = OpenAI(name="mistralai/devstral-small-2507",base_url=api_base, api_key='lm-studio')

# --- 2. Advanced Retrieval Function ---
def advanced_retriever(query, top_n_hybrid=10, top_n_rerank=3):
    # Hybrid Search
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indexes = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_n_hybrid]
    keyword_docs = [chunks_data[i] for i in bm25_top_indexes]

    semantic_docs_langchain = semantic_retriever.invoke(query)
    semantic_docs = [{"content": doc.page_content, "metadata": doc.metadata} for doc in semantic_docs_langchain]

    # Fuse results (simple union and deduplication)
    combined_docs = {doc['content']: doc for doc in keyword_docs + semantic_docs}.values()

    # Re-ranking
    pairs = [[query, doc['content']] for doc in combined_docs]
    scores = cross_encoder.predict(pairs)

    # Combine docs with scores and sort
    scored_docs = list(zip(scores, combined_docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    reranked_docs = [doc for score, doc in scored_docs[:top_n_rerank]]
    return reranked_docs

# --- 3. RAG Chain ---
template = """
Answer the question based ONLY on the following context.
If you don't know the answer, say that you don't know. Cite the sources used.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_context(docs):
    return "\n\n".join([f"Source: {d['metadata']['source']}\n{d['content']}" for d in docs])

if __name__ == "__main__":
    while True:
        query = input("Ask a question (Advanced RAG) (or 'exit'): ")
        if query.lower() == 'exit':
            break

        # Run the advanced retrieval
        context_docs = advanced_retriever(query)
        context_str = format_context(context_docs)

        # Build and run the chain for this query
        chain = (prompt | llm | StrOutputParser())

        response = chain.invoke({
            "context": context_str,
            "question": query
        })
        print(response)
        print("\n")