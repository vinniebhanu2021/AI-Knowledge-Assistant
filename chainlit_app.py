import json
import os
import shutil
import tempfile
from typing import AsyncIterator, Dict
from typing import List

import chainlit as cl
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import asyncio

from total_rag_app import DocumentProcessor, LMStudioEmbeddings, API_BASE, CROSS_ENCODER_MODEL, LLM_NAME


class Retriever:
    def __init__(self, db_path: str, chunks_file: str, embedding_function: Embeddings):
        # The __init__ method remains synchronous as it's part of the setup
        if not os.path.exists(db_path):
            self._build_chroma_db(db_path, chunks_file, embedding_function)
        self.db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
        self.semantic_retriever = self.db.as_retriever(search_kwargs={"k": 5})

        with open(chunks_file, 'r') as f:
            self.chunks_data = json.load(f)
        corpus = [chunk['content'] for chunk in self.chunks_data]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _build_chroma_db(self, db_path: str, chunks_file: str, embedding_function: Embeddings):
        # This helper also remains synchronous
        print(f"Building new ChromaDB at {db_path}...")
        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)
        documents = [Document(page_content=chunk['content'], metadata=chunk['metadata']) for chunk in chunks_data]
        Chroma.from_documents(documents, embedding_function, persist_directory=db_path)
        print("Database built successfully.")

    def _get_keyword_docs(self, query: str, top_n: int) -> List[Dict]:
        """Synchronous helper for the CPU-bound BM25 task."""
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indexes = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_n]
        return [self.chunks_data[i] for i in bm25_top_indexes]

    # ASYNC RETRIEVE METHOD
    async def aretrieve(self, query: str, top_n_hybrid=10) -> List[dict]:
        # 1. Run CPU-bound keyword search in an executor
        keyword_docs = await asyncio.to_thread(
            self._get_keyword_docs, query, top_n_hybrid
        )

        # 2. Run I/O-bound semantic search using LangChain's async method
        semantic_docs_langchain = await self.semantic_retriever.ainvoke(query)
        semantic_docs = [{"content": doc.page_content, "metadata": doc.metadata} for doc in semantic_docs_langchain]

        # 3. Fuse results (this part is fast)
        combined_docs = {doc['content']: doc for doc in keyword_docs + semantic_docs}.values()
        return list(combined_docs)


class ReRanker:
    def __init__(self, model_name: str):
        # __init__ remains synchronous
        self.cross_encoder = CrossEncoder(model_name)

    # ASYNC RERANK METHOD ⚡️
    async def arerank(self, query: str, docs: List[dict], top_n=3) -> List[dict]:
        pairs = [[query, doc['content']] for doc in docs]

        # Run the heavy, blocking ML model prediction in an executor
        scores = await asyncio.to_thread(
            self.cross_encoder.predict, pairs
        )

        # The rest is fast and doesn't need to be wrapped
        scored_docs = list(zip(scores, docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_n]]


class RAGPipeline:
    def __init__(self, retriever: Retriever, reranker: ReRanker, llm: OpenAI):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.template = '''
Answer the question based ONLY on the following context.
If you don't know the answer, just say that you don't know. Do not make up an answer.
Cite the sources used in your answer.

Context:
{context}

Question:
{question}
'''
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def _format_context(self, docs: List[dict]) -> str:
        return " ".join([f"Source: {d['metadata']['source']}{d['content']}" for d in docs])

    # ASYNC INVOKE METHOD
    async def ainvoke(self, query: str) -> str:
        retrieved_docs = await self.retriever.aretrieve(query)
        reranked_docs = await self.reranker.arerank(query, retrieved_docs)
        context_str = self._format_context(reranked_docs)

        chain = (self.prompt | self.llm | StrOutputParser())

        answer = await chain.ainvoke({
            "context": context_str,
            "question": query
        })
        return answer

    # ASYNC STREAMING METHOD
    async def astream_answer(self, query: str) -> AsyncIterator[str]:
        """Asynchronous method to stream the answer token by token."""
        retrieved_docs = await self.retriever.aretrieve(query)
        reranked_docs = await self.reranker.arerank(query, retrieved_docs)
        context_str = self._format_context(reranked_docs)

        chain = (self.prompt | self.llm | StrOutputParser())

        # Use chain.astream() and "async for" to yield each chunk
        async for chunk in chain.astream({
            "context": context_str,
            "question": query
        }):
            yield chunk


@cl.on_chat_start
async def on_chat_start():
    # --- 1. Ask for and wait for files to be uploaded ---
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload up to 3 text or PDF files to begin!",
            accept=["text/plain", "application/pdf"],
            max_size_mb=10,  # Increased max size slightly
            max_files=3,
            timeout=300,  # Increased timeout
        ).send()

    # --- 2. Process the uploaded files ---
    # Create a message to update the user on the progress
    file_names = ", ".join([f"`{f.name}`" for f in files])
    msg = cl.Message(content=f"Processing {file_names}...")
    await msg.send()

    # Create a temporary directory to store the files
    temp_dir = tempfile.mkdtemp()

    # Use asyncio.to_thread to run the synchronous file operations in the background
    for file in files:
        # Get a unique path for the file
        dest_path = os.path.join(temp_dir, file.name)
        print(f"Copying {file.name} to {dest_path}")
        # Copy the file to the temporary directory
        await asyncio.to_thread(shutil.copy, file.path, dest_path)

    # --- 3. Run the RAG pipeline setup in steps ---
    # Wrap the document processing in a step
    async with cl.Step(name="Processing Documents", show_input=False) as step:
        step.output = "Chunking and preparing your documents..."
        doc_processor = DocumentProcessor(temp_dir, "chunks_chainlit.json")
        # Run the synchronous processing in a separate thread
        await asyncio.to_thread(doc_processor.process)

    # Wrap the RAG pipeline initialization in a step
    async with cl.Step(name="Initializing RAG Pipeline", show_input=False) as step:
        step.output = "Loading models and building the vector database..."

        # Initialize components (blocking calls) in a separate thread
        def initialize_pipeline():
            embeddings = LMStudioEmbeddings(api_base=API_BASE)
            retriever = Retriever(
                db_path="chroma_db_chainlit",
                chunks_file="chunks_chainlit.json",
                embedding_function=embeddings
            )
            reranker = ReRanker(model_name=CROSS_ENCODER_MODEL)
            llm = OpenAI(name=LLM_NAME, base_url=API_BASE, api_key='lm-studio')
            return RAGPipeline(retriever, reranker, llm)

        pipeline = await asyncio.to_thread(initialize_pipeline)
        print("RAG Application Ready.")

    # --- 4. Finalize and store the pipeline in the session ---
    # Clean up the temporary directory
    shutil.rmtree(temp_dir)

    msg.content = f"Setup complete for {file_names}. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", pipeline)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: RAGPipeline
    msg = cl.Message(content="")

    async for chunk in chain.astream_answer(message.content):
        await msg.stream_token(chunk)

    await msg.send()
