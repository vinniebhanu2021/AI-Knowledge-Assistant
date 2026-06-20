from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI

from phase3_semantic_search import LMStudioEmbeddings, api_base

# Setup from Phase 3
db = Chroma(persist_directory="chroma_db", embedding_function=LMStudioEmbeddings(api_base=api_base))
retriever = db.as_retriever(search_kwargs={"k": 3})
llm = OpenAI(name="mistralai/devstral-small-2507",base_url=api_base, api_key='lm-studio')

# Prompt Template
template = """
Answer the question based ONLY on the following context.
If you don't know the answer, just say that you don't know. Do not make up an answer.
Cite the sources used in your answer.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([f"Source: {d.metadata['source']}\n{d.page_content}" for d in docs])

# RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    while True:
        query = input("Ask a question about your documents (or 'exit'): ")
        if query.lower() == 'exit':
            break

        # Stream the answer
        for chunk in rag_chain.stream(query):
            print(chunk, end="", flush=True)
        print("\n\n")