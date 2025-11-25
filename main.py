import os
import time
import streamlit as st
import chromadb
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# ---------------------------
# Load ENV
# ---------------------------
load_dotenv()

# ---------------------------
# Chroma DB setup
# ---------------------------
CHROMA_DIR = "chroma_store"

chroma_client = chromadb.PersistentClient(CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name="news_collection",
    metadata={"hnsw:space": "cosine"}
)

# ---------------------------
# LLM Pipeline
# ---------------------------
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_length=300,
    do_sample=True,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("News Research Tool 📈")
st.sidebar.title("Enter News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
urls = [u for u in urls if u]

process_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()


# ---------------------------
# Process URLs → ChromaDB
# ---------------------------
def process_urls(urls):

    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()

    # Ensure each document has metadata
    for i, doc in enumerate(docs):
        doc.metadata["source"] = urls[i]

    # Text splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunked_docs = []

    for doc in docs:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_docs.append(
                Document(page_content=chunk, metadata=doc.metadata)
            )

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Add to ChromaDB
    for i, c in enumerate(chunked_docs):
        collection.add(
            ids=[f"chunk_{time.time()}_{i}"],
            documents=[c.page_content],
            metadatas=[c.metadata],
            embeddings=[embedding_model.embed_query(c.page_content)]
        )

    main_placeholder.success("ChromaDB updated successfully! ✅")


# ---------------------------
# Answer question using ChromaDB retrieval
# ---------------------------
def answer_question(query):

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Retrieve top 3 relevant docs
    results = collection.query(
        query_embeddings=[embedding_model.embed_query(query)],
        n_results=3
    )

    retrieved_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    retriever = lambda _: retrieved_docs

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    result = qa({"query": query})

    st.write("### Answer")
    st.write(result["result"])

    st.write("### Sources")
    for src in results["metadatas"][0]:
        st.write(f"- {src['source']}")


# ---------------------------
# Button actions
# ---------------------------
if process_clicked and urls:
    process_urls(urls)

query = st.text_input("Ask a question:")
if query:
    answer_question(query)
