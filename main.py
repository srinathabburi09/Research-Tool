import os
import time
import pickle
import streamlit as st
from dotenv import load_dotenv

# LangChain + Chroma
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# HuggingFace Pipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
PERSIST_DIR = "chromadb_store"  # folder for vector DB

# ---------------------------
# Initialize LLM pipeline globally
# ---------------------------
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_length=300,
    do_sample=True,
    temperature=0.9
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📈 News Research Tool (ChromaDB Version)")
st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
urls = [u for u in urls if u]

process_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# ---------------------------
# Function to process URLs → ChromaDB
# ---------------------------
def process_urls(url_list):
    try:
        main_placeholder.text("Loading URLs...")
        loader = UnstructuredURLLoader(urls=url_list)
        docs = loader.load()

        for i, doc in enumerate(docs):
            if "source" not in doc.metadata:
                doc.metadata["source"] = url_list[i]

        # Split documents
        main_placeholder.text("Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunked_docs = []
        for doc in docs:
            chunks = splitter.split_text(doc.page_content)
            for chunk in chunks:
                chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))

        # Embeddings
        main_placeholder.text("Generating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create ChromaDB
        main_placeholder.text("Storing vectors in ChromaDB...")
        vector_db = Chroma.from_documents(chunked_docs, embeddings, persist_directory=PERSIST_DIR)
        vector_db.persist()

        main_placeholder.success("Processing complete! Ask your question below.")

    except Exception as e:
        st.error(f"Error processing URLs: {e}")

# ---------------------------
# Function to answer question
# ---------------------------
def answer_question(query):
    if not os.path.exists(PERSIST_DIR):
        st.warning("No vector database found. Process URLs first.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        return_source_documents=True
    )

    result = qa_chain({"query": query})
    answer = result["result"]
    sources = result.get("source_documents", [])

    st.subheader("Answer:")
    st.write(answer)

    if sources:
        st.subheader("Source Context:")
        for doc in sources[:1]:
            st.markdown(f"**URL:** {doc.metadata.get('source', 'Unknown')}")  
            st.write(doc.page_content[:300] + "...")


# ---------------------------
# Streamlit actions
# ---------------------------
if process_clicked and urls:
    process_urls(urls)

query = st.text_input("Ask a question:")
if query:
    answer_question(query)
