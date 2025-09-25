import os
import pickle
import time
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
FILE_PATH = "vectorindex_hf.pkl"

# ---------------------------
# Initialize LLM pipeline globally
# ---------------------------
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # upgraded model
    max_length=500,
    do_sample=True,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Collect URLs from sidebar input
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
urls = [u for u in urls if u]  # remove empty inputs

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# ---------------------------
# Function to process URLs and build FAISS
# ---------------------------
def process_urls(urls):
    phase_messages = []
    try:
        # Load documents
        phase_messages.append("Data Loading...Started...✅")
        loader = UnstructuredURLLoader(urls=urls)
        docs = loader.load()
        for i, doc in enumerate(docs):
            if "source" not in doc.metadata:
                doc.metadata["source"] = urls[i]
        phase_messages.append("Data Loaded Successfully ✅")

        # Split documents into chunks
        phase_messages.append("Text Splitter...Started...✅")
        chunked_docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)  # increased overlap
        for doc in docs:
            chunks = splitter.split_text(doc.page_content)
            for chunk in chunks:
                chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))
        phase_messages.append(f"Text Split into {len(chunked_docs)} chunks ✅")

        # Build embeddings and FAISS vector store
        phase_messages.append("Embedding Vector Started Building...✅")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # better embeddings
        vectorindex_hf = FAISS.from_documents(chunked_docs, embeddings)

        # Save FAISS index
        with open(FILE_PATH, "wb") as f:
            pickle.dump(vectorindex_hf, f)
        phase_messages.append("FAISS Index Saved Successfully ✅")

        # Show messages in Streamlit
        for msg in phase_messages:
            main_placeholder.text(msg)
            time.sleep(0.5)

    except Exception as e:
        st.error(f"Error while processing URLs: {str(e)}")

# ---------------------------
# Function to answer question
# ---------------------------
def answer_question(query):
    if not os.path.exists(FILE_PATH):
        st.warning("FAISS index not found. Please process URLs first.")
        return

    with open(FILE_PATH, "rb") as f:
        vectorindex_hf = pickle.load(f)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorindex_hf.as_retriever(search_kwargs={"k": 3}),
        chain_type="map_reduce",
        return_source_documents=True
    )

    result = qa_chain({"query": query})
    answer = result.get("result", "")
    source_docs = result.get("source_documents", [])

    if answer:
        if source_docs:
            first_doc = source_docs[0]
            # Include context from the first source in the answer
            answer_with_context = f"{answer}\n\nContext from source:\n{first_doc.page_content[:500]}..."
            st.header("Answer")
            st.write(answer_with_context)

            # Display only the source URL
            st.subheader("Source:")
            st.markdown(f"- {first_doc.metadata.get('source', 'No source')}")
        else:
            st.header("Answer")
            st.write(answer)

# ---------------------------
# Streamlit button actions
# ---------------------------
if process_url_clicked and urls:
    process_urls(urls)

query = st.text_input("Ask a question:")
if query:
    answer_question(query)
