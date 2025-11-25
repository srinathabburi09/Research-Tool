import os
import pickle
from django.shortcuts import render
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

from .forms import URLForm, QueryForm

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
    model="google/flan-t5-small",
    max_length = 300,
    do_sample = True,
    temperature = 0.9
    )
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ---------------------------
# Django view
# ---------------------------
def home(request):
    urls = []
    answer = None
    sources = None
    phase_messages = []
    relevant_sources = []

    url_form = URLForm()
    query_form = QueryForm()

    if request.method == "POST":
        url_form = URLForm(request.POST)
        query_form = QueryForm(request.POST)

        # ---------------------------
        # Process URLs and build FAISS
        # ---------------------------
        if "process_urls" in request.POST and url_form.is_valid():
            urls = [u for u in [
                url_form.cleaned_data.get('url_1'),
                url_form.cleaned_data.get('url_2'),
                url_form.cleaned_data.get('url_3')
            ] if u]

            if urls:
                # Load documents from URLs
                phase_messages.append("Data Loading...Started...✅✅✅")
                loader = UnstructuredURLLoader(urls=urls)
                docs = loader.load()
                for i, doc in enumerate(docs):
                    if "source" not in doc.metadata:
                        doc.metadata["source"] = urls[i]

                # Split documents into chunks while preserving metadata
                phase_messages.append("Text Splitter...Started...✅✅✅")
                chunked_docs = []
                splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                for doc in docs:
                    chunks = splitter.split_text(doc.page_content)
                    for chunk in chunks:
                        chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))

                # Create embeddings and FAISS vector store
                phase_messages.append("Embedding Vector Started Building...✅✅✅")
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorindex_hf = FAISS.from_documents(chunked_docs, embeddings)

                # Save FAISS index
                with open(FILE_PATH, "wb") as f:
                    pickle.dump(vectorindex_hf, f)

        # ---------------------------
        # Ask a question
        # ---------------------------
        elif "ask_question" in request.POST and query_form.is_valid():
            question = query_form.cleaned_data.get("question")
            

            if os.path.exists(FILE_PATH):
                with open(FILE_PATH, "rb") as f:
                    vectorindex_hf = pickle.load(f)

                # Use RetrievalQAWithSourcesChain for accurate sources
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectorindex_hf.as_retriever(search_kwargs={"k": 3}),
                    chain_type = "map_reduce",
                    return_source_documents = True 

                )

                # Run the query
                result = qa_chain({"query": question})

                # Extract answer
                answer = result["result"]

                # Extract sources (URLs + snippet of text)
                source_docs = result.get("source_documents", [])
                if source_docs:
                    relevant_sources = [
                        {
                            "url": doc.metadata.get("source", "No source"),
                            "text": doc.page_content[:300] + "..."
                        } for doc in source_docs[:1]
                    ]
            if not relevant_sources and source_docs:
                docs = source_docs[0]
                relevant_sources = [{
                    "urls": docs.metadata.get("source","No source"),
                    "text" : docs.page_content[:300] + "..."
                }]
    # ---------------------------
    # Context for template
    # ---------------------------
    context = {
        "phase_messages" : phase_messages,
        "url_form": url_form,
        "query_form": query_form,
        "answer": answer,
        "sources": relevant_sources,
    }

    return render(request, "app/home.html", context)
