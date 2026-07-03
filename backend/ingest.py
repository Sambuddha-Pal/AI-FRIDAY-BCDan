import os

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    DirectoryLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from config import embeddings


# ----------------------------
# Load all documents
# ----------------------------

def load_documents():

    documents = []

    # Load CSV files
    csv_loader = DirectoryLoader(
        "knowledge_base",
        glob="*.csv",
        loader_cls=CSVLoader
    )

    documents.extend(csv_loader.load())

    # Load PDF files
    

    return documents


# ----------------------------
# Split documents
# ----------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=700,

        chunk_overlap=100

    )

    return splitter.split_documents(documents)


# ----------------------------
# Build Vector Store
# ----------------------------

def build_vector_store():

    print("Loading documents...")

    docs = load_documents()

    print(f"Loaded {len(docs)} documents")

    chunks = split_documents(docs)

    print(f"Generated {len(chunks)} chunks")

    db = FAISS.from_documents(

        chunks,

        embeddings

    )

    db.save_local("vectorstore")

    print("Vector Store Created Successfully")


if __name__ == "__main__":

    build_vector_store()