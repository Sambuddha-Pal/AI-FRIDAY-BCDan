from langchain_community.vectorstores import FAISS

from backend.config import embeddings


# -------------------------
# Load Vector Database
# -------------------------

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)


# -------------------------
# Retrieve Similar Documents
# -------------------------

def retrieve_documents(query, k=4):

    docs = db.similarity_search(
        query,
        k=k
    )

    return docs


# -------------------------
# Retrieve Context String
# -------------------------

def retrieve_context(query):

    docs = retrieve_documents(query)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context, docs