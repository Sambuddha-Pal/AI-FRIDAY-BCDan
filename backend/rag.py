from backend.config import llm
from backend.retriever import retrieve_context
from backend.prompt import PROMPT
from backend.parser import parse_response


def analyze_complaint(complaint: str):

    # ---------------------------------
    # Retrieve Similar Documents
    # ---------------------------------

    context, source_docs = retrieve_context(complaint)

    # ---------------------------------
    # Build Prompt
    # ---------------------------------

    messages = PROMPT.format_messages(

        complaint=complaint,

        context=context

    )

    # ---------------------------------
    # Invoke LLM
    # ---------------------------------

    response = llm.invoke(messages)

    # ---------------------------------
    # Parse JSON
    # ---------------------------------

    result = parse_response(response.content)

    # ---------------------------------
    # Add Retrieved Sources
    # ---------------------------------

    result["sources"] = []

    for doc in source_docs:

        result["sources"].append({

            "source": doc.metadata.get("source", "Unknown"),

            "content": doc.page_content[:250]

        })

    return result