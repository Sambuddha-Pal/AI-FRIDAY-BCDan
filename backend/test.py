from retriever import retrieve_context

context, docs = retrieve_context(
    "Battery drains after firmware update."
)

print(context)