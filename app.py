import streamlit as st

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Manufacturing Complaint RAG",
    page_icon="🏭",
    layout="wide"
)

# -----------------------------------
# Dummy Backend
# Replace with backend.rag.analyze()
# -----------------------------------

def analyze_complaint(complaint):

    return {
        "summary": "Battery overheating after firmware update.",
        "category": "Battery",
        "severity": "High",
        "recommendation": "Escalate to firmware engineering.",
        "sources": [
            "Defect #102 - Battery overheating after firmware update",
            "Historical Complaint #58"
        ]
    }

# -----------------------------------
# UI
# -----------------------------------

st.title("🏭 AI Customer Complaint Analyzer (RAG)")

st.write(
    "Analyze manufacturing customer complaints using Retrieval-Augmented Generation (RAG)."
)

st.divider()

complaint = st.text_area(
    "Customer Complaint",
    height=220,
    placeholder="Paste customer complaint here..."
)

uploaded_file = st.file_uploader(
    "Upload Knowledge Base (Optional)",
    type=["pdf", "csv", "txt"]
)

if st.button("Analyze Complaint", use_container_width=True):

    if complaint.strip():

        with st.spinner("Retrieving relevant documents..."):

            result = analyze_complaint(complaint)

        st.success("Analysis Complete")

        st.subheader("Summary")

        st.write(result["summary"])

        col1, col2 = st.columns(2)

        col1.metric("Category", result["category"])
        col2.metric("Severity", result["severity"])

        st.subheader("Recommended Action")

        st.info(result["recommendation"])

        st.subheader("Retrieved Documents")

        for source in result["sources"]:
            st.write(f"• {source}")

    else:
        st.warning("Please enter a complaint.")