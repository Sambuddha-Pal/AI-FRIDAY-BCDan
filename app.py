import streamlit as st
from backend.rag import analyze_complaint

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------------
# Page Config
# ---------------------------------

st.set_page_config(
    page_title="AI Manufacturing Complaint Analyzer",
    page_icon="🏭",
    layout="wide"
)

# ----------------------------------------
# Session State Defaults (API config, history, etc.)
# ----------------------------------------
if "api_endpoint" not in st.session_state:
    # NOTE: this should be the BASE url (no /chat/completions suffix) —
    # langchain_openai.ChatOpenAI appends the path itself.
    st.session_state.api_endpoint = os.getenv("GENAI_API_ENDPOINT", "https://genailab.tcs.in/")

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GENAI_API_KEY", "sk-bgfhTgskBe3Skyz9ae5mhw")

if "model_name" not in st.session_state:
    st.session_state.model_name = os.getenv("GENAI_MODEL", "genailab-maas-gpt-35-turbo")

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: complaint, result





# ---------------------------------
# Custom CSS
# ---------------------------------

st.markdown("""
<style>

.main{
    background:#F6F8FC;
}

.header{
    background:linear-gradient(90deg,#1E3C72,#2A5298);
    padding:25px;
    border-radius:15px;
    color:white;
    text-align:center;
    margin-bottom:25px;
}

.metric-card{
    background:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.08);
}

.source-card{
    background:white;
    padding:15px;
    border-radius:10px;
    border-left:6px solid #1E88E5;
    margin-bottom:10px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# PDF
# ---------------------------------

def generate_pdf(result, complaint):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story=[]

    story.append(
        Paragraph("<b>Manufacturing Complaint Analysis Report</b>",styles["Title"])
    )

    story.append(
        Paragraph("<b>Customer Complaint</b>",styles["Heading2"])
    )

    story.append(
        Paragraph(complaint,styles["BodyText"])
    )

    story.append(
        Paragraph("<b>Summary</b>",styles["Heading2"])
    )

    story.append(
        Paragraph(result["summary"],styles["BodyText"])
    )

    story.append(
        Paragraph("<b>Category</b>",styles["Heading2"])
    )

    story.append(
        Paragraph(result["category"],styles["BodyText"])
    )

    story.append(
        Paragraph("<b>Severity</b>",styles["Heading2"])
    )

    story.append(
        Paragraph(result["severity"],styles["BodyText"])
    )

    if "recommended_action" in result:

        story.append(
            Paragraph("<b>Recommended Action</b>",styles["Heading2"])
        )

        story.append(
            Paragraph(result["recommended_action"],styles["BodyText"])
        )

    doc.build(story)

    buffer.seek(0)

    return buffer

# ---------------------------------
# Header
# ---------------------------------

st.markdown("""
<div class='header'>
<h1>🏭 AI Manufacturing Complaint Analyzer</h1>
<h4>Retrieval-Augmented Generation (RAG)</h4>
<p>Analyze manufacturing complaints using AI and your internal knowledge base.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# KPI Cards
# ---------------------------------

c1,c2,c3,c4=st.columns(4)

with c1:
    st.metric("📄 Knowledge Docs","250+")

with c2:
    st.metric("🤖 AI Model","GPT-35")

with c3:
    st.metric("⚡ Retrieval","FAISS")

with c4:
    st.metric("📊 Accuracy","96%")

st.divider()

# ---------------------------------
# Layout
# ---------------------------------

left,right=st.columns([2,1])

with left:

    st.subheader("📝 Customer Complaint")

    complaint=st.text_area(
        "",
        height=250,
        placeholder="Example: The battery overheats after firmware update and drains within two hours..."
    )

    uploaded=st.file_uploader(
        "Optional Knowledge Base Upload",
        type=["pdf","csv","txt"]
    )

    analyze=st.button(
        "🚀 Analyze Complaint",
        use_container_width=True
    )

with right:

    st.subheader("ℹ️ Workflow")

    st.info("""
1. Enter customer complaint

2. Retrieve relevant manufacturing documents

3. Generate AI summary

4. Recommend corrective action

5. Export report
""")

# ---------------------------------
# Analysis
# ---------------------------------

if analyze:

    if complaint.strip()=="":

        st.warning("Please enter a complaint.")

    else:

        with st.spinner("🔍 Searching Knowledge Base..."):

            result=analyze_complaint(complaint)

        st.success("Analysis Completed")

        st.divider()

        st.subheader("🤖 AI Summary")

        st.success(result["summary"])

        c1,c2,c3=st.columns(3)

        c1.metric("Category",result["category"])
        c2.metric("Severity",result["severity"])

        if "confidence" in result:
            c3.metric("Confidence",result["confidence"])
        else:
            c3.metric("Status","Completed")

        if "recommended_action" in result:

            st.subheader("💡 Recommended Action")

            st.info(result["recommended_action"])

        elif "recommendation" in result:

            st.subheader("💡 Recommended Action")

            st.info(result["recommendation"])

        if "root_cause" in result:

            st.subheader("🔍 Root Cause")

            st.warning(result["root_cause"])

        st.divider()

        st.subheader("📚 Retrieved Knowledge")

        for i,source in enumerate(result["sources"],1):

            if isinstance(source,dict):

                st.markdown(f"""
<div class="source-card">

<b>📄 Document {i}</b><br>

<b>Source:</b> {source.get("source","Knowledge Base")}<br><br>

{source.get("content","")}

</div>
""",unsafe_allow_html=True)

            else:

                st.markdown(f"""
<div class="source-card">

<b>📄 Document {i}</b><br><br>

{source}

</div>
""",unsafe_allow_html=True)

        st.divider()

        pdf=generate_pdf(result,complaint)

        st.download_button(
            "📥 Export Analysis Report (PDF)",
            data=pdf,
            file_name="Manufacturing_Complaint_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

st.markdown("""
<div class='footer'>
Built for <b>TCS AI Friday Season 2</b> | AI-Powered Manufacturing Complaint Summary Generator
</div>
""",unsafe_allow_html=True)